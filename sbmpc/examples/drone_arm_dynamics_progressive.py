"""
drone_arm_dynamics_progressive.py
渐进式动力学模型 - 从简单到复杂
"""

import jax
import jax.numpy as jnp
from sbmpc.geometry import quat_product, quat2rotm

# 系统参数（从XML文件精确读取）
MASS_BASE = 1.0
MASS_LINK1 = 0.08  
MASS_LINK2 = 0.105
MASS_TOTAL = MASS_BASE + MASS_LINK1 + MASS_LINK2  # 1.185 kg

# 质心偏移（重要！）
COM_BASE = jnp.array([1.928E-06, 0.0086666, 0.027403])  # base_link质心
COM_LINK1 = jnp.array([-0.0016184, -7.0854E-06, -0.08892])  # Link_1质心
COM_LINK2 = jnp.array([-0.00195, 0, 0.079412])  # Link_2质心

# 计算系统总质心（简化：假设关节在0位置）
LINK1_POS = jnp.array([0, -2.5e-05, -0.038])  # Link_1相对base的位置
LINK2_POS = LINK1_POS + jnp.array([0, 0, -0.1308])  # Link_2相对base的位置

# 系统总质心（在base坐标系中）
COM_SYSTEM = (MASS_BASE * COM_BASE + 
              MASS_LINK1 * (LINK1_POS + COM_LINK1) + 
              MASS_LINK2 * (LINK2_POS + COM_LINK2)) / MASS_TOTAL

print(f"System COM offset: {COM_SYSTEM}")  # 调试输出

GRAVITY = 9.81
INERTIA_BASE = jnp.array([0.00409, 0.0055803, 0.0094981])
INERTIA_BASE_INV = 1.0 / INERTIA_BASE

# ============================================================================
# 步骤1：最简化动力学（验证MPPI）
# ============================================================================
@jax.jit
def drone_arm_dynamics_step1(state, inputs, params):
    """
    修正版本 - 考虑质心偏移
    """
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q_joints = state[7:9]
    vel = state[9:12]
    omega = state[12:15]
    dq_joints = state[15:17]
    
    # 归一化四元数
    quat_norm = jnp.linalg.norm(quat)
    quat = jnp.where(
        quat_norm > 0.01,
        quat / quat_norm,
        jnp.array([1.0, 0.0, 0.0, 0.0])
    )
    
    # 解析输入并限制
    thrust = jnp.clip(inputs[0], 9.0, 14.0)  # 调整范围
    torques = jnp.clip(inputs[1:4], -0.1, 0.1)  # 减小扭矩
    tau_joints = inputs[4:6] * 0.0  # 关节锁定
    
    # === 平动动力学（考虑质心偏移）===
    R = quat2rotm(quat)
    
    # 推力作用在机体原点，但质心有偏移
    # 这会产生额外的力矩
    thrust_body = jnp.array([0., 0., thrust])
    thrust_world = R @ thrust_body
    
    # 重力作用在真实质心
    gravity_force = jnp.array([0., 0., -MASS_TOTAL * GRAVITY])
    
    # 考虑质心偏移的力矩（推力不过质心）
    com_offset_body = COM_SYSTEM  # 在body坐标系
    thrust_induced_torque = jnp.cross(com_offset_body, thrust_body) * 0.1  # 缩放因子
    
    # 阻尼力
    drag_linear = 3.0  # 增加线性阻尼
    drag_quadratic = 0.3  # 二次阻尼
    drag_force = -drag_linear * vel - drag_quadratic * vel * jnp.abs(vel)
    
    # 总加速度
    acc = (thrust_world + gravity_force + drag_force) / MASS_TOTAL
    acc = jnp.clip(acc, -5.0, 5.0)
    
    # === 转动动力学 ===
    angular_damping = 8.0  # 增加角阻尼
    
    # 姿态恢复力矩
    kp_attitude = 1.0  # 增加姿态恢复
    attitude_error = jnp.array([
        jnp.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 
                   1 - 2*(quat[1]**2 + quat[2]**2)),  # roll
        jnp.arcsin(jnp.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)),  # pitch
        0.0  # yaw
    ])
    restoration_torque = -kp_attitude * attitude_error
    
    # 总扭矩（包括质心偏移引起的）
    total_torque = torques + restoration_torque + thrust_induced_torque
    
    alpha = INERTIA_BASE_INV * (total_torque - angular_damping * omega)
    alpha = jnp.clip(alpha, -2.0, 2.0)
    
    # 关节固定
    ddq_joints = jnp.zeros(2)
    
    # 四元数导数
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # 组合状态导数
    state_dot = jnp.concatenate([
        vel,
        quat_dot,
        dq_joints,
        acc,
        alpha,
        ddq_joints
    ])
    
    # NaN保护
    state_dot = jnp.nan_to_num(state_dot, nan=0.0, posinf=0.0, neginf=0.0)
    
    return state_dot

# ============================================================================
# 步骤2：添加陀螺效应和基础关节动力学
# ============================================================================
@jax.jit
def drone_arm_dynamics_step2(state, inputs, params):
    """
    中等复杂度：
    - 添加陀螺效应
    - 简单关节动力学（无耦合）
    - 基础重力补偿
    """
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q_joints = state[7:9]
    vel = state[9:12]
    omega = state[12:15]
    dq_joints = state[15:17]
    
    # 归一化四元数
    quat = quat / (jnp.linalg.norm(quat) + 1e-10)
    
    # 输入限制
    thrust = jnp.clip(inputs[0], 0.0, 20.0)
    torques = jnp.clip(inputs[1:4], -0.5, 0.5)
    tau_joints = jnp.clip(inputs[4:6], -0.5, 0.5)
    
    # 平动动力学
    R = quat2rotm(quat)
    thrust_world = R @ jnp.array([0., 0., thrust])
    gravity_force = jnp.array([0., 0., -MASS_TOTAL * GRAVITY])
    
    drag_coeff = 0.3
    acc = (thrust_world + gravity_force) / MASS_TOTAL - drag_coeff * vel
    acc = jnp.clip(acc, -10.0, 10.0)
    
    # 转动动力学（添加陀螺效应）
    gyro_torque = jnp.array([
        (INERTIA_BASE[1] - INERTIA_BASE[2]) * omega[1] * omega[2],
        (INERTIA_BASE[2] - INERTIA_BASE[0]) * omega[2] * omega[0],
        (INERTIA_BASE[0] - INERTIA_BASE[1]) * omega[0] * omega[1]
    ])
    
    angular_damping = 1.5
    alpha = INERTIA_BASE_INV * (torques - gyro_torque - angular_damping * omega)
    alpha = jnp.clip(alpha, -5.0, 5.0)
    
    # 简单关节动力学
    JOINT_DAMPING = jnp.array([0.5, 0.5])
    JOINT_INERTIA = jnp.array([0.01, 0.01])
    
    # 简单重力补偿（固定值）
    gravity_compensation = jnp.array([0.1, 0.05]) * jnp.sin(q_joints)
    
    ddq_joints = (tau_joints - gravity_compensation - JOINT_DAMPING * dq_joints) / JOINT_INERTIA
    ddq_joints = jnp.clip(ddq_joints, -5.0, 5.0)
    
    # 四元数导数
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # 组合
    state_dot = jnp.concatenate([
        vel,
        quat_dot,
        dq_joints,
        acc,
        alpha,
        ddq_joints
    ])
    
    # NaN保护
    state_dot = jnp.where(jnp.isnan(state_dot), 0.0, state_dot)
    state_dot = jnp.where(jnp.isinf(state_dot), 0.0, state_dot)
    
    return state_dot