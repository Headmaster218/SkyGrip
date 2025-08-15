"""
稳定的无人机-机械臂动力学模型
改进版：解决悬停漂移问题
"""

import jax
import jax.numpy as jnp
from sbmpc.geometry import quat_product, quat2rotm

# 系统参数（从XML精确提取）
MASS_BASE = 1.0
MASS_LINK1 = 0.08  
MASS_LINK2 = 0.105
MASS_TOTAL = MASS_BASE + MASS_LINK1 + MASS_LINK2

# 惯性参数
INERTIA_BASE = jnp.array([0.00409, 0.0055803, 0.0094981])
INERTIA_BASE_INV = 1.0 / (INERTIA_BASE + 1e-10)

# 质心位置
COM_BASE = jnp.array([1.928E-06, 0.0086666, 0.027403])
COM_LINK1 = jnp.array([-0.0016184, -7.0854E-06, -0.08892])
COM_LINK2 = jnp.array([-0.00195, 0, 0.079412])

# 连接位置
LINK1_POS = jnp.array([0, -2.5e-05, -0.038])
LINK2_POS = jnp.array([0, 0, -0.1308])

# 关节参数（从XML）
JOINT_DAMPING = jnp.array([0.372, 0.235])
JOINT_FRICTION = jnp.array([0.08, 0.203])

GRAVITY = 9.81

# ============================================================================
# 步骤1：改进的基础动力学（解决漂移问题）
# ============================================================================
@jax.jit
def dynamics_step1(state, inputs, params):
    """
    步骤1：平衡的基础动力学
    - 适中的阻尼系数
    - 物理真实且稳定
    """
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q_joints = state[7:9]
    vel = state[9:12]
    omega = state[12:15]
    dq_joints = state[15:17]
    
    # 四元数归一化
    quat_norm = jnp.linalg.norm(quat)
    quat = jnp.where(
        quat_norm > 0.01,
        quat / quat_norm,
        jnp.array([1.0, 0.0, 0.0, 0.0])
    )
    
    # 输入限制
    thrust = jnp.clip(inputs[0], 5.0, 20.0)
    torques = jnp.clip(inputs[1:4], -1.0, 1.0)
    tau_joints = jnp.clip(inputs[4:6], -0.5, 0.5)
    
    # === 平动动力学 ===
    R = quat2rotm(quat)
    
    # 推力向量
    thrust_body = jnp.array([0., 0., thrust])
    thrust_world = R @ thrust_body
    
    # 重力
    gravity_force = jnp.array([0., 0., -MASS_TOTAL * GRAVITY])
    
    # 平衡的线性阻尼（不太大也不太小）
    linear_damping_coeff = 0.5  # 平衡值
    drag_force = -linear_damping_coeff * vel
    
    # 加速度计算
    acc = (thrust_world + gravity_force + drag_force) / MASS_TOTAL
    
    # 合理的加速度限制
    acc_mag = jnp.linalg.norm(acc)
    acc = jnp.where(
        acc_mag > 20.0,
        acc * (20.0 / acc_mag),
        acc
    )
    
    # === 转动动力学 ===
    # 陀螺效应
    gyro_torque = jnp.cross(omega, INERTIA_BASE * omega)
    
    # 平衡的角阻尼
    angular_damping_coeff = 2.0  # 平衡值
    damping_torque = -angular_damping_coeff * omega
    
    # 角加速度
    alpha = INERTIA_BASE_INV * (torques - gyro_torque + damping_torque)
    
    # 合理的角加速度限制
    alpha_mag = jnp.linalg.norm(alpha)
    alpha = jnp.where(
        alpha_mag > 15.0,
        alpha * (15.0 / alpha_mag),
        alpha
    )
    
    # === 关节动力学 ===
    joint_inertia = jnp.array([0.005, 0.003])
    
    # 重力补偿
    gravity_torque = jnp.array([
        -0.5 * MASS_LINK1 * GRAVITY * jnp.sin(q_joints[0]),
        -0.3 * MASS_LINK2 * GRAVITY * jnp.sin(q_joints[1])
    ])
    
    # 关节加速度
    ddq_joints = (tau_joints + gravity_torque - 
                  JOINT_DAMPING * dq_joints - 
                  JOINT_FRICTION * jnp.sign(dq_joints)) / joint_inertia
    
    ddq_joints = jnp.clip(ddq_joints, -5.0, 5.0)
    
    # === 四元数导数 ===
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # === 组合状态导数 ===
    state_dot = jnp.concatenate([
        vel,
        quat_dot,
        dq_joints,
        acc,
        alpha,
        ddq_joints
    ])
    
    # 数值稳定性保护
    state_dot = jnp.nan_to_num(state_dot, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 限制状态导数幅度
    state_dot_mag = jnp.linalg.norm(state_dot)
    state_dot = jnp.where(
        state_dot_mag > 50.0,
        state_dot * (50.0 / state_dot_mag),
        state_dot
    )
    
    return state_dot

# ============================================================================
# 步骤2：添加重心补偿（改进版）
# ============================================================================
@jax.jit
def compute_com_offset(q1, q2):
    """计算由于关节运动引起的重心偏移"""
    # Link1旋转矩阵（绕X轴）
    c1, s1 = jnp.cos(q1), jnp.sin(q1)
    R1 = jnp.array([
        [1, 0, 0],
        [0, c1, -s1],
        [0, s1, c1]
    ])
    
    # Link2级联旋转
    c2, s2 = jnp.cos(q2), jnp.sin(q2)
    R2_local = jnp.array([
        [1, 0, 0],
        [0, c2, -s2],
        [0, s2, c2]
    ])
    R2 = R1 @ R2_local
    
    # 计算各部分质心位置
    com_link1 = LINK1_POS + R1 @ COM_LINK1
    com_link2 = LINK1_POS + R1 @ (LINK2_POS + R2_local @ COM_LINK2)
    
    # 系统总重心
    com_total = (MASS_BASE * COM_BASE + 
                 MASS_LINK1 * com_link1 + 
                 MASS_LINK2 * com_link2) / MASS_TOTAL
    
    return com_total - COM_BASE  # 返回相对偏移

@jax.jit
def dynamics_step2(state, inputs, params):
    """
    步骤2：添加重心补偿（修正版）
    - 保持与step1一致的阻尼系数
    - 只在关节移动时激活耦合
    - 数值稳定的重心计算
    """
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q1, q2 = state[7], state[8]
    vel = state[9:12]
    omega = state[12:15]
    dq1, dq2 = state[15], state[16]
    
    # 四元数归一化
    quat_norm = jnp.linalg.norm(quat)
    quat = quat / (quat_norm + 1e-10)
    
    # 输入限制
    thrust = jnp.clip(inputs[0], 5.0, 20.0)
    torques = jnp.clip(inputs[1:4], -1.0, 1.0)
    tau1 = jnp.clip(inputs[4], -0.5, 0.5)
    tau2 = jnp.clip(inputs[5], -0.5, 0.5)
    
    # === 计算重心偏移（改进版）===
    # 只在关节角度显著时计算
    joint_threshold = 0.01  # 0.01弧度 ≈ 0.57度
    q_joints = jnp.array([q1, q2])
    joint_magnitude = jnp.linalg.norm(q_joints)
    
    # 条件激活重心补偿
    com_offset = jnp.where(
        joint_magnitude > joint_threshold,
        compute_com_offset(q1, q2),
        jnp.zeros(3)  # 关节接近零时不计算偏移
    )
    
    # === 平动动力学（与step1保持一致）===
    R = quat2rotm(quat)
    
    # 推力作用点偏移产生的力矩
    thrust_body = jnp.array([0., 0., thrust])
    thrust_world = R @ thrust_body
    
    # 重心偏移引起的额外力矩（缩放因子减小）
    offset_torque = jnp.cross(com_offset, thrust_body) * 0.1  # 从0.3减到0.1
    
    # 重力和阻尼（关键：使用与step1相同的系数）
    gravity_force = jnp.array([0., 0., -MASS_TOTAL * GRAVITY])
    linear_damping_coeff = 0.5  # 与step1保持一致！
    drag_force = -linear_damping_coeff * vel
    
    # 加速度
    acc = (thrust_world + gravity_force + drag_force) / MASS_TOTAL
    acc = jnp.clip(acc, -20.0, 20.0)  # 与step1一致
    
    # === 转动动力学（与step1保持一致）===
    # 陀螺效应
    gyro_torque = jnp.cross(omega, INERTIA_BASE * omega)
    
    # 固定角阻尼（关键：与step1一致）
    angular_damping_coeff = 2.0  # 与step1保持一致！
    damping_torque = -angular_damping_coeff * omega
    
    # 机械臂运动耦合（只在关节速度显著时激活）
    joint_vel_magnitude = jnp.sqrt(dq1**2 + dq2**2)
    joint_vel_threshold = 0.01  # rad/s
    
    arm_coupling = jnp.where(
        joint_vel_magnitude > joint_vel_threshold,
        jnp.array([
            dq1 * 0.002 + dq1 * dq2 * 0.001,
            dq2 * 0.002 + dq1 * dq2 * 0.001,
            (dq1**2 + dq2**2) * 0.001
        ]),
        jnp.zeros(3)  # 关节速度很小时不耦合
    )
    
    # 总力矩
    total_torque = torques - gyro_torque + damping_torque - offset_torque - arm_coupling
    alpha = INERTIA_BASE_INV * total_torque
    alpha = jnp.clip(alpha, -15.0, 15.0)  # 与step1一致
    
    # === 改进的关节动力学 ===
    # 更准确的重力矩计算（但只在关节角度显著时）
    g_tau1 = jnp.where(
        jnp.abs(q1) > joint_threshold,
        -(MASS_LINK1 * 0.089 + MASS_LINK2 * 0.22) * GRAVITY * jnp.sin(q1),
        0.0
    )
    g_tau2 = jnp.where(
        jnp.abs(q1 + q2) > joint_threshold,
        -MASS_LINK2 * 0.079 * GRAVITY * jnp.sin(q1 + q2),
        0.0
    )
    
    # 惯性矩阵（简化对角形式）
    joint_inertia = jnp.array([0.008, 0.005])
    
    # 关节加速度
    ddq1 = (tau1 + g_tau1 - JOINT_DAMPING[0]*dq1 - 
            JOINT_FRICTION[0]*jnp.tanh(10*dq1)) / joint_inertia[0]
    ddq2 = (tau2 + g_tau2 - JOINT_DAMPING[1]*dq2 - 
            JOINT_FRICTION[1]*jnp.tanh(10*dq2)) / joint_inertia[1]
    
    ddq_joints = jnp.array([ddq1, ddq2])
    ddq_joints = jnp.clip(ddq_joints, -5.0, 5.0)  # 与step1一致
    
    # === 四元数导数 ===
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # 组合
    state_dot = jnp.concatenate([
        vel, quat_dot, 
        jnp.array([dq1, dq2]),
        acc, alpha, ddq_joints
    ])
    
    # 数值保护
    state_dot = jnp.nan_to_num(state_dot, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 限制幅度
    state_dot_mag = jnp.linalg.norm(state_dot)
    state_dot = jnp.where(
        state_dot_mag > 50.0,  # 与step1一致
        state_dot * (50.0 / state_dot_mag),
        state_dot
    )
    
    return state_dot

# 步骤3保持占位
dynamics_step3_stable = dynamics_step2  # 暂时使用步骤2

@jax.jit
def compute_end_effector_position(base_pos, base_quat, q1, q2):
    """
    计算末端执行器在世界坐标系中的位置
    
    Args:
        base_pos: 基座位置 [x, y, z]
        base_quat: 基座姿态四元数 [w, x, y, z]
        q1: Joint_1角度
        q2: Joint_2角度
    
    Returns:
        end_effector_pos: 末端执行器世界坐标 [x, y, z]
    """
    # 从XML获取的连接位置
    LINK1_OFFSET = jnp.array([0, -2.5e-05, -0.038])  # Link1相对base_link
    LINK2_OFFSET = jnp.array([0, 0, -0.1308])         # Link2相对Link1
    END_EFFECTOR_OFFSET = jnp.array([0, 0, 0.16])     # 末端相对Link2
    
    # 基座旋转矩阵
    R_base = quat2rotm(base_quat)
    
    # Joint_1旋转矩阵（绕X轴）
    c1, s1 = jnp.cos(q1), jnp.sin(q1)
    R1 = jnp.array([
        [1, 0, 0],
        [0, c1, -s1],
        [0, s1, c1]
    ])
    
    # Joint_2旋转矩阵（绕X轴，注意Link2有初始90度旋转）
    c2, s2 = jnp.cos(q2), jnp.sin(q2)
    R2_local = jnp.array([
        [1, 0, 0],
        [0, c2, -s2],
        [0, s2, c2]
    ])
    
    # Link2的初始旋转（从XML: quat="0 -1 0 0" 对应绕X轴旋转-90度）
    R2_init = jnp.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    
    # 级联变换
    link1_pos_world = base_pos + R_base @ LINK1_OFFSET
    link2_pos_world = link1_pos_world + R_base @ R1 @ LINK2_OFFSET
    
    # 末端执行器在世界坐标系中的位置
    R2_combined = R2_init @ R2_local
    end_effector_local = R2_combined @ END_EFFECTOR_OFFSET
    end_effector_world = link2_pos_world + R_base @ R1 @ end_effector_local
    
    return end_effector_world