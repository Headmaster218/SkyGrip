"""
无人机-机械臂系统的完整动力学模型
基于XML文件的精确参数和几何关系
"""

import jax
import jax.numpy as jnp
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse

# ============================================================================
# 系统参数（从XML文件提取）
# ============================================================================

# 质量参数
MASS_BASE = 1.0        # base_link质量
MASS_LINK1 = 0.08      # Link_1质量  
MASS_LINK2 = 0.105     # Link_2质量
TOTAL_MASS = MASS_BASE + MASS_LINK1 + MASS_LINK2  # 总质量

# 惯性参数（对角惯性矩）
INERTIA_BASE = jnp.array([0.00409, 0.0055803, 0.0094981])  # base_link
INERTIA_LINK1 = jnp.array([1.7491E-05, 2.2905E-05, 1.2e-5])  # Link_1
INERTIA_LINK2 = jnp.array([4e-5, 4.16e-5, 2.8e-5])  # Link_2

# 预计算逆矩阵（避免GPU问题）
INERTIA_BASE_INV = 1.0 / INERTIA_BASE

# 质心位置（局部坐标系）
COM_BASE_LOCAL = jnp.array([1.928E-06, 0.0086666, 0.027403])
COM_LINK1_LOCAL = jnp.array([-0.0016184, -7.0854E-06, -0.08892])
COM_LINK2_LOCAL = jnp.array([-0.00195, 0, 0.079412])

# 连接位置
LINK1_OFFSET = jnp.array([0, -2.5e-05, -0.038])  # Link_1相对base的位置
LINK2_OFFSET = jnp.array([0, 0, -0.1308])        # Link_2相对Link_1的位置

# 关节参数
JOINT_DAMPING = jnp.array([0.372, 0.235])      # 关节阻尼
JOINT_FRICTION = jnp.array([0.08, 0.203])      # 关节摩擦
JOINT_RANGE = jnp.array([[-1.6, 1.6], [-1.6, 1.6]])  # 关节范围

# 物理常数
GRAVITY = 9.81


# ============================================================================
# 辅助函数
# ============================================================================

@jax.jit
def rotation_x(angle):
    """绕X轴的旋转矩阵"""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


@jax.jit
def skew_symmetric(v):
    """向量的反对称矩阵"""
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


@jax.jit
def compute_system_com(q1, q2, R_base):
    """
    计算系统总重心位置（世界坐标系）
    
    关键：Link_2有180度初始旋转（quat=[0,-1,0,0]）
    """
    # Link_1的旋转矩阵（相对base）
    R1 = rotation_x(q1)
    
    # Link_2的初始180度旋转
    R2_init = jnp.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Link_2的总旋转（相对base）
    R2 = R1 @ R2_init @ rotation_x(q2)
    
    # 各部件质心在base坐标系中的位置
    com_base_in_base = COM_BASE_LOCAL
    com_link1_in_base = LINK1_OFFSET + R1 @ COM_LINK1_LOCAL
    
    link2_mount_in_base = LINK1_OFFSET + R1 @ LINK2_OFFSET
    com_link2_in_base = link2_mount_in_base + R2 @ COM_LINK2_LOCAL
    
    # 系统总重心（base坐标系）
    com_total_base = (
        MASS_BASE * com_base_in_base +
        MASS_LINK1 * com_link1_in_base +
        MASS_LINK2 * com_link2_in_base
    ) / TOTAL_MASS
    
    # 转换到世界坐标系
    com_total_world = R_base @ com_total_base
    
    return com_total_world, com_total_base


@jax.jit
def compute_coupling_effects(q1, q2, dq1, dq2, omega_base):
    """
    计算机械臂运动对无人机的耦合效应
    
    包括：
    1. 科里奥利力
    2. 离心力
    3. 陀螺效应
    """
    # 各连杆的角速度（body坐标系）
    omega_link1 = omega_base + jnp.array([dq1, 0, 0])
    omega_link2 = omega_base + jnp.array([dq1 + dq2, 0, 0])
    
    # 科里奥利和离心力矩
    # 这是简化模型，完整模型需要计算每个连杆的角动量
    coupling_torque = jnp.zeros(3)
    
    # Link_1的贡献
    L1_angular_momentum = INERTIA_LINK1 * omega_link1
    coupling_torque += skew_symmetric(omega_base) @ L1_angular_momentum
    
    # Link_2的贡献（考虑级联效应）
    L2_angular_momentum = INERTIA_LINK2 * omega_link2
    coupling_torque += skew_symmetric(omega_base) @ L2_angular_momentum
    
    return coupling_torque


# ============================================================================
# 主动力学函数
# ============================================================================

@jax.jit
def drone_arm_dynamics_complete(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:
    """
    完整的无人机-机械臂系统动力学
    
    状态向量（17维）：
    - pos[3]: 世界坐标系位置
    - quat[4]: 姿态四元数
    - q_joints[2]: 关节角度 [q1, q2]
    - vel[3]: 世界坐标系速度
    - omega[3]: body坐标系角速度
    - dq_joints[2]: 关节角速度 [dq1, dq2]
    
    控制输入（6维）：
    - thrust[1]: 总推力（沿body z轴）
    - torques[3]: body坐标系扭矩
    - tau_joints[2]: 关节扭矩
    """
    
    # ===== 1. 解析状态 =====
    pos = state[0:3]
    quat = state[3:7]
    q1, q2 = state[7], state[8]
    vel = state[9:12]
    omega = state[12:15]
    dq1, dq2 = state[15], state[16]
    
    # ===== 2. 解析输入 =====
    thrust = inputs[0]
    torques = inputs[1:4]
    tau1, tau2 = inputs[4], inputs[5]
    
    # ===== 3. 计算旋转矩阵 =====
    R = quat2rotm(quat)
    
    # ===== 4. 计算系统重心和偏移 =====
    com_world, com_base = compute_system_com(q1, q2, R)
    
    # 重心偏移在body坐标系中
    com_offset_base = com_base - COM_BASE_LOCAL
    
    # ===== 5. 无人机平动动力学 =====
    # 推力作用在base_link质心，但系统重心已偏移
    thrust_body = jnp.array([0., 0., thrust])
    thrust_world = R @ thrust_body
    
    # 重力作用在系统重心
    gravity_force = jnp.array([0., 0., -TOTAL_MASS * GRAVITY])
    
    # 总力
    total_force = thrust_world + gravity_force
    
    # 线加速度
    acc = total_force / TOTAL_MASS
    
    # ===== 6. 无人机转动动力学 =====
    # 基础扭矩
    total_torque = torques
    
    # 重心偏移产生的附加力矩（推力不通过重心）
    thrust_offset_torque = skew_symmetric(com_offset_base) @ thrust_body
    total_torque += thrust_offset_torque
    
    # 机械臂运动的耦合效应
    coupling_torque = compute_coupling_effects(q1, q2, dq1, dq2, omega)
    total_torque -= coupling_torque  # 反作用
    
    # 陀螺效应（base的）
    gyro_torque = jnp.array([
        (INERTIA_BASE[1] - INERTIA_BASE[2]) * omega[1] * omega[2],
        (INERTIA_BASE[2] - INERTIA_BASE[0]) * omega[2] * omega[0],
        (INERTIA_BASE[0] - INERTIA_BASE[1]) * omega[0] * omega[1]
    ])
    
    # 计算总的有效惯性（考虑机械臂的贡献）
    # 这是简化版本，完整版本需要计算整个系统的惯性张量
    R1 = rotation_x(q1)
    R2 = R1 @ jnp.diag(jnp.array([-1, -1, 1])) @ rotation_x(q2) # 包含180度初始旋转
    
    # 有效惯性矩（近似）
    I_eff = INERTIA_BASE + \
            R1 @ jnp.diag(INERTIA_LINK1) @ R1.T * 0.1 + \
            R2 @ jnp.diag(INERTIA_LINK2) @ R2.T * 0.1
    
    # 使用对角近似避免矩阵求逆
    I_eff_diag = jnp.array([I_eff[0,0], I_eff[1,1], I_eff[2,2]])
    I_eff_inv = 1.0 / I_eff_diag
    
    # 角加速度
    alpha = I_eff_inv * (total_torque - gyro_torque)
    
    # ===== 7. 关节动力学 =====
    # 考虑重力、科里奥利力、阻尼和摩擦
    
    # 重力矩（机械臂在重力场中）
    # Link_1重力矩
    R1_world = R @ rotation_x(q1)
    com_link1_world = R @ (LINK1_OFFSET + rotation_x(q1) @ COM_LINK1_LOCAL)
    g_vec = jnp.array([0, 0, -GRAVITY])
    tau_g1 = MASS_LINK1 * jnp.cross(com_link1_world, g_vec)[0]  # 只取X分量
    
    # Link_2重力矩（更复杂，因为受两个关节影响）
    link2_mount = LINK1_OFFSET + rotation_x(q1) @ LINK2_OFFSET
    R2_full = rotation_x(q1) @ jnp.diag(jnp.array([1, -1, -1])) @ rotation_x(q2)
    com_link2_world = R @ (link2_mount + R2_full @ COM_LINK2_LOCAL)
    tau_g2 = MASS_LINK2 * jnp.cross(com_link2_world, g_vec)[0]
    
    # 有效惯量（关于关节轴）
    I_joint1 = INERTIA_LINK1[0] + MASS_LINK1 * (0.08**2) + \
               INERTIA_LINK2[0] + MASS_LINK2 * (0.15**2)  # 近似值
    I_joint2 = INERTIA_LINK2[0] + MASS_LINK2 * (0.08**2)
    
    # 关节加速度（考虑所有效应）
    ddq1 = (tau1 - tau_g1 - JOINT_DAMPING[0]*dq1 - 
            JOINT_FRICTION[0]*jnp.sign(dq1)) / I_joint1
    
    ddq2 = (tau2 - tau_g2 - JOINT_DAMPING[1]*dq2 - 
            JOINT_FRICTION[1]*jnp.sign(dq2)) / I_joint2
    
    # ===== 8. 四元数导数 =====
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # ===== 9. 组合状态导数 =====
    state_dot = jnp.concatenate([
        vel,                    # 位置导数
        quat_dot,              # 四元数导数
        jnp.array([dq1, dq2]), # 关节角度导数
        acc,                   # 速度导数
        alpha,                 # 角速度导数
        jnp.array([ddq1, ddq2]) # 关节角速度导数
    ])
    
    return state_dot


# ============================================================================
# 简化版本（用于快速测试）
# ============================================================================

@jax.jit
def drone_arm_dynamics_simplified(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:
    """
    简化的动力学模型（忽略一些耦合项，但保留主要效应）
    """
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q_joints = state[7:9]
    vel = state[9:12]
    omega = state[12:15]
    dq_joints = state[15:17]
    
    # 解析输入
    thrust = inputs[0]
    torques = inputs[1:4]
    tau_joints = inputs[4:6]
    
    # 旋转矩阵
    R = quat2rotm(quat)
    
    # ===== 简化的平动动力学 =====
    thrust_world = R @ jnp.array([0., 0., thrust])
    gravity = jnp.array([0., 0., -TOTAL_MASS * GRAVITY])
    acc = (thrust_world + gravity) / TOTAL_MASS
    
    # ===== 简化的转动动力学 =====
    # 只考虑主要的陀螺效应
    gyro_torque = jnp.array([
        (INERTIA_BASE[1] - INERTIA_BASE[2]) * omega[1] * omega[2],
        (INERTIA_BASE[2] - INERTIA_BASE[0]) * omega[2] * omega[0],
        (INERTIA_BASE[0] - INERTIA_BASE[1]) * omega[0] * omega[1]
    ])
    
    # 角加速度（使用base惯性）
    alpha = INERTIA_BASE_INV * (torques - gyro_torque)
    
    # ===== 简化的关节动力学 =====
    # 简单的二阶系统
    joint_inertias = jnp.array([0.001, 0.001])  # 简化的关节惯量
    ddq_joints = (tau_joints - JOINT_DAMPING * dq_joints) / joint_inertias
    
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
    
    return state_dot


# ============================================================================
# 动力学选择器
# ============================================================================

def get_dynamics_function(use_simplified=False):
    """
    返回动力学函数
    
    Args:
        use_simplified: 是否使用简化版本（更快但精度较低）
    """
    if use_simplified:
        return drone_arm_dynamics_simplified
    else:
        return drone_arm_dynamics_complete