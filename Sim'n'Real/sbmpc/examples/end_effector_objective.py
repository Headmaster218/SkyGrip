import jax
import jax.numpy as jnp
from sbmpc import BaseObjective
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse
from drone_arm_dynamics_stable import (
    MASS_TOTAL, GRAVITY,
    compute_end_effector_position,
    compute_com_offset
)

class EndEffectorTrajectoryObjective(BaseObjective):
    """
    末端执行器轨迹跟踪目标函数
    基于arm_control模式的成功经验
    """
    
    def __init__(self):
        super().__init__()
        self.nominal_hover_thrust = MASS_TOTAL * GRAVITY
        self._setup_weights()
        
    def _setup_weights(self):
        """使用arm_control模式的成功权重参数"""
        # 末端执行器跟踪权重
        self.w_ee_pos = 120.0          # 末端位置误差（主要目标）
        self.w_ee_vel = 40.0           # 末端速度
        self.w_ee_prediction = 35.0    # 末端位置预测
        
        # 基座稳定性权重（借鉴arm_control的成功经验）
        self.w_base_stability = 50.0   # 基座稳定性
        self.w_att = 50.0              # 姿态控制
        self.w_omega = 20.0            # 角速度
        
        # 关节控制权重
        self.w_joint_vel = 15.0        # 关节速度平滑性
        self.w_joint_acc = 10.0        # 关节加速度限制
        
        # 控制输入权重
        self.w_thrust = 0.001
        self.w_torque = 0.01
        self.w_joint_ctrl = 0.001
        
    def running_cost(self, state, inputs, reference):
        """
        运行成本函数
        reference格式: [ee_x, ee_y, ee_z, base_quat(4), reserved...]
        """
        # 解析状态
        pos = state[0:3]
        quat = state[3:7]
        q1, q2 = state[7], state[8]
        vel = state[9:12]
        omega = state[12:15]
        dq1, dq2 = state[15], state[16]
        
        # 解析参考（末端执行器目标位置）
        ref_ee_pos = reference[0:3]
        ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
        
        # 归一化四元数
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        
        cost = 0.0
        
        # === 1. 末端执行器位置跟踪（核心） ===
        ee_pos = compute_end_effector_position(pos, quat_normalized, q1, q2)
        ee_error = ee_pos - ref_ee_pos
        ee_error_norm = jnp.linalg.norm(ee_error)
        
        # 对小误差更敏感的成本函数
        cost += self.w_ee_pos * jnp.where(
            ee_error_norm < 0.05,
            200.0 * jnp.sum(ee_error**2),  # 接近目标时更敏感
            jnp.sum(ee_error**2)
        )
        
        # === 2. 末端执行器速度估计 ===
        # 通过雅可比矩阵估计末端速度
        # 简化版：假设主要由基座速度贡献
        ee_vel_approx = vel  # 可以后续改进为完整雅可比计算
        
        # 期望末端速度（与位置误差成比例）
        k_p_ee = 2.0
        desired_ee_vel = -k_p_ee * ee_error
        desired_ee_vel = jnp.clip(desired_ee_vel, -0.5, 0.5)
        
        ee_vel_error = ee_vel_approx - desired_ee_vel
        cost += self.w_ee_vel * jnp.sum(ee_vel_error**2)
        
        # === 3. 末端位置预测 ===
        dt_pred = 0.2
        future_ee_pos = ee_pos + ee_vel_approx * dt_pred
        future_ee_error = future_ee_pos - ref_ee_pos
        cost += self.w_ee_prediction * jnp.sum(future_ee_error**2)
        
        # === 4. 基座稳定性（借鉴arm_control成功经验） ===
        # 保持基座相对稳定，避免剧烈运动
        base_vel_penalty = jnp.where(
            jnp.linalg.norm(vel) > 0.5,
            100.0 * jnp.sum(vel**2),
            10.0 * jnp.sum(vel**2)
        )
        cost += self.w_base_stability * base_vel_penalty
        
        # === 5. 姿态控制 ===
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]
        
        att_error_norm = jnp.linalg.norm(att_error)
        cost += self.w_att * jnp.where(
            att_error_norm < 0.05,
            100.0 * jnp.sum(att_error**2),
            jnp.sum(att_error**2)
        )
        
        # === 6. 角速度控制 ===
        desired_omega = -5.0 * att_error
        omega_error = omega - desired_omega
        cost += self.w_omega * jnp.sum(omega_error**2)
        
        # === 7. 关节运动平滑性 ===
        # 惩罚过快的关节运动
        joint_vel = jnp.array([dq1, dq2])
        cost += self.w_joint_vel * jnp.sum(joint_vel**2)
        
        # 关节加速度限制（通过输入差分近似）
        if inputs.shape[0] > 4:
            joint_acc_approx = inputs[4:6]  # 关节力矩近似加速度
            cost += self.w_joint_acc * jnp.sum(joint_acc_approx**2)
        
        # === 8. 控制输入成本 ===
        # 推力控制（动态参考）
        z_error = pos[2] - 1.5  # 假设期望高度
        z_vel = vel[2]
        
        k_p_thrust = 2.0
        k_d_thrust = 1.0
        thrust_adjustment = -k_p_thrust * z_error - k_d_thrust * z_vel
        
        # 考虑重心偏移的补偿
        com_offset = compute_com_offset(q1, q2)
        gravity_compensation = MASS_TOTAL * GRAVITY * (1.0 + 0.1 * com_offset[2])
        
        expected_thrust = gravity_compensation + MASS_TOTAL * thrust_adjustment
        expected_thrust = jnp.clip(expected_thrust, 9.0, 14.0)
        
        thrust_error = inputs[0] - expected_thrust
        cost += self.w_thrust * thrust_error**2
        
        # 扭矩成本
        cost += self.w_torque * jnp.sum(inputs[1:4]**2)
        
        # 关节控制成本
        if inputs.shape[0] > 4:
            cost += self.w_joint_ctrl * jnp.sum(inputs[4:6]**2)
        
        # === 9. 约束 ===
        # 位置边界
        pos_limit = 3.0
        for i in range(3):
            cost += jnp.where(
                jnp.abs(pos[i]) > pos_limit,
                5000.0 * (jnp.abs(pos[i]) - pos_limit)**2,
                0.0
            )
        
        # 关节限制
        joint_limit = 1.6  # 从XML中的range
        cost += jnp.where(
            jnp.abs(q1) > joint_limit,
            1000.0 * (jnp.abs(q1) - joint_limit)**2,
            0.0
        )
        cost += jnp.where(
            jnp.abs(q2) > joint_limit,
            1000.0 * (jnp.abs(q2) - joint_limit)**2,
            0.0
        )
        
        # NaN保护
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        cost = jnp.where(jnp.isinf(cost), 1e6, cost)
        cost = jnp.clip(cost, 0.0, 1e6)
        
        return cost
    
    def final_cost(self, state, reference):
        """终端成本"""
        pos = state[0:3]
        quat = state[3:7]
        q1, q2 = state[7], state[8]
        vel = state[9:12]
        
        # 归一化四元数
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        
        # 计算末端执行器位置
        ee_pos = compute_end_effector_position(pos, quat_normalized, q1, q2)
        ref_ee_pos = reference[0:3]
        
        ee_error = ee_pos - ref_ee_pos
        
        # 终端成本：强调精确到达
        cost = 300.0 * jnp.sum(ee_error**2) + 50.0 * jnp.sum(vel**2)
        
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        
        return cost