#!/usr/bin/env python3
"""
优化的MPPI控制器 - 改善漂移问题
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import time

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_all
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse, skew

# GPU设置
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
jax.config.update("jax_default_matmul_precision", "high")

# 系统参数
MASS_BASE = 1.0
MASS_LINK1 = 0.08
MASS_LINK2 = 0.105
TOTAL_MASS = MASS_BASE + MASS_LINK1 + MASS_LINK2
INERTIA_BASE = jnp.array([0.00409, 0.0055803, 0.0094981])
GRAVITY = 9.81
JOINT_DAMPING = jnp.array([0.372, 0.235])
JOINT_INERTIA = jnp.array([0.001, 0.001])

# ============================================================================
# 改进的动力学模型（增强稳定性）
# ============================================================================
@jax.jit
def drone_arm_dynamics_improved(state: jnp.array, inputs: jnp.array, params: jnp.array) -> jnp.array:
    """改进的动力学模型 - 增加阻尼和稳定性"""
    # 解析状态
    pos = state[0:3]
    quat = state[3:7]
    q_joints = state[7:9]
    vel = state[9:12]
    omega = state[12:15]
    dq_joints = state[15:17]
    
    # 归一化四元数
    quat_norm = jnp.linalg.norm(quat) + 1e-10
    quat = quat / quat_norm
    
    # 解析输入（更严格的限制）
    thrust = jnp.clip(inputs[0], 0.0, 25.0)
    torques = jnp.clip(inputs[1:4], -0.2, 0.2)  # 减小扭矩限制
    tau_joints = jnp.clip(inputs[4:6], -0.3, 0.3)  # 减小关节扭矩
    
    # 旋转矩阵
    R = quat2rotm(quat)
    
    # 平动动力学（增加线性阻尼）
    thrust_body = jnp.array([0., 0., thrust])
    thrust_world = R @ thrust_body
    gravity_force = jnp.array([0., 0., -TOTAL_MASS * GRAVITY])
    drag_force = -0.1 * vel  # 空气阻力
    acc = (thrust_world + gravity_force + drag_force) / TOTAL_MASS
    acc = jnp.clip(acc, -15.0, 15.0)  # 减小加速度限制
    
    # 转动动力学（增强阻尼）
    I = jnp.diag(INERTIA_BASE)
    I_inv = jnp.diag(1.0 / INERTIA_BASE)
    gyro_torque = skew(omega) @ I @ omega
    damping_torque = 0.05 * omega  # 增加角速度阻尼
    alpha = I_inv @ (torques - gyro_torque - damping_torque)
    alpha = jnp.clip(alpha, -5.0, 5.0)  # 减小角加速度限制
    
    # 关节动力学
    ddq_joints = (tau_joints - JOINT_DAMPING * dq_joints) / JOINT_INERTIA
    ddq_joints = jnp.clip(ddq_joints, -3.0, 3.0)
    
    # 四元数导数
    omega_quat = jnp.array([0., omega[0], omega[1], omega[2]])
    quat_dot = 0.5 * quat_product(quat, omega_quat)
    
    # 组合状态导数
    state_dot = jnp.concatenate([
        vel,           # 位置导数
        quat_dot,      # 四元数导数
        dq_joints,     # 关节角度导数
        acc,           # 速度导数
        alpha,         # 角速度导数
        ddq_joints     # 关节角速度导数
    ])
    
    # 防止NaN
    state_dot = jnp.where(jnp.isnan(state_dot), 0.0, state_dot)
    
    return state_dot

# ============================================================================
# 优化的目标函数（改善跟踪性能）
# ============================================================================
class ImprovedObjective(BaseObjective):
    """改进的目标函数 - 更好的位置跟踪"""
    
    def __init__(self):
        super().__init__()
        # 调整权重以改善位置跟踪
        self.w_pos = 100.0      # 大幅增加位置权重
        self.w_vel = 20.0       # 增加速度权重
        self.w_att = 25.0       # 姿态权重
        self.w_omega = 10.0     # 角速度权重
        self.w_joint = 10.0     # 关节权重
        self.w_joint_vel = 3.0  # 关节速度权重
        
        # 分离的控制权重
        self.w_thrust = 0.0001  # 推力权重（很小）
        self.w_torque = 0.01    # 扭矩权重
        self.w_joint_ctrl = 0.005  # 关节控制权重
        
        # 积分误差（用于消除稳态误差）
        self.pos_integral = jnp.zeros(3)
        self.integral_gain = 0.01
        
        self.nominal_hover_thrust = TOTAL_MASS * GRAVITY
    
    def running_cost(self, state, inputs, reference):
        """改进的运行成本"""
        # 解析状态
        pos = state[0:3]
        quat = state[3:7]
        q_joints = state[7:9]
        vel = state[9:12]
        omega = state[12:15]
        dq_joints = state[15:17]
        
        # 解析参考
        ref_pos = reference[0:3] if reference.shape[0] >= 3 else jnp.array([0, 0, 1.5])
        ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
        ref_joints = reference[7:9] if reference.shape[0] >= 9 else jnp.zeros(2)
        
        # 归一化四元数
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        
        cost = 0.0
        
        # 1. 位置误差（主要项）
        pos_error = pos - ref_pos
        cost += self.w_pos * jnp.sum(pos_error**2)
        
        # 2. 速度误差（希望静止）
        cost += self.w_vel * jnp.sum(vel**2)
        
        # 3. 姿态误差（保持水平）
        # 使用四元数误差
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]  # 虚部表示误差
        cost += self.w_att * jnp.sum(att_error**2)
        
        # 4. 角速度误差
        cost += self.w_omega * jnp.sum(omega**2)
        
        # 5. 关节误差
        joint_error = q_joints - ref_joints
        cost += self.w_joint * jnp.sum(joint_error**2)
        cost += self.w_joint_vel * jnp.sum(dq_joints**2)
        
        # 6. 控制代价（分离处理）
        # 推力代价（允许偏离悬停推力）
        thrust_deviation = (inputs[0] - self.nominal_hover_thrust) / self.nominal_hover_thrust
        cost += self.w_thrust * thrust_deviation**2
        
        # 扭矩代价
        cost += self.w_torque * jnp.sum(inputs[1:4]**2)
        
        # 关节控制代价
        cost += self.w_joint_ctrl * jnp.sum(inputs[4:6]**2)
        
        # 7. 障碍惩罚（防止过大倾角）
        max_tilt = 0.3  # 最大倾角（弧度）
        tilt_magnitude = jnp.sqrt(att_error[0]**2 + att_error[1]**2)
        tilt_penalty = jnp.where(
            tilt_magnitude > max_tilt,
            1000.0 * (tilt_magnitude - max_tilt)**2,
            0.0
        )
        cost += tilt_penalty
        
        # 8. 位置边界惩罚（防止漂移太远）
        boundary_limit = 3.0  # 米
        for i in range(3):
            cost += jnp.where(
                jnp.abs(pos[i] - ref_pos[i]) > boundary_limit,
                1000.0 * (jnp.abs(pos[i] - ref_pos[i]) - boundary_limit)**2,
                0.0
            )
        
        # 防止NaN和Inf
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        cost = jnp.where(jnp.isinf(cost), 1e6, cost)
        
        return cost
    
    def final_cost(self, state, reference):
        """改进的终端成本"""
        pos = state[0:3]
        quat = state[3:7]
        q_joints = state[7:9]
        vel = state[9:12]
        omega = state[12:15]
        
        ref_pos = reference[0:3] if reference.shape[0] >= 3 else jnp.array([0, 0, 1.5])
        ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
        ref_joints = reference[7:9] if reference.shape[0] >= 9 else jnp.zeros(2)
        
        # 位置误差
        pos_error = pos - ref_pos
        
        # 姿态误差
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]
        
        # 关节误差
        joint_error = q_joints - ref_joints
        
        cost = (
            50.0 * jnp.sum(pos_error**2) +    # 强调终端位置
            20.0 * jnp.sum(att_error**2) +    # 终端姿态
            10.0 * jnp.sum(joint_error**2) +  # 终端关节
            15.0 * jnp.sum(vel**2) +          # 终端速度
            5.0 * jnp.sum(omega**2)           # 终端角速度
        )
        
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        
        return cost

# ============================================================================
# 主函数
# ============================================================================
def run_optimized_controller():
    """运行优化的控制器"""
    
    print("\n" + "="*60)
    print("Optimized MPPI Controller")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print("="*60 + "\n")
    
    # 1. 配置
    robot_config = settings.RobotConfig()
    robot_config.robot_scene_path = "examples/drone_direct_control.xml"
    robot_config.nq = 9
    robot_config.nv = 8
    robot_config.nu = 6
    
    # 调整输入限制
    robot_config.input_min = jnp.array([0., -0.2, -0.2, -0.2, -0.3, -0.3])
    robot_config.input_max = jnp.array([20., 0.2, 0.2, 0.2, 0.3, 0.3])
    
    # 初始状态
    robot_config.q_init = jnp.array([
        0., 0., 1.5,      # 位置
        1., 0., 0., 0.,   # 四元数
        0., 0.            # 关节
    ], dtype=jnp.float32)
    
    config = settings.Config(robot_config)
    config.general.visualize = False  # 使用自定义可视化
    config.general.integrator_type = "rk4"  # 使用RK4以提高精度
    
    # MPPI参数（优化后）
    config.MPC.dt = 0.02
    config.MPC.horizon = 20  # 稍长的预测时域
    config.MPC.num_parallel_computations = 200  # 更多采样
    config.MPC.lambda_mpc = 50.0  # 温度参数
    
    # 减小探索噪声
    config.MPC.std_dev_mppi = jnp.array([
        0.2,                    # 推力噪声
        0.001, 0.001, 0.0005,  # 扭矩噪声（很小）
        0.002, 0.002           # 关节噪声（很小）
    ])
    
    # 初始猜测
    config.MPC.initial_guess = jnp.array([
        TOTAL_MASS * GRAVITY, 0., 0., 0., 0., 0.
    ])
    
    # 使用样条平滑（可选）
    config.MPC.smoothing = None  # 或者 "Spline"
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = False
    
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_iterations = 500  # 10秒
    
    # 2. 定义任务
    print("Task: Stable hovering at [0, 0, 1.5]")
    print("Configuration:")
    print(f"  - Horizon: {config.MPC.horizon}")
    print(f"  - Samples: {config.MPC.num_parallel_computations}")
    print(f"  - Lambda: {config.MPC.lambda_mpc}")
    print(f"  - dt: {config.MPC.dt}")
    
    # 目标状态
    target_pos = jnp.array([0.0, 0.0, 1.5])
    target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    target_joints = jnp.array([0.0, 0.0])
    
    # 创建参考状态
    ref_state = jnp.zeros(17)
    ref_state = ref_state.at[0:3].set(target_pos)
    ref_state = ref_state.at[3:7].set(target_quat)
    ref_state = ref_state.at[7:9].set(target_joints)
    
    # 创建时变参考（可选）
    reference = jnp.tile(ref_state, (config.MPC.horizon + 1, 1))
    
    # 3. 创建目标函数
    objective = ImprovedObjective()
    
    # 4. 构建仿真
    print("\nBuilding simulation...")
    sim = build_all(
        config,
        objective,
        reference,
        custom_dynamics_fn=drone_arm_dynamics_improved,
        obstacles=False
    )
    
    # 5. 创建MuJoCo可视化
    print("Setting up visualization...")
    mj_model = mujoco.MjModel.from_xml_path("examples/drone_direct_control.xml")
    mj_data = mujoco.MjData(mj_model)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    # 设置相机
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 45
    
    print("\nSimulation running...")
    print("Press Ctrl+C to stop\n")
    print("-" * 60)
    
    # 6. 运行仿真循环
    try:
        last_print_time = 0
        errors_history = []
        
        for i in range(config.sim_iterations):
            # SBMPC步进
            sim.step()
            
            # 获取当前状态
            current_state = sim.state_traj[i+1, :]
            
            # 更新MuJoCo可视化
            mj_data.qpos[0:3] = current_state[0:3]
            mj_data.qpos[3:7] = current_state[3:7]
            if mj_model.nq > 7:
                mj_data.qpos[7:9] = current_state[7:9]
            
            mj_data.qvel[0:3] = current_state[9:12]
            mj_data.qvel[3:6] = current_state[12:15]
            if mj_model.nv > 6:
                mj_data.qvel[6:8] = current_state[15:17]
            
            # 前向运动学
            mujoco.mj_forward(mj_model, mj_data)
            
            # 同步查看器
            viewer.sync()
            
            # 记录误差
            pos = current_state[0:3]
            pos_error = np.linalg.norm(pos - target_pos)
            errors_history.append(pos_error)
            
            # 打印状态（每0.5秒）
            current_time = i * config.MPC.dt
            if current_time - last_print_time >= 0.5:
                vel_mag = np.linalg.norm(current_state[9:12])
                thrust = sim.input_traj[i, 0] if i < len(sim.input_traj) else 0
                
                print(f"t={current_time:5.1f}s | "
                      f"Pos=[{pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}] | "
                      f"Err={pos_error:.4f}m | "
                      f"Vel={vel_mag:.3f}m/s | "
                      f"T={thrust:.1f}N")
                
                last_print_time = current_time
            
            # 控制频率
            time.sleep(config.MPC.dt)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        viewer.close()
    
    # 7. 结果分析
    print("\n" + "="*60)
    print("Results Analysis")
    print("="*60)
    
    # 最终状态
    final_state = sim.state_traj[-1, :]
    final_pos = final_state[0:3]
    final_vel = final_state[9:12]
    final_joints = final_state[7:9]
    
    # 计算指标
    pos_error = np.linalg.norm(final_pos - target_pos)
    vel_magnitude = np.linalg.norm(final_vel)
    joint_error = np.linalg.norm(final_joints - target_joints)
    
    print(f"\nFinal State:")
    print(f"  Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
    print(f"  Position error: {pos_error:.4f} m")
    print(f"  Velocity magnitude: {vel_magnitude:.4f} m/s")
    print(f"  Joint angles: [{np.rad2deg(final_joints[0]):.1f}°, {np.rad2deg(final_joints[1]):.1f}°]")
    print(f"  Joint error: {np.rad2deg(joint_error):.2f}°")
    
    # 计算统计
    if len(errors_history) > 100:
        steady_state_errors = errors_history[-100:]  # 最后2秒
        avg_error = np.mean(steady_state_errors)
        max_error = np.max(steady_state_errors)
        std_error = np.std(steady_state_errors)
        
        print(f"\nSteady-State Performance (last 2s):")
        print(f"  Average error: {avg_error:.4f} m")
        print(f"  Maximum error: {max_error:.4f} m")
        print(f"  Error std dev: {std_error:.4f} m")
    
    # 判断成功
    if pos_error < 0.05 and vel_magnitude < 0.05:
        print("\n✓✓✓ EXCELLENT! Achieved stable hovering!")
    elif pos_error < 0.1 and vel_magnitude < 0.1:
        print("\n✓✓ GOOD! Nearly stable hovering")
    elif pos_error < 0.2:
        print("\n✓ OK - Some drift but controlled")
    else:
        print("\n⚠ WARNING - Significant drift detected")
        print("  Consider:")
        print("  - Increasing position weight (w_pos)")
        print("  - Reducing noise standard deviation")
        print("  - Increasing horizon length")
        print("  - Adding integral control")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        
        time_vec = config.MPC.dt * np.arange(sim.state_traj.shape[0])
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # 位置
        axes[0, 0].plot(time_vec, sim.state_traj[:, 0:3])
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 0].axhline(y=0, color='g', linestyle='--', alpha=0.3)
        axes[0, 0].axhline(y=1.5, color='b', linestyle='--', alpha=0.3)
        axes[0, 0].set_ylabel('Position [m]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].legend(['x', 'y', 'z'])
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Position Tracking')
        
        # 位置误差
        errors = np.array([np.linalg.norm(sim.state_traj[i, 0:3] - target_pos) 
                          for i in range(len(sim.state_traj))])
        axes[0, 1].plot(time_vec, errors)
        axes[0, 1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1m threshold')
        axes[0, 1].set_ylabel('Position Error [m]')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Position Error')
        
        # 速度
        axes[1, 0].plot(time_vec, sim.state_traj[:, 9:12])
        axes[1, 0].set_ylabel('Velocity [m/s]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].legend(['vx', 'vy', 'vz'])
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Velocity')
        
        # 姿态（欧拉角）
        euler_angles = []
        for i in range(len(sim.state_traj)):
            quat = sim.state_traj[i, 3:7]
            # 简单的欧拉角转换
            roll = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 
                              1 - 2*(quat[1]**2 + quat[2]**2))
            pitch = np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1))
            yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                             1 - 2*(quat[2]**2 + quat[3]**2))
            euler_angles.append([roll, pitch, yaw])
        euler_angles = np.array(euler_angles)
        
        axes[1, 1].plot(time_vec, np.rad2deg(euler_angles))
        axes[1, 1].set_ylabel('Euler Angles [deg]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].legend(['Roll', 'Pitch', 'Yaw'])
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Attitude')
        
        # 控制输入
        if len(sim.input_traj) > 0:
            time_vec_ctrl = time_vec[:-1]
            
            # 推力
            axes[2, 0].plot(time_vec_ctrl, sim.input_traj[:, 0])
            axes[2, 0].axhline(y=TOTAL_MASS*GRAVITY, color='r', linestyle='--', label='Hover thrust')
            axes[2, 0].set_ylabel('Thrust [N]')
            axes[2, 0].set_xlabel('Time [s]')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            axes[2, 0].set_title('Thrust Control')
            
            # 扭矩
            axes[2, 1].plot(time_vec_ctrl, sim.input_traj[:, 1:4])
            axes[2, 1].set_ylabel('Torque [Nm]')
            axes[2, 1].set_xlabel('Time [s]')
            axes[2, 1].legend(['τx', 'τy', 'τz'])
            axes[2, 1].grid(True)
            axes[2, 1].set_title('Torque Control')
        
        plt.suptitle('MPPI Control Performance Analysis')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    return sim

if __name__ == "__main__":
    sim = run_optimized_controller()