#!/usr/bin/env python3
"""
渐进式无人机-机械臂MPPI控制系统
使用progressive动力学模型，从简单到复杂逐步测试
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
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse

# 导入渐进式动力学模型
from drone_arm_dynamics_progressive import (
    drone_arm_dynamics_step1,
    drone_arm_dynamics_step2,
    MASS_TOTAL, GRAVITY
)

# GPU设置
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
jax.config.update("jax_default_matmul_precision", "high")


# ============================================================================
# 渐进式目标函数
# ============================================================================
class DroneArmObjectiveProgressive(BaseObjective):
    """渐进式目标函数 - 根据步骤调整权重"""
    
    def __init__(self, step=1):
        super().__init__()
        
        self.step = step
        
        if step == 1:
            # 步骤1：只关注悬停稳定性
            self.w_pos = 100.0      # 位置权重
            self.w_vel = 10.0      # 速度权重  
            self.w_att = 10.0      # 姿态权重
            self.w_omega = 5.0     # 角速度权重
            self.w_joint = 0.0     # 忽略关节
            self.w_joint_vel = 0.0 # 忽略关节速度
            
        elif step == 2:
            # 步骤2：添加关节控制
            self.w_pos = 50.0
            self.w_vel = 10.0  
            self.w_att = 15.0
            self.w_omega = 5.0
            self.w_joint = 5.0     # 开始控制关节
            self.w_joint_vel = 2.0
            
        else:  # step == 3
            # 步骤3：完整控制
            self.w_pos = 100.0
            self.w_vel = 20.0  
            self.w_att = 25.0
            self.w_omega = 10.0
            self.w_joint = 10.0
            self.w_joint_vel = 3.0
        
        # 控制权重（所有步骤相同）
        self.w_thrust = 0.01    # 推力平滑
        self.w_torque = 0.1     # 扭矩平滑
        self.w_joint_ctrl = 0.01 if step > 1 else 0.0
        
        self.nominal_hover_thrust = MASS_TOTAL * GRAVITY
        
    def running_cost(self, state, inputs, reference):
        """运行成本"""
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
        
        # 1. 位置误差
        pos_error = pos - ref_pos
        cost += self.w_pos * jnp.sum(pos_error**2)
        
        # 2. 速度误差
        cost += self.w_vel * jnp.sum(vel**2)
        
        # 3. 姿态误差
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]
        cost += self.w_att * jnp.sum(att_error**2)
        
        # 4. 角速度误差
        cost += self.w_omega * jnp.sum(omega**2)
        
        # 5. 关节误差（步骤2和3）
        if self.step > 1:
            joint_error = q_joints - ref_joints
            cost += self.w_joint * jnp.sum(joint_error**2)
            cost += self.w_joint_vel * jnp.sum(dq_joints**2)
        
        # 6. 控制代价
        thrust_deviation = (inputs[0] - self.nominal_hover_thrust) / self.nominal_hover_thrust
        cost += self.w_thrust * thrust_deviation**2
        cost += self.w_torque * jnp.sum(inputs[1:4]**2)
        
        if self.step > 1:
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
        
        # 8. 位置边界惩罚
        boundary_limit = 3.0
        for i in range(3):
            cost += jnp.where(
                jnp.abs(pos[i] - ref_pos[i]) > boundary_limit,
                1000.0 * (jnp.abs(pos[i] - ref_pos[i]) - boundary_limit)**2,
                0.0
            )
        
        # 防止NaN
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        cost = jnp.where(jnp.isinf(cost), 1e6, cost)
        
        return cost
    
    def final_cost(self, state, reference):
        """终端成本"""
        pos = state[0:3]
        quat = state[3:7]
        q_joints = state[7:9]
        vel = state[9:12]
        omega = state[12:15]
        
        ref_pos = reference[0:3] if reference.shape[0] >= 3 else jnp.array([0, 0, 1.5])
        ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
        ref_joints = reference[7:9] if reference.shape[0] >= 9 else jnp.zeros(2)
        
        # 归一化四元数
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        
        pos_error = pos - ref_pos
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]
        joint_error = q_joints - ref_joints if self.step > 1 else jnp.zeros(2)
        
        cost = (
            50.0 * jnp.sum(pos_error**2) +
            20.0 * jnp.sum(att_error**2) +
            10.0 * jnp.sum(joint_error**2) +
            15.0 * jnp.sum(vel**2) +
            5.0 * jnp.sum(omega**2)
        )
        
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        
        return cost


# ============================================================================
# 主函数
# ============================================================================
def run_drone_arm_mppi_with_mujoco(dynamics_step=1):
    """
    运行带MuJoCo可视化的无人机-机械臂MPPI控制
    
    Args:
        dynamics_step: 动力学复杂度级别 (1=简单, 2=中等, 3=完整)
    """
    
    print("\n" + "="*60)
    print(f"Progressive Drone-Arm System - Step {dynamics_step}")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print("="*60 + "\n")
    
    # 1. 配置机器人
    robot_config = settings.RobotConfig()
    robot_config.robot_scene_path = "examples/drone_direct_control.xml"
    robot_config.nq = 9
    robot_config.nv = 8
    robot_config.nu = 6
    
    # 控制限制
    robot_config.input_min = jnp.array([0., -0.5, -0.5, -0.5, -0.5, -0.5])
    robot_config.input_max = jnp.array([20., 0.5, 0.5, 0.5, 0.5, 0.5])
    
    # 初始状态
    robot_config.q_init = jnp.array([
        0., 0., 1.5,      # 位置
        1., 0., 0., 0.,   # 四元数
        0., 0.            # 关节
    ], dtype=jnp.float32)
    
    # 2. 创建配置
    config = settings.Config(robot_config)
    
    # 关闭SBMPC内置可视化
    config.general.visualize = False
    config.general.integrator_type = "rk4"
    
    # MPPI参数 - 根据步骤调整
    if dynamics_step == 1:
        # 步骤1：优化后的参数
        config.MPC.dt = 0.02
        config.MPC.horizon = 12  # 稍长一点
        config.MPC.num_parallel_computations = 1500  # 增加采样
        config.MPC.lambda_mpc = 0.01  # 极低的温度
        
        # 进一步减小噪声
        config.MPC.std_dev_mppi = jnp.array([
            0.2,                    # 推力噪声
            0.01, 0.01, 0.005,     # 扭矩噪声（更小）
            0.0, 0.0               # 关节锁定
        ])
        
        # 微调初始推力
        hover_thrust = MASS_TOTAL * GRAVITY
        config.MPC.initial_guess = jnp.array([
            hover_thrust,  # 不要1.05倍，直接用悬停值
            0., 0., 0., 0., 0.
        ])
        
    elif dynamics_step == 2:
        # 步骤2：中等参数
        config.MPC.dt = 0.02
        config.MPC.horizon = 20
        config.MPC.num_parallel_computations = 800
        config.MPC.lambda_mpc = 8.0
        
        config.MPC.std_dev_mppi = jnp.array([
            1.2,                    # 推力噪声
            0.08, 0.08, 0.03,      # 扭矩噪声
            0.05, 0.05             # 关节噪声
        ])
        
    else:  # dynamics_step == 3
        # 步骤3：完整参数
        config.MPC.dt = 0.02
        config.MPC.horizon = 25
        config.MPC.num_parallel_computations = 1000
        config.MPC.lambda_mpc = 5.0
        
        config.MPC.std_dev_mppi = jnp.array([
            1.5,                    # 推力噪声
            0.1, 0.1, 0.05,        # 扭矩噪声
            0.1, 0.1               # 关节噪声
        ])
    
    # 初始猜测
    hover_thrust = MASS_TOTAL * GRAVITY
    config.MPC.initial_guess = jnp.array([
        hover_thrust, 0., 0., 0., 0., 0.
    ])
    
    config.MPC.smoothing = None
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = False
    
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_iterations = 500  # 10秒
    
    # 3. 定义任务
    print(f"Step {dynamics_step} Configuration:")
    if dynamics_step == 1:
        print("  - Simplified dynamics (no coupling, locked joints)")
        print("  - Task: Stable hovering at [0, 0, 1.5]")
        target_joints = jnp.array([0.0, 0.0])
    elif dynamics_step == 2:
        print("  - Medium complexity (gyro effects, simple joints)")
        print("  - Task: Hovering with small joint movement")
        target_joints = jnp.array([0.1, -0.1])
    else:
        print("  - Full dynamics (all coupling effects)")
        print("  - Task: Hovering with joint tracking")
        target_joints = jnp.array([0.2, -0.2])
    
    print(f"  - Horizon: {config.MPC.horizon}")
    print(f"  - Samples: {config.MPC.num_parallel_computations}")
    print(f"  - Lambda: {config.MPC.lambda_mpc}")
    print(f"  - dt: {config.MPC.dt}")
    
    # 目标状态
    target_pos = jnp.array([0.0, 0.0, 1.5])
    target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    
    # 创建参考状态
    ref_state = jnp.zeros(17)
    ref_state = ref_state.at[0:3].set(target_pos)
    ref_state = ref_state.at[3:7].set(target_quat)
    ref_state = ref_state.at[7:9].set(target_joints)
    
    reference = jnp.tile(ref_state, (config.MPC.horizon + 1, 1))
    
    # 4. 创建目标函数
    objective = DroneArmObjectiveProgressive(step=dynamics_step)
    
    # 5. 选择动力学
    if dynamics_step == 1:
        print("\nUsing dynamics step 1: Simplified")
        dynamics_fn = drone_arm_dynamics_step1
    elif dynamics_step == 2:
        print("\nUsing dynamics step 2: Medium complexity")
        dynamics_fn = drone_arm_dynamics_step2
    else:
        # 如果你有step3，可以导入并使用
        print("\nUsing dynamics step 2: Medium complexity (step 3 not yet implemented)")
        dynamics_fn = drone_arm_dynamics_step2
    
    # 6. 构建仿真
    print("\nBuilding simulation...")
    sim = build_all(
        config,
        objective,
        reference,
        custom_dynamics_fn=dynamics_fn,
        obstacles=False
    )
    
    # 7. 创建MuJoCo可视化
    print("Setting up MuJoCo visualization...")
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
    
    # 8. 运行仿真循环
    try:
        last_print_time = 0
        errors_history = []
        thrust_history = []
        
        for i in range(config.sim_iterations):
            # SBMPC步进
            sim.step()
            
            # 获取当前状态
            current_state = sim.state_traj[i+1, :]
            
            # 状态有效性检查
            if jnp.any(jnp.isnan(current_state)):
                print(f"\n⚠️ WARNING: NaN detected at step {i}!")
                print(f"Last valid state: {sim.state_traj[i, :]}")
                break
            
            if jnp.any(jnp.abs(current_state[0:3]) > 10):
                print(f"\n⚠️ WARNING: Position out of bounds at step {i}!")
                print(f"Position: {current_state[0:3]}")
                break
            
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
            
            # 记录误差和推力
            pos = current_state[0:3]
            pos_error = np.linalg.norm(pos - target_pos)
            errors_history.append(pos_error)
            
            if i < len(sim.input_traj):
                thrust = sim.input_traj[i, 0]
                thrust_history.append(thrust)
            
            # 打印状态（每0.5秒）
            current_time = i * config.MPC.dt
            if current_time - last_print_time >= 0.5:
                vel_mag = np.linalg.norm(current_state[9:12])
                omega_mag = np.linalg.norm(current_state[12:15])
                joints = current_state[7:9]
                
                print(f"t={current_time:5.1f}s | "
                      f"Pos=[{pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}] | "
                      f"Err={pos_error:.4f}m | "
                      f"Vel={vel_mag:.3f}m/s | "
                      f"Omega={omega_mag:.3f}rad/s | "
                      f"J=[{np.rad2deg(joints[0]):5.1f}°,{np.rad2deg(joints[1]):5.1f}°] | "
                      f"T={thrust:.1f}N")
                
                # 检查推力是否合理
                if abs(thrust - hover_thrust) > 5.0:
                    print(f"  ⚠️ Thrust deviation: {thrust - hover_thrust:.2f}N from hover")
                
                last_print_time = current_time
            
            # 控制频率
            time.sleep(config.MPC.dt)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        viewer.close()
    
    # 9. 结果分析
    print("\n" + "="*60)
    print(f"Results Analysis - Step {dynamics_step}")
    print("="*60)
    
    # 最终状态
    final_state = sim.state_traj[-1, :]
    final_pos = final_state[0:3]
    final_vel = final_state[9:12]
    final_omega = final_state[12:15]
    final_joints = final_state[7:9]
    
    # 计算指标
    pos_error = np.linalg.norm(final_pos - target_pos)
    vel_magnitude = np.linalg.norm(final_vel)
    omega_magnitude = np.linalg.norm(final_omega)
    joint_error = np.linalg.norm(final_joints - target_joints)
    
    print(f"\nFinal State:")
    print(f"  Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
    print(f"  Position error: {pos_error:.4f} m")
    print(f"  Velocity magnitude: {vel_magnitude:.4f} m/s")
    print(f"  Angular velocity: {omega_magnitude:.4f} rad/s")
    
    if dynamics_step > 1:
        print(f"  Joint angles: [{np.rad2deg(final_joints[0]):.1f}°, {np.rad2deg(final_joints[1]):.1f}°]")
        print(f"  Target joints: [{np.rad2deg(target_joints[0]):.1f}°, {np.rad2deg(target_joints[1]):.1f}°]")
        print(f"  Joint error: {np.rad2deg(joint_error):.2f}°")
    
    # 稳态性能
    if len(errors_history) > 100:
        steady_state_errors = errors_history[-100:]
        avg_error = np.mean(steady_state_errors)
        max_error = np.max(steady_state_errors)
        std_error = np.std(steady_state_errors)
        
        print(f"\nSteady-State Performance (last 2s):")
        print(f"  Average error: {avg_error:.4f} m")
        print(f"  Maximum error: {max_error:.4f} m")
        print(f"  Error std dev: {std_error:.4f} m")
        
        # 推力分析
        if len(thrust_history) > 100:
            steady_thrust = thrust_history[-100:]
            avg_thrust = np.mean(steady_thrust)
            thrust_variation = np.std(steady_thrust)
            print(f"\nThrust Analysis:")
            print(f"  Average thrust: {avg_thrust:.2f} N")
            print(f"  Expected hover: {hover_thrust:.2f} N")
            print(f"  Thrust variation: {thrust_variation:.3f} N")
    
    # 性能评估
    print("\n" + "-"*60)
    if dynamics_step == 1:
        if pos_error < 0.05 and vel_magnitude < 0.05:
            print("✓✓✓ EXCELLENT! Step 1 passed - stable hovering achieved!")
            print("→ Ready to proceed to Step 2")
        elif pos_error < 0.1:
            print("✓✓ GOOD! Nearly stable, minor tuning needed")
        elif pos_error < 0.2:
            print("✓ OK - Some drift but controlled")
        else:
            print("✗ FAILED - Need to debug before proceeding")
            print("\nTroubleshooting:")
            print("  1. Check if average thrust is close to hover thrust")
            print("  2. Try reducing lambda_mpc further (e.g., 5.0)")
            print("  3. Increase samples to 1000")
            print("  4. Check for numerical issues in dynamics")
    
    elif dynamics_step == 2:
        if pos_error < 0.05 and joint_error < 0.1:
            print("✓✓✓ EXCELLENT! Step 2 passed - joint control working!")
            print("→ Ready for Step 3 (full dynamics)")
        elif pos_error < 0.1:
            print("✓✓ GOOD! Position stable, joints need tuning")
        else:
            print("✓ Partial success - continue tuning")
    
    # 绘图分析
    try:
        import matplotlib.pyplot as plt
        
        time_vec = config.MPC.dt * np.arange(len(errors_history))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 位置误差
        axes[0, 0].plot(time_vec, errors_history)
        axes[0, 0].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Target: 5cm')
        axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Limit: 10cm')
        axes[0, 0].set_ylabel('Position Error [m]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Position Error Over Time')
        
        # 推力历史
        if len(thrust_history) > 0:
            time_vec_thrust = config.MPC.dt * np.arange(len(thrust_history))
            axes[0, 1].plot(time_vec_thrust, thrust_history)
            axes[0, 1].axhline(y=hover_thrust, color='r', linestyle='--', label=f'Hover: {hover_thrust:.1f}N')
            axes[0, 1].set_ylabel('Thrust [N]')
            axes[0, 1].set_xlabel('Time [s]')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            axes[0, 1].set_title('Thrust Command')
        
        # 3D轨迹
        axes[1, 0].remove()
        ax3d = fig.add_subplot(223, projection='3d')
        positions = sim.state_traj[:len(errors_history), 0:3]
        ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax3d.scatter([0], [0], [1.5], c='r', s=100, marker='*', label='Target')
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('Z [m]')
        ax3d.legend()
        ax3d.set_title('3D Trajectory')
        
        # 速度
        velocities = sim.state_traj[:len(errors_history), 9:12]
        vel_mags = np.linalg.norm(velocities, axis=1)
        axes[1, 1].plot(time_vec, vel_mags)
        axes[1, 1].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Target: 5cm/s')
        axes[1, 1].set_ylabel('Velocity Magnitude [m/s]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Velocity Magnitude')
        
        plt.suptitle(f'Drone-Arm System Performance - Step {dynamics_step}')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    return sim


if __name__ == "__main__":
    # 选择要测试的步骤
    # 1 = 简单动力学（推荐从这里开始）
    # 2 = 中等复杂度
    # 3 = 完整动力学
    
    DYNAMICS_STEP = 1  # 从步骤1开始
    
    print("\n" + "="*60)
    print("PROGRESSIVE TESTING FRAMEWORK")
    print("="*60)
    print(f"Starting with Step {DYNAMICS_STEP}")
    print("Recommendation: Start with Step 1, verify stability,")
    print("then progress to Step 2 and 3")
    print("="*60)
    
    sim = run_drone_arm_mppi_with_mujoco(dynamics_step=DYNAMICS_STEP)