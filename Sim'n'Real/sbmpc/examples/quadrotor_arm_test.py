#!/usr/bin/env python3
"""
无人机-机械臂系统渐进式测试框架
支持三个步骤和多种任务的测试
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import time
from enum import Enum
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_all
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse

# 导入稳定的动力学模型
from drone_arm_dynamics_stable import (
    dynamics_step1,
    dynamics_step2,
    dynamics_step3_stable,
    MASS_TOTAL, GRAVITY,
    compute_com_offset,
    compute_end_effector_position
)

# 导入任务配置
from task_configs import TaskConfig

# # GPU设置
# os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
# jax.config.update("jax_default_matmul_precision", "high")


# ============================================================================
# 任务类型定义
# ============================================================================
class TaskType(Enum):
    HOVER = "hover"
    REACH_POINT = "reach_point"
    ARM_CONTROL = "arm_control"
    TRAJECTORY = "trajectory"
    END_EFFECTOR_TRAJECTORY = "end_effector_trajectory"

# ============================================================================
# 设备选择和配置
# ============================================================================
def setup_compute_device():
    """
    自动检测并配置计算设备（GPU或CPU）
    
    Returns:
        str: 使用的设备类型 ('gpu' 或 'cpu')
    """
    device_type = 'cpu'
    
    try:
        # 检查可用的设备
        devices = jax.devices()
        
        # 查找GPU设备
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if gpu_devices:
            # GPU可用
            print("✓ GPU detected and available")
            print(f"  Device: {gpu_devices[0]}")
            
            # 设置GPU优化选项
            os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
            
            # 确保使用第一个GPU
            jax.config.update('jax_default_device', gpu_devices[0])
            device_type = 'gpu'
            
            # 测试GPU是否真的可用
            try:
                test_array = jnp.ones((100, 100))
                result = jnp.dot(test_array, test_array)
                result.block_until_ready()
                print("  GPU test successful")
            except Exception as e:
                print(f"  ⚠️ GPU test failed: {e}")
                print("  Falling back to CPU")
                device_type = 'cpu'
        else:
            print("ℹ️ No GPU detected, using CPU")
            device_type = 'cpu'
            
    except Exception as e:
        print(f"⚠️ Error detecting devices: {e}")
        print("  Falling back to CPU")
        device_type = 'cpu'
    
    # CPU配置
    if device_type == 'cpu':
        # CPU优化设置
        cpu_devices = [d for d in jax.devices() if d.platform == 'cpu']
        if cpu_devices:
            jax.config.update('jax_default_device', cpu_devices[0])
            print(f"  Using CPU: {cpu_devices[0]}")
        
        # CPU并行设置（根据核心数调整）
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        print(f"  Available CPU cores: {num_cores}")
        
        # 设置CPU线程数（可以根据需要调整）
        os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count={min(num_cores, 8)}'
    
    # 通用优化设置
    jax.config.update("jax_default_matmul_precision", "high")
    
    # 打印最终配置
    print(f"\n{'='*50}")
    print(f"Compute Configuration:")
    print(f"  Device Type: {device_type.upper()}")
    print(f"  JAX Version: {jax.__version__}")
    print(f"  Default Backend: {jax.default_backend()}")
    print(f"  Active Devices: {jax.devices()}")
    print(f"{'='*50}\n")
    
    return device_type

DEVICE_TYPE = setup_compute_device()

# ============================================================================
# 自适应目标函数
# ============================================================================
class AdaptiveObjective(BaseObjective):
    """纯MPPI优化版本：不依赖反馈增益的稳定控制"""
    
    def __init__(self, task_type: TaskType, step: int = 1):
        super().__init__()
        self.task_type = task_type
        self.step = step
        self._setup_weights()
        self.nominal_hover_thrust = MASS_TOTAL * GRAVITY
        
    def _setup_weights(self):
        """根据任务设置权重（纯MPPI优化）"""
        if self.task_type == TaskType.HOVER:
            # 悬停任务：平衡且有效的权重
            self.w_pos = 100.0      # 位置权重
            self.w_vel = 30.0       # 速度权重
            self.w_att = 50.0       # 姿态权重
            self.w_omega = 20.0     # 角速度权重
            self.w_joint = 0.0 if self.step == 1 else 5.0
            self.w_joint_vel = 0.0 if self.step == 1 else 2.0
            # 特殊项权重
            self.w_pos_integral = 50.0  # 位置积分效应
            self.w_vel_ref = 40.0       # 速度参考跟踪
            self.w_prediction = 30.0    # 预测项
            
        elif self.task_type == TaskType.REACH_POINT:
            self.w_pos = 80.0
            self.w_vel = 25.0
            self.w_att = 40.0
            self.w_omega = 15.0
            self.w_joint = 0.0
            self.w_joint_vel = 0.0
            self.w_pos_integral = 30.0
            self.w_vel_ref = 30.0
            self.w_prediction = 20.0
            
        elif self.task_type == TaskType.ARM_CONTROL:
            self.w_pos = 100.0
            self.w_vel = 35.0
            self.w_att = 50.0
            self.w_omega = 20.0
            self.w_joint = 40.0
            self.w_joint_vel = 15.0
            self.w_pos_integral = 50.0
            self.w_vel_ref = 40.0
            self.w_prediction = 30.0

        elif self.task_type == TaskType.END_EFFECTOR_TRAJECTORY:
            # 末端执行器轨迹跟踪：借鉴ARM_CONTROL的成功数值
            self.w_pos = 120.0      # 提高末端位置权重（主要目标）
            self.w_vel = 40.0       # 借鉴ARM_CONTROL
            self.w_att = 50.0       # 借鉴ARM_CONTROL
            self.w_omega = 20.0     # 借鉴ARM_CONTROL
            self.w_joint = 15.0     # 关节权重降低（因为是间接控制）
            self.w_joint_vel = 15.0 # 借鉴ARM_CONTROL
            self.w_pos_integral = 60.0  # 提高积分效应
            self.w_vel_ref = 45.0       # 提高速度参考跟踪
            self.w_prediction = 35.0    # 借鉴ARM_CONTROL
            
        else:  # TRAJECTORY
            self.w_pos = 80.0
            self.w_vel = 30.0
            self.w_att = 40.0
            self.w_omega = 15.0
            self.w_joint = 5.0
            self.w_joint_vel = 2.0
            self.w_pos_integral = 30.0
            self.w_vel_ref = 35.0
            self.w_prediction = 25.0
            
        
        # 控制权重
        self.w_thrust = 0.001
        self.w_torque = 0.01
        self.w_joint_ctrl = 0.001 if self.step > 1 else 0.0
        
    def running_cost(self, state, inputs, reference):
        """改进的运行成本函数"""
        # 解析状态
        pos = state[0:3]
        quat = state[3:7]
        q_joints = state[7:9]
        vel = state[9:12]
        omega = state[12:15]
        dq_joints = state[15:17]
        
        # 归一化四元数
        quat_norm = jnp.linalg.norm(quat) + 1e-10
        quat_normalized = quat / quat_norm
        
        cost = 0.0
        
        # 初始化变量（避免作用域问题）
        ideal_base_pos = pos  # 默认值
        ideal_joints = q_joints  # 默认值
        
        # 根据任务类型选择目标
        if self.task_type == TaskType.END_EFFECTOR_TRAJECTORY:
            # === 纯MPPI：只关注最终目标 ===
            ref_ee_pos = reference[0:3]
            
            # 计算实际末端位置
            ee_pos = compute_end_effector_position(pos, quat_normalized, 
                                                q_joints[0], q_joints[1])
            ee_error = ee_pos - ref_ee_pos
            ee_error_norm = jnp.linalg.norm(ee_error)
            
            # ===== 简单直接的成本 =====
            
            # 1. 末端误差（主要目标）
            ee_weight = jnp.where(
                ee_error_norm < 0.1,
                300.0,  # 接近时增加权重
                150.0   # 正常权重
            )
            cost += ee_weight * jnp.sum(ee_error**2)
            
            # 2. 速度惩罚（避免过快移动）
            cost += 20.0 * jnp.sum(vel**2)
            
            # 3. 角速度惩罚
            cost += 10.0 * jnp.sum(omega**2)
            
            # 4. 关节速度惩罚
            cost += 5.0 * jnp.sum(dq_joints**2)
            
            # 5. 能量消耗
            thrust_diff = inputs[0] - MASS_TOTAL * GRAVITY
            cost += 0.01 * thrust_diff**2
            cost += 0.01 * jnp.sum(inputs[1:4]**2)  # 扭矩
            cost += 0.001 * jnp.sum(inputs[4:6]**2)  # 关节控制
            
            # 6. 稳态误差惩罚（鼓励收敛）
            static_penalty = ee_error_norm * jnp.exp(-10.0 * jnp.linalg.norm(vel))
            cost += 100.0 * static_penalty**2
            
            # 就这样！让MPPI自己找出如何移动基座和关节
            
        else:
            # === 原有模式的位置控制 ===
            ref_pos = reference[0:3] if reference.shape[0] >= 3 else jnp.array([0, 0, 1.5])
            ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
            ref_joints = reference[7:9] if reference.shape[0] >= 9 else jnp.zeros(2)
            
            # 基础位置误差
            pos_error = pos - ref_pos
            pos_error_norm = jnp.linalg.norm(pos_error)
            
            cost += self.w_pos * jnp.where(
                pos_error_norm < 0.1,
                100.0 * jnp.sum(pos_error**2),
                jnp.sum(pos_error**2)
            )
            
            # 速度误差
            k_p = 3.0
            desired_vel = -k_p * pos_error
            desired_vel = jnp.clip(desired_vel, -0.5, 0.5)
            vel_error = vel - desired_vel
            cost += self.w_vel * jnp.sum(vel_error**2)
            
            # 积分效应
            static_error = pos_error_norm * jnp.exp(-5.0 * jnp.linalg.norm(vel))
            cost += self.w_pos_integral * static_error**2
            
            # 预测控制
            dt_pred = 0.2
            future_pos = pos + vel * dt_pred
            future_error = future_pos - ref_pos
            cost += self.w_prediction * jnp.sum(future_error**2)

        # === 公共部分：姿态、角速度、控制等 ===
        
        # 姿态控制
        ref_quat = reference[3:7] if reference.shape[0] >= 7 else jnp.array([1, 0, 0, 0])
        quat_error = quat_product(quat_inverse(ref_quat), quat_normalized)
        att_error = quat_error[1:4]
        att_error_norm = jnp.linalg.norm(att_error)
        
        cost += self.w_att * jnp.where(
            att_error_norm < 0.05,
            100.0 * jnp.sum(att_error**2),
            jnp.sum(att_error**2)
        )
        
        # 角速度
        desired_omega = -5.0 * att_error
        omega_error = omega - desired_omega
        cost += self.w_omega * jnp.sum(omega_error**2)
        
        # 推力控制
        # 使用规划的基座高度作为目标
        z_target = ideal_base_pos[2]
        z_error = pos[2] - z_target
        z_vel = vel[2]
        
        k_p_thrust = 2.5
        k_d_thrust = 1.2
        thrust_adjustment = -k_p_thrust * z_error - k_d_thrust * z_vel
        
        # 考虑重心偏移（step2及以上）
        if self.step >= 2:
            com_offset = compute_com_offset(q_joints[0], q_joints[1])
            gravity_compensation = MASS_TOTAL * GRAVITY * (1.0 + 0.1 * com_offset[2])
        else:
            gravity_compensation = self.nominal_hover_thrust
            
        expected_thrust = gravity_compensation + MASS_TOTAL * thrust_adjustment
        
        # 动态调整推力限制
        if self.task_type == TaskType.END_EFFECTOR_TRAJECTORY:
            # 根据目标高度调整推力范围
            thrust_min = 8.0
            thrust_max = jnp.minimum(18.0, 14.0 + (z_target - 1.5) * 2.0)
            expected_thrust = jnp.clip(expected_thrust, thrust_min, thrust_max)
        else:
            expected_thrust = jnp.clip(expected_thrust, 9.0, 15.0)
        
        thrust_error = inputs[0] - expected_thrust
        cost += self.w_thrust * thrust_error**2
        
        # 扭矩成本
        cost += self.w_torque * jnp.sum(inputs[1:4]**2)
        
        # 关节控制
        if self.task_type == TaskType.ARM_CONTROL and self.step > 1:
            ref_joints = reference[7:9] if reference.shape[0] >= 9 else jnp.zeros(2)
            joint_pos_error = q_joints - ref_joints
            cost += self.w_joint * jnp.sum(joint_pos_error**2)
            
            desired_joint_vel = -2.0 * joint_pos_error
            joint_vel_error = dq_joints - desired_joint_vel
            cost += self.w_joint_vel * jnp.sum(joint_vel_error**2)
        elif self.task_type == TaskType.END_EFFECTOR_TRAJECTORY:
            # 惩罚过快的关节运动
            cost += self.w_joint_vel * jnp.sum(dq_joints**2)
        
        if inputs.shape[0] > 4:
            cost += self.w_joint_ctrl * jnp.sum(inputs[4:6]**2)
        
        # 任务特定约束
        if self.task_type == TaskType.HOVER:
            xy_vel = vel[0:2]
            cost += 50.0 * jnp.sum(xy_vel**2)
            
            max_tilt = 0.1
            tilt_magnitude = jnp.sqrt(att_error[0]**2 + att_error[1]**2)
            cost += jnp.where(
                tilt_magnitude > max_tilt,
                10000.0 * (tilt_magnitude - max_tilt)**2,
                0.0
            )
        
        # 稳定性约束
        vel_mag = jnp.linalg.norm(vel)
        cost += jnp.where(
            vel_mag > 1.0,
            1000.0 * (vel_mag - 1.0)**2,
            0.0
        )
        
        # 位置边界
        pos_limit = 50.0 if self.task_type == TaskType.END_EFFECTOR_TRAJECTORY else 2.0
        for i in range(3):
            cost += jnp.where(
                jnp.abs(pos[i]) > pos_limit,
                5000.0 * (jnp.abs(pos[i]) - pos_limit)**2,
                0.0
            )
        
        # 关节限制
        joint_limit = 1.4  # 比物理极限1.6小
        cost += jnp.where(
            jnp.abs(q_joints[0]) > joint_limit,
            1000.0 * (jnp.abs(q_joints[0]) - joint_limit)**2,
            0.0
        )
        cost += jnp.where(
            jnp.abs(q_joints[1]) > joint_limit,
            1000.0 * (jnp.abs(q_joints[1]) - joint_limit)**2,
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
        vel = state[9:12]
        
        if self.task_type == TaskType.END_EFFECTOR_TRAJECTORY:
            # 末端执行器模式
            quat = state[3:7]
            q1, q2 = state[7], state[8]
            quat_norm = jnp.linalg.norm(quat) + 1e-10
            quat_normalized = quat / quat_norm
            
            ee_pos = compute_end_effector_position(pos, quat_normalized, q1, q2)
            ref_ee_pos = reference[0:3]
            ee_error = ee_pos - ref_ee_pos
            
            cost = 300.0 * jnp.sum(ee_error**2) + 50.0 * jnp.sum(vel**2)
        else:
            # 原有模式
            ref_pos = reference[0:3] if reference.shape[0] >= 3 else jnp.array([0, 0, 1.5])
            pos_error = pos - ref_pos
            
            if self.task_type == TaskType.HOVER:
                cost = 200.0 * jnp.sum(pos_error**2) + 100.0 * jnp.sum(vel**2)
            else:
                cost = 150.0 * jnp.sum(pos_error**2) + 50.0 * jnp.sum(vel**2)
        
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        return cost


# ============================================================================
# 测试场景生成器
# ============================================================================
class TestScenario:
    """生成不同的测试场景"""
    
    @staticmethod
    def hover_test(height=1.5, duration=10.0):
        """悬停测试"""
        return {
            'name': 'Hover Test',
            'task_type': TaskType.HOVER,
            'target_pos': jnp.array([0.0, 0.0, height]),
            'target_quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
            'target_joints': jnp.array([0.0, 0.0]),
            'duration': duration
        }
    
    @staticmethod
    def reach_point_test(target_pos, duration=10.0):
        """到达目标点测试"""
        return {
            'name': 'Reach Point Test',
            'task_type': TaskType.REACH_POINT,
            'target_pos': jnp.array(target_pos),
            'target_quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
            'target_joints': jnp.array([0.0, 0.0]),
            'duration': duration
        }
    
    @staticmethod
    def arm_control_test(hover_pos, joint_targets, duration=15.0):
        """机械臂控制测试（悬停时）"""
        return {
            'name': 'Arm Control Test',
            'task_type': TaskType.ARM_CONTROL,
            'target_pos': jnp.array(hover_pos),
            'target_quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
            'target_joints': jnp.array(joint_targets),
            'duration': duration
        }
    
    @staticmethod
    def trajectory_test(waypoints, duration=20.0):
        """轨迹跟踪测试"""
        return {
            'name': 'Trajectory Test',
            'task_type': TaskType.TRAJECTORY,
            'waypoints': waypoints,
            'duration': duration
        }
    
    @staticmethod
    def end_effector_trajectory_test(ee_target, duration=20.0):
        """末端执行器轨迹跟踪测试 - 简化为单目标
        
        Args:
            ee_target: 单个3D目标点 [x, y, z]
            duration: 测试持续时间
        """
        # 确保ee_target是一个3D坐标
        target_point = jnp.array(ee_target) if not isinstance(ee_target, jnp.ndarray) else ee_target
        
        return {
            'name': 'End-Effector Trajectory Test',
            'task_type': TaskType.END_EFFECTOR_TRAJECTORY,
            'ee_waypoints': [target_point],  # 包装为列表，但保持完整的3D坐标
            'target_pos': jnp.array([0.0, 0.0, 1.5]),
            'target_quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
            'target_joints': jnp.array([0.0, 0.0]),
            'duration': duration
        }


# ============================================================================
# 主测试函数
# ============================================================================
def run_test(scenario: Dict, dynamics_step: int = 1, visualize: bool = True):
    """
    运行单个测试场景
    
    Args:
        scenario: 测试场景配置
        dynamics_step: 动力学复杂度 (1, 2, 或 3)
        visualize: 是否显示MuJoCo可视化
    """
    
    print("\n" + "="*70)
    print(f"Test: {scenario['name']}")
    print(f"Dynamics Step: {dynamics_step}")
    print(f"Task Type: {scenario['task_type'].value}")
    print("="*70)
    
    # 1. 配置机器人
    robot_config = settings.RobotConfig()
    robot_config.robot_scene_path = "examples/drone_direct_control.xml"
    robot_config.nq = 9
    robot_config.nv = 8
    robot_config.nu = 6
    
    # 控制限制（通用范围）
    robot_config.input_min = jnp.array([0., -1.0, -1.0, -1.0, -0.5, -0.5])
    robot_config.input_max = jnp.array([20., 1.0, 1.0, 1.0, 0.5, 0.5])
    
    # 初始状态（末端执行器模式使用非零关节角度）
    if scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        
        robot_config.q_init = jnp.array([
            0., 0., 1.5,      # 位置
            1., 0., 0., 0.,   # 四元数
            0.2, -0.2         # 初始关节角度（非零）
        ], dtype=jnp.float32)
        initial_state = robot_config.q_init
        initial_ee = compute_end_effector_position(
            initial_state[0:3],  # base_pos
            initial_state[3:7],  # base_quat
            initial_state[7],    # q1
            initial_state[8]     # q2
        )
        print(f"\nInitial end-effector position: {initial_ee}")
        print(f"Target end-effector position: {scenario['ee_waypoints'][0]}")
        print(f"Initial error: {jnp.linalg.norm(initial_ee - scenario['ee_waypoints'][0]):.4f} m")
    else:
        robot_config.q_init = jnp.array([
            0., 0., 1.5,
            1., 0., 0., 0.,
            0., 0.
        ], dtype=jnp.float32)
    
    # 2. 创建配置
    config = settings.Config(robot_config)
    config.general.visualize = False
    config.general.integrator_type = "rk4"
    
    # 3. 获取任务特定的MPPI参数
    task_config = TaskConfig.get_config_for_task(
        scenario['task_type'].value, dynamics_step
    )

    # 根据设备类型调整参数
    if DEVICE_TYPE == 'cpu':
        # CPU上可能需要减少采样数以保持性能
        print("ℹ️ Adjusting parameters for CPU execution")
        
        # 降低采样数以提高CPU性能
        original_samples = task_config['samples']
        task_config['samples'] = min(original_samples, 3000)  # CPU上限制最大采样数
        
        if original_samples != task_config['samples']:
            print(f"  Reduced samples from {original_samples} to {task_config['samples']}")
        
        # 可选：稍微增加噪声以补偿采样数减少
        # task_config['noise'] = task_config['noise'] * 1.1
    
    config.MPC.dt = task_config['dt']
    config.MPC.horizon = task_config['horizon']
    config.MPC.num_parallel_computations = task_config['samples']
    config.MPC.lambda_mpc = task_config['lambda']
    config.MPC.std_dev_mppi = task_config['noise']
    
    # 初始猜测
    hover_thrust = MASS_TOTAL * GRAVITY
    config.MPC.initial_guess = jnp.array([
        hover_thrust, 0., 0., 0., 0., 0.
    ])
    
    config.MPC.smoothing = None
    config.MPC.num_control_points = config.MPC.horizon
    
    config.MPC.gains = False
    config.MPC.sensitivity = False
    
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_iterations = int(scenario['duration'] / config.MPC.dt)
    
    # 4. 打印配置
    print("\nConfiguration:")
    print(f"  Duration: {scenario['duration']}s")
    print(f"  Iterations: {config.sim_iterations}")
    print(f"  dt: {config.MPC.dt}")
    print(f"  Horizon: {config.MPC.horizon}")
    print(f"  Samples: {config.MPC.num_parallel_computations}")
    print(f"  Lambda: {config.MPC.lambda_mpc}")
    print(f"  Noise std: {task_config['noise']}")
    
    # 5. 创建参考轨迹
    if scenario['task_type'] == TaskType.TRAJECTORY:
        # 轨迹跟踪需要时变参考
        reference = generate_trajectory_reference(
            scenario['waypoints'], 
            config.sim_iterations,
            config.MPC.horizon,
            config.MPC.dt
        )
    elif scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        # 末端执行器轨迹
        reference = generate_end_effector_trajectory_reference(
            scenario['ee_waypoints'],
            config.sim_iterations,
            config.MPC.horizon,
            config.MPC.dt
        )
    else:
        # 固定目标
        ref_state = jnp.zeros(17)
        ref_state = ref_state.at[0:3].set(scenario['target_pos'])
        ref_state = ref_state.at[3:7].set(scenario['target_quat'])
        ref_state = ref_state.at[7:9].set(scenario['target_joints'])
        reference = jnp.tile(ref_state, (config.MPC.horizon + 1, 1))
    
    # 6. 创建目标函数
    objective = AdaptiveObjective(scenario['task_type'], dynamics_step)
    
    # 7. 选择动力学
    if dynamics_step == 1:
        print("\nUsing Step 1: Basic stable dynamics")
        dynamics_fn = dynamics_step1
    elif dynamics_step == 2:
        print("\nUsing Step 2: With COM compensation")
        dynamics_fn = dynamics_step2
    else:
        print("\nUsing Step 3: Full coupled dynamics (stabilized)")
        dynamics_fn = dynamics_step3_stable
    
    # 8. 构建仿真
    print("\nBuilding simulation...")
    sim = build_all(
        config,
        objective,
        reference,
        custom_dynamics_fn=dynamics_fn,
        obstacles=False
    )
    
    # 9. 运行仿真
    if visualize:
        run_with_visualization(sim, config, scenario)
    else:
        run_headless(sim, config, scenario)
    
    # 10. 分析结果
    if scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        results = analyze_end_effector_results(sim, scenario, config)
    else:
        results = analyze_results(sim, scenario, config)
    
    return sim, results

def run_test_with_diagnostics(scenario: Dict, dynamics_step: int = 1, visualize: bool = True):
    """运行测试并提供诊断信息"""
    
    # 运行原始的run_test
    sim, results = run_test(scenario, dynamics_step, visualize)
    
    # 添加诊断
    gains = diagnose_controller(sim, scenario)
    
    # 如果增益为零，提供修复建议
    if gains is not None and np.linalg.norm(gains) < 1e-6:
        print("\n" + "="*70)
        print("SUGGESTED FIXES:")
        print("="*70)
        print("1. Check if sensitivity computation in RolloutGenerator is working")
        print("2. Verify that gains computation in MPPIGain is correct")
        print("3. Try increasing the noise levels to get better gradient estimates")
        print("4. Consider using finite differences for sensitivity if automatic differentiation fails")
    
    return sim, results

def run_with_visualization(sim, config, scenario):
    """带MuJoCo可视化运行"""
    print("\nRunning with MuJoCo visualization...")
    
    mj_model = mujoco.MjModel.from_xml_path("examples/drone_direct_control.xml")
    mj_data = mujoco.MjData(mj_model)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    # 设置相机
    viewer.cam.distance = 5.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 45

    # 如果是末端执行器任务，更新目标球位置
    if scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        # 获取目标末端执行器位置
        target_ee_pos = scenario['ee_waypoints'][0]
        # 查找current_target body的索引
        target_body_id = mj_model.body('current_target').id
        # 更新mocap body的位置
        mj_data.mocap_pos[0] = target_ee_pos
    
    try:
        last_print_time = 0
        
        for i in range(config.sim_iterations):
            # SBMPC步进
            sim.step()
            
            # 获取当前状态
            current_state = sim.state_traj[i+1, :]
            
            # 检查NaN
            if jnp.any(jnp.isnan(current_state)):
                print(f"\n⚠️ NaN detected at step {i}!")
                break
            
            # 更新MuJoCo
            mj_data.qpos[0:3] = current_state[0:3]
            mj_data.qpos[3:7] = current_state[3:7]
            if mj_model.nq > 7:
                mj_data.qpos[7:9] = current_state[7:9]
            
            mj_data.qvel[0:3] = current_state[9:12]
            mj_data.qvel[3:6] = current_state[12:15]
            if mj_model.nv > 6:
                mj_data.qvel[6:8] = current_state[15:17]
            
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            
            # 定期打印状态
            current_time = i * config.MPC.dt
            if current_time - last_print_time >= 1.0:  # 每秒打印
                pos = current_state[0:3]
                target_pos = scenario['target_pos']
                pos_error = np.linalg.norm(pos - target_pos)
                vel_mag = np.linalg.norm(current_state[9:12])
                
                print(f"t={current_time:5.1f}s | "
                      f"Pos=[{pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}] | "
                      f"Err={pos_error:.4f}m | "
                      f"Vel={vel_mag:.3f}m/s")
                
                last_print_time = current_time
            
            time.sleep(config.MPC.dt)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        viewer.close()


def run_headless(sim, config, scenario):
    """无可视化运行（更快）"""
    print("\nRunning headless simulation...")
    
    progress_steps = config.sim_iterations // 10
    
    for i in range(config.sim_iterations):
        sim.step()
        
        # 进度条
        if i % progress_steps == 0:
            progress = (i / config.sim_iterations) * 100
            print(f"Progress: {progress:.0f}%")
        
        # 检查NaN
        current_state = sim.state_traj[i+1, :]
        if jnp.any(jnp.isnan(current_state)):
            print(f"\n⚠️ NaN detected at step {i}!")
            break
    
    print("Simulation complete!")

def diagnose_controller(sim, scenario):
    """诊断控制器行为"""
    print("\n" + "="*70)
    print("CONTROLLER DIAGNOSTICS")
    print("="*70)
    
    # 检查增益矩阵
    if hasattr(sim.controller, 'gains_obj'):
        gains = sim.controller.gains_obj.cur_gains
        print(f"\nFeedback Gains Matrix Shape: {gains.shape}")
        print(f"Gains Norm: {np.linalg.norm(gains):.6f}")
        print(f"Max Gain Element: {np.max(np.abs(gains)):.6f}")
        
        if np.linalg.norm(gains) < 1e-6:
            print("⚠️ WARNING: Gains are essentially zero!")
            print("   This means feedback control is not working.")
    else:
        print("⚠️ WARNING: No gains object found!")
    
    # 检查控制输入统计
    if len(sim.input_traj) > 0:
        thrust = sim.input_traj[:, 0]
        torques = sim.input_traj[:, 1:4]
        
        print(f"\nThrust Statistics:")
        print(f"  Mean: {np.mean(thrust):.3f} N (Expected: {MASS_TOTAL * GRAVITY:.3f} N)")
        print(f"  Std:  {np.std(thrust):.3f} N")
        print(f"  Min:  {np.min(thrust):.3f} N")
        print(f"  Max:  {np.max(thrust):.3f} N")
        
        # 检查推力偏差
        thrust_bias = np.mean(thrust) - MASS_TOTAL * GRAVITY
        if abs(thrust_bias) > 0.1:
            print(f"  ⚠️ Thrust bias detected: {thrust_bias:.3f} N")
        
        print(f"\nTorque Statistics:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"  {axis}-axis: Mean={np.mean(torques[:, i]):.4f}, "
                  f"Std={np.std(torques[:, i]):.4f}")
    
    # 检查状态轨迹
    if len(sim.state_traj) > 10:
        # 检查前10步的行为
        early_states = sim.state_traj[:10]
        early_pos = early_states[:, 0:3]
        early_vel = early_states[:, 9:12]
        
        print(f"\nEarly Trajectory Analysis (first 10 steps):")
        print(f"  Initial position error: {np.linalg.norm(early_pos[0] - scenario['target_pos']):.4f} m")
        print(f"  Position drift: {np.linalg.norm(early_pos[-1] - early_pos[0]):.4f} m")
        print(f"  Max velocity: {np.max(np.linalg.norm(early_vel, axis=1)):.4f} m/s")
        
        # 检测发散点
        pos_errors = [np.linalg.norm(sim.state_traj[i, 0:3] - scenario['target_pos']) 
                     for i in range(len(sim.state_traj))]
        
        divergence_threshold = 0.2  # 20cm
        divergence_step = None
        for i, err in enumerate(pos_errors):
            if err > divergence_threshold:
                divergence_step = i
                break
        
        if divergence_step is not None:
            divergence_time = divergence_step * 0.02  # assuming dt=0.02
            print(f"\n⚠️ System diverges at step {divergence_step} (t={divergence_time:.2f}s)")
            print(f"   Position error at divergence: {pos_errors[divergence_step]:.4f} m")
            
            # 分析发散时的状态
            div_state = sim.state_traj[divergence_step]
            div_vel = div_state[9:12]
            print(f"   Velocity at divergence: [{div_vel[0]:.3f}, {div_vel[1]:.3f}, {div_vel[2]:.3f}] m/s")
            
            if divergence_step > 0:
                div_input = sim.input_traj[divergence_step-1]
                print(f"   Control at divergence: Thrust={div_input[0]:.3f} N")
    
    return gains if 'gains' in locals() else None


def analyze_results(sim, scenario, config):
    """分析测试结果"""
    print("\n" + "="*70)
    print("Results Analysis")
    print("="*70)
    
    results = {}
    
    # 最终状态
    final_state = sim.state_traj[-1, :]
    final_pos = final_state[0:3]
    final_vel = final_state[9:12]
    final_joints = final_state[7:9]
    
    target_pos = scenario['target_pos']
    target_joints = scenario['target_joints']
    
    # 计算误差
    pos_error = np.linalg.norm(final_pos - target_pos)
    vel_magnitude = np.linalg.norm(final_vel)
    joint_error = np.linalg.norm(final_joints - target_joints)
    
    results['final_pos_error'] = pos_error
    results['final_vel_magnitude'] = vel_magnitude
    results['final_joint_error'] = joint_error
    
    print(f"\nFinal State:")
    print(f"  Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
    print(f"  Target:   [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
    print(f"  Position error: {pos_error:.4f} m")
    print(f"  Velocity magnitude: {vel_magnitude:.4f} m/s")
    
    if scenario['task_type'] == TaskType.ARM_CONTROL:
        print(f"  Joint angles: [{np.rad2deg(final_joints[0]):.1f}°, "
              f"{np.rad2deg(final_joints[1]):.1f}°]")
        print(f"  Target joints: [{np.rad2deg(target_joints[0]):.1f}°, "
              f"{np.rad2deg(target_joints[1]):.1f}°]")
        print(f"  Joint error: {np.rad2deg(joint_error):.2f}°")
    
    # 稳态性能（最后20%的数据）
    steady_start = int(0.8 * len(sim.state_traj))
    steady_states = sim.state_traj[steady_start:, :]
    
    steady_pos_errors = [
        np.linalg.norm(steady_states[i, 0:3] - target_pos) 
        for i in range(len(steady_states))
    ]
    
    results['steady_avg_error'] = np.mean(steady_pos_errors)
    results['steady_max_error'] = np.max(steady_pos_errors)
    results['steady_std_error'] = np.std(steady_pos_errors)
    
    print(f"\nSteady-State Performance (last 20%):")
    print(f"  Average error: {results['steady_avg_error']:.4f} m")
    print(f"  Maximum error: {results['steady_max_error']:.4f} m")
    print(f"  Error std dev: {results['steady_std_error']:.4f} m")
    
    # 性能评估
    print("\n" + "-"*70)
    if scenario['task_type'] == TaskType.HOVER:
        if pos_error < 0.05 and vel_magnitude < 0.05:
            print("✓✓✓ EXCELLENT! Stable hovering achieved!")
            results['success'] = 'excellent'
        elif pos_error < 0.1:
            print("✓✓ GOOD! Nearly stable")
            results['success'] = 'good'
        elif pos_error < 0.2:
            print("✓ OK - Some drift")
            results['success'] = 'ok'
        else:
            print("✗ FAILED - Unstable")
            results['success'] = 'failed'
            
    elif scenario['task_type'] == TaskType.REACH_POINT:
        if pos_error < 0.1 and vel_magnitude < 0.1:
            print("✓✓✓ EXCELLENT! Target reached accurately!")
            results['success'] = 'excellent'
        elif pos_error < 0.2:
            print("✓✓ GOOD! Target reached with minor error")
            results['success'] = 'good'
        else:
            print("✗ FAILED - Did not reach target")
            results['success'] = 'failed'
            
    elif scenario['task_type'] == TaskType.ARM_CONTROL:
        if pos_error < 0.1 and joint_error < 0.1:
            print("✓✓✓ EXCELLENT! Stable hovering with accurate arm control!")
            results['success'] = 'excellent'
        elif pos_error < 0.2 and joint_error < 0.2:
            print("✓✓ GOOD! Minor deviations")
            results['success'] = 'good'
        else:
            print("✗ FAILED - Lost stability during arm movement")
            results['success'] = 'failed'
    
    return results

def analyze_end_effector_results(sim, scenario, config):
    """分析末端执行器轨迹跟踪结果"""
    print("\n" + "="*70)
    print("End-Effector Trajectory Results")
    print("="*70)
    
    # 计算实际末端执行器轨迹
    ee_positions = []
    for i in range(len(sim.state_traj)):
        state = sim.state_traj[i]
        pos = state[0:3]
        quat = state[3:7]
        q1, q2 = state[7], state[8]
        
        # 归一化四元数
        quat = quat / (jnp.linalg.norm(quat) + 1e-10)
        
        ee_pos = compute_end_effector_position(pos, quat, q1, q2)
        ee_positions.append(ee_pos)
    
    ee_positions = jnp.array(ee_positions)
    
    # 单目标跟踪误差计算
    target_ee = jnp.array(scenario['ee_waypoints'][0])  # 获取单个目标
    
    tracking_errors = []
    for i in range(len(ee_positions)):
        error = jnp.linalg.norm(ee_positions[i] - target_ee)
        tracking_errors.append(error)
    
    tracking_errors = jnp.array(tracking_errors)
    
    print(f"\nEnd-Effector Tracking Performance:")
    print(f"  Target position: [{target_ee[0]:.3f}, {target_ee[1]:.3f}, {target_ee[2]:.3f}] m")
    print(f"  Final position: [{ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f}] m")
    print(f"  Average error: {jnp.mean(tracking_errors):.4f} m")
    print(f"  Maximum error: {jnp.max(tracking_errors):.4f} m")
    print(f"  Final error: {tracking_errors[-1]:.4f} m")
    
    # 稳态性能（最后20%）
    steady_start = int(0.8 * len(tracking_errors))
    steady_errors = tracking_errors[steady_start:]
    print(f"\nSteady-State Performance (last 20%):")
    print(f"  Average error: {jnp.mean(steady_errors):.4f} m")
    print(f"  Maximum error: {jnp.max(steady_errors):.4f} m")
    print(f"  Error std dev: {jnp.std(steady_errors):.4f} m")
    
    # 评估结果
    results = {
        'final_ee_error': tracking_errors[-1],
        'avg_ee_error': jnp.mean(tracking_errors),
        'steady_avg_error': jnp.mean(steady_errors),
        'steady_max_error': jnp.max(steady_errors),
        'steady_std_error': jnp.std(steady_errors)
    }
    
    # 成功判定
    if tracking_errors[-1] < 0.05 and jnp.mean(steady_errors) < 0.05:
        print("\n✓✓✓ EXCELLENT! End-effector reached target accurately!")
        results['success'] = 'excellent'
    elif tracking_errors[-1] < 0.1:
        print("\n✓✓ GOOD! End-effector near target")
        results['success'] = 'good'
    elif tracking_errors[-1] < 0.2:
        print("\n✓ OK - Some error remaining")
        results['success'] = 'ok'
    else:
        print("\n✗ FAILED - Did not reach target")
        results['success'] = 'failed'
    
    # 简化的绘图（只显示到单个目标的跟踪）
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 跟踪误差
    time_vec = config.MPC.dt * jnp.arange(len(tracking_errors))
    axes[0, 0].plot(time_vec, tracking_errors)
    axes[0, 0].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='5cm')
    axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10cm')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Tracking Error [m]')
    axes[0, 0].set_title('End-Effector Tracking Error')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 末端执行器位置分量
    axes[0, 1].plot(time_vec[:-1], ee_positions[:-1, 0], label='X')
    axes[0, 1].plot(time_vec[:-1], ee_positions[:-1, 1], label='Y')
    axes[0, 1].plot(time_vec[:-1], ee_positions[:-1, 2], label='Z')
    axes[0, 1].axhline(y=target_ee[0], color='r', linestyle='--', alpha=0.3)
    axes[0, 1].axhline(y=target_ee[1], color='g', linestyle='--', alpha=0.3)
    axes[0, 1].axhline(y=target_ee[2], color='b', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('End-Effector Position [m]')
    axes[0, 1].set_title('End-Effector Position Components')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 关节角度
    axes[1, 0].plot(time_vec[:-1], np.rad2deg(sim.state_traj[:-1, 7]), label='Joint 1')
    axes[1, 0].plot(time_vec[:-1], np.rad2deg(sim.state_traj[:-1, 8]), label='Joint 2')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Joint Angles [deg]')
    axes[1, 0].set_title('Joint Angles')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 基座位置
    axes[1, 1].plot(time_vec[:-1], sim.state_traj[:-1, 0], label='X')
    axes[1, 1].plot(time_vec[:-1], sim.state_traj[:-1, 1], label='Y')
    axes[1, 1].plot(time_vec[:-1], sim.state_traj[:-1, 2], label='Z')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Base Position [m]')
    axes[1, 1].set_title('UAV Base Position')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

def plot_results(sim, scenario, config):
    """绘制结果图表"""
    # 如果没有传入config，使用sim中的信息
    if config is None:
        dt = 0.02  # 默认值
        time_vec = dt * np.arange(len(sim.state_traj))
    else:
        time_vec = config.MPC.dt * np.arange(len(sim.state_traj))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 位置轨迹
    axes[0, 0].plot(time_vec, sim.state_traj[:, 0:3])
    axes[0, 0].axhline(y=scenario['target_pos'][0], color='r', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=scenario['target_pos'][1], color='g', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=scenario['target_pos'][2], color='b', linestyle='--', alpha=0.3)
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].legend(['x', 'y', 'z'])
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Position')
    
    # 位置误差
    errors = [
        np.linalg.norm(sim.state_traj[i, 0:3] - scenario['target_pos']) 
        for i in range(len(sim.state_traj))
    ]
    axes[0, 1].plot(time_vec, errors)
    axes[0, 1].axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='5cm')
    axes[0, 1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10cm')
    axes[0, 1].set_ylabel('Position Error [m]')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Position Error')
    
    # 速度
    velocities = sim.state_traj[:, 9:12]
    vel_mags = np.linalg.norm(velocities, axis=1)
    axes[0, 2].plot(time_vec, vel_mags)
    axes[0, 2].set_ylabel('Velocity [m/s]')
    axes[0, 2].set_xlabel('Time [s]')
    axes[0, 2].grid(True)
    axes[0, 2].set_title('Velocity Magnitude')
    
    # 关节角度
    axes[1, 0].plot(time_vec, np.rad2deg(sim.state_traj[:, 7:9]))
    if scenario['task_type'] == TaskType.ARM_CONTROL:
        axes[1, 0].axhline(y=np.rad2deg(scenario['target_joints'][0]), 
                          color='r', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=np.rad2deg(scenario['target_joints'][1]), 
                          color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel('Joint Angles [deg]')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].legend(['Joint 1', 'Joint 2'])
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Joint Angles')
    
    # 控制输入
    if len(sim.input_traj) > 0:
        time_vec_ctrl = time_vec[:-1]
        
        # 推力
        axes[1, 1].plot(time_vec_ctrl, sim.input_traj[:, 0])
        axes[1, 1].axhline(y=MASS_TOTAL*GRAVITY, color='r', linestyle='--', 
                          label=f'Hover: {MASS_TOTAL*GRAVITY:.1f}N')
        axes[1, 1].set_ylabel('Thrust [N]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Thrust')
        
        # 扭矩
        axes[1, 2].plot(time_vec_ctrl, sim.input_traj[:, 1:4])
        axes[1, 2].set_ylabel('Torque [Nm]')
        axes[1, 2].set_xlabel('Time [s]')
        axes[1, 2].legend(['τx', 'τy', 'τz'])
        axes[1, 2].grid(True)
        axes[1, 2].set_title('Torques')
    
    # plt.suptitle(f'{scenario["name"]} - Step {dynamics_step}')
    plt.tight_layout()
    plt.show()


def generate_trajectory_reference(waypoints, num_iters, horizon, dt):
    """生成时变轨迹参考"""
    reference = jnp.zeros((num_iters, horizon + 1, 17))

    # 计算轨迹总时长
    total_time = num_iters * dt
    segment_time = total_time / (len(waypoints) - 1)

    for iter_idx in range(num_iters):
        current_time = iter_idx * dt

        # 确定当前在哪个轨迹段
        segment_idx = min(int(current_time / segment_time), len(waypoints) - 2)
        local_time = (current_time - segment_idx * segment_time) / segment_time
        local_time = jnp.clip(local_time, 0.0, 1.0)

        # 线性插值当前位置
        start_pos = jnp.array(waypoints[segment_idx])
        end_pos = jnp.array(waypoints[segment_idx + 1])
        current_pos = start_pos + local_time * (end_pos - start_pos)

        # 生成预测时域参考
        for h in range(horizon + 1):
            future_time = current_time + h * dt
            future_segment_idx = min(int(future_time / segment_time), len(waypoints) - 2)
            future_local_time = (future_time - future_segment_idx * segment_time) / segment_time
            future_local_time = jnp.clip(future_local_time, 0.0, 1.0)

            future_start = jnp.array(waypoints[future_segment_idx])
            future_end = jnp.array(waypoints[future_segment_idx + 1])
            future_pos = future_start + future_local_time * (future_end - future_start)

            # 填充参考状态
            reference = reference.at[iter_idx, h, 0:3].set(future_pos)
            reference = reference.at[iter_idx, h, 3:7].set(jnp.array([1., 0., 0., 0.]))  # 四元数
            reference = reference.at[iter_idx, h, 7:9].set(jnp.zeros(2))  # 关节角度  
    return reference

def generate_end_effector_trajectory_reference(ee_waypoints, num_iters, horizon, dt):
    """生成末端执行器轨迹参考 - 改为2D数组版本"""
    # 直接生成2D数组 (horizon+1, 17)
    reference = jnp.zeros((horizon + 1, 17))
    
    # 简化为单目标测试：使用第一个waypoint作为固定目标
    target_ee_pos = jnp.array(ee_waypoints[0])  # 或者使用ee_waypoints[-1]作为最终目标
    
    # 为整个预测时域设置相同的目标
    for h in range(horizon + 1):
        reference = reference.at[h, 0:3].set(target_ee_pos)  # 末端执行器目标位置
        reference = reference.at[h, 3:7].set(jnp.array([1., 0., 0., 0.]))  # 期望姿态
        # 其余元素保持为0
    
    return reference

# ============================================================================
# 批量测试运行器
# ============================================================================
def run_progressive_tests():
    """运行完整的渐进式测试序列"""
    
    print("\n" + "="*70)
    print("PROGRESSIVE TESTING SEQUENCE")
    print("="*70)
    
    all_results = {}
    
    # 测试序列
    test_sequence = [
        # 步骤1：基础测试
        (TestScenario.hover_test(height=1.5), 1),
        (TestScenario.reach_point_test([1.0, 0.0, 1.5]), 1),
        (TestScenario.reach_point_test([0.0, 1.0, 2.0]), 1),
        
        # 步骤2：添加重心补偿
        (TestScenario.hover_test(height=1.5), 2),
        (TestScenario.arm_control_test([0.0, 0.0, 1.5], [0.2, -0.2]), 2),
        
        # 步骤3：完整动力学（如果稳定）
        # (TestScenario.hover_test(height=1.5), 3),
        # (TestScenario.arm_control_test([0.0, 0.0, 1.5], [0.3, -0.3]), 3),
    ]
    
    for i, (scenario, step) in enumerate(test_sequence):
        print(f"\n\nTest {i+1}/{len(test_sequence)}")
        print("-"*70)
        
        try:
            sim, results = run_test_with_diagnostics(scenario, step, visualize=False)
            all_results[f"test_{i+1}"] = {
                'scenario': scenario['name'],
                'step': step,
                'results': results
            }
            
            # 如果测试失败，询问是否继续
            if results['success'] == 'failed':
                response = input("\nTest failed. Continue? (y/n): ")
                if response.lower() != 'y':
                    break
                    
        except Exception as e:
            print(f"\n⚠️ Test failed with error: {e}")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                break
    
    # 总结
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, test_data in all_results.items():
        print(f"\n{test_name}:")
        print(f"  Scenario: {test_data['scenario']}")
        print(f"  Step: {test_data['step']}")
        print(f"  Result: {test_data['results']['success']}")
        print(f"  Final error: {test_data['results']['final_pos_error']:.4f}m")
    
    return all_results


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quadrotor-Arm Progressive Testing')
    parser.add_argument('--test', type=str, default='hover',
                       choices=['hover', 'reach', 'arm', 'trajectory', 
                               'ee_trajectory', 'all'],
                       help='Test type to run')
    parser.add_argument('--step', type=int, default=1, choices=[1, 2, 3],
                       help='Dynamics complexity step')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable MuJoCo visualization')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results after simulation')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        # 运行完整测试序列
        run_progressive_tests()
    else:
        # 运行单个测试
        if args.test == 'hover':
            scenario = TestScenario.hover_test()
        elif args.test == 'reach':
            scenario = TestScenario.reach_point_test([1.0, 0.5, 2.0])
        elif args.test == 'arm':
            scenario = TestScenario.arm_control_test([0.0, 0.0, 1.5], [1.5, 0.3])
        elif args.test == 'ee_trajectory':
            # 定义末端执行器目标轨迹
            ee_target = [0.1, 0, 12]  # 单个目标点
            scenario = TestScenario.end_effector_trajectory_test(ee_target, duration=15.0)
        elif args.test == 'trajectory':
            waypoints = [[0, 0, 1.5], [1, 0, 1.5], [1, 1, 2.0], [0, 1, 2.0], [0, 0, 1.5]]
            scenario = TestScenario.trajectory_test(waypoints)
        
        sim, results = run_test_with_diagnostics(scenario, args.step, args.visualize)
        
        if args.plot:
            plot_results(sim, scenario, None)
        
        print(f"\nTest completed with result: {results['success'].upper()}")


        # python .\examples\quadrotor_arm_test.py --test ee_trajectory --step 2 --visualize