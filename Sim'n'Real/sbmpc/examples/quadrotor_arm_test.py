#!/usr/bin/env python3
"""
无人机-机械臂系统渐进式测试框架
支持三个步骤和多种任务的测试
"""

import os
import socket
import struct
import sys
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import time
import json
import threading
from enum import Enum
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dynamixel_sdk import *  # Dynamixel SDK


from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_all
from sbmpc.geometry import quat_product, quat2rotm, quat_inverse




# 导入稳定的动力学模型
from drone_arm_dynamics_stable import (
    dynamics_step1,
    dynamics_step2,
    dynamics_step3,
    MASS_TOTAL, GRAVITY,
    compute_com_offset,
    compute_end_effector_position
)

# 导入任务配置
from task_configs import TaskConfig

USING_ARM_POSITION = False
USING_WIFI_RASPI = False
USING_ROS2_PHASESPACE = True

if USING_WIFI_RASPI:
    raspi_ip =  '192.168.0.206'
    raspi_port = 5606
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5605))   # 监听本机 5005 端口

def recv_latest(sock, fmt="ffi", bufsize=1024):
    """
    从UDP缓冲区取出所有数据，只返回 i 最大的那条
    :param sock: 已经 bind 的 UDP socket
    :param fmt: struct 格式串，默认 "ffi" (float, float, int)
    :param bufsize: 单个 UDP 包最大字节数
    :return: (unpacked_data, addr) 或 None
    """
    size = struct.calcsize(fmt)
    newest = None
    max_i = -float("inf")

    sock.setblocking(False)  # 设置非阻塞
    while True:
        try:
            data, addr = sock.recvfrom(bufsize)
            values = struct.unpack(fmt, data[:size])
            *others, i = values
            if i > max_i:
                max_i = i
                newest = values
        except BlockingIOError:
            break  # 缓冲区已空

    sock.setblocking(True)   # 恢复阻塞
    return newest


if USING_ROS2_PHASESPACE:

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(("0.0.0.0", 5005))
    sock2.setblocking(False)


    def recv_latest_json(sock, bufsize=4096):
        """
        从UDP缓冲区读取所有JSON包，只返回time最大的那条
        """
        newest, max_time = None, -1
        while True:
            try:
                data, addr = sock.recvfrom(bufsize)
                msg = json.loads(data.decode())
                if msg["time"] > max_time:
                    max_time = msg["time"]
                    newest = msg
            except BlockingIOError:
                break
        return newest


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


class RealRobotController:
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=1000000, dxl_ids=[1, 2]):
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.DXL_IDS = dxl_ids

        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116  # 4 bytes
        self.LEN_GOAL_POSITION = 4
        self.ADDR_PRESENT_POSITION = 132  # X 系列 Present Position
        self.ADDR_PRESENT_VELOCITY = 128  # X 系列 Present Velocity
        self.LEN_4 = 4
        # self.TORQUE_ENABLE = 1

        self.angle_offset = [-115,-180]
        self.max_torque_per_joint = [4.5, 1]
        self.max_pwm_val = 855

        self.e_break = [0.235, 0.372]  # 电子制动系数
        self.stall_torque = [3.0, 1.5]  # 每个关节的额定力矩（Nm）
        self.torque_limit = [0.5, 0.2]  # 每个关节的力矩限制（Nm）
        self.full_speed = [77 * 2 * np.pi / 60 , 57 * 2 * np.pi / 60]  # 每个关节的最大速度（弧度/秒）


        # 初始化串口和协议处理器
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # 打开串口
        if not self.portHandler.openPort():
            raise RuntimeError("❌ 串口打开失败")
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("❌ 设置波特率失败")

        # 初始化同步写对象
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION
        )

        # 初始化 GroupSyncRead
        self.groupSyncRead = GroupSyncRead(
            self.portHandler, self.packetHandler,
            self.ADDR_PRESENT_VELOCITY, 8
        )
        for dxl_id in self.DXL_IDS:
            if not self.groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"GroupSyncRead addParam failed: ID={dxl_id}")

        # 当前模式缓存（ID → mode），mode：0=PWM, 1=Current, 3=Position 等
        self.current_mode_map = {dxl_id: None for dxl_id in self.DXL_IDS}
        self.pwmgroupWrite = GroupSyncWrite(self.portHandler, self.packetHandler, 100, 2)


    def _set_control_mode_if_needed(self, dxl_id, target_mode):
        """
        如果当前模式不是目标模式，则切换舵机控制模式
        target_mode: int，0=PWM, 3=Position 等
        """
        MODE_ADDR = 11  # Control Mode
        if self.current_mode_map[dxl_id] != target_mode:
            # 切换模式流程：关闭力矩 → 修改模式 → 启用力矩
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, MODE_ADDR, target_mode)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)
            self.current_mode_map[dxl_id] = target_mode

    def get_joint_positions(self):
        """
        返回当前舵机位置（单位：弧度），长度与 self.DXL_IDS 一致
        """
        current_positions = []
        for dxl_id in self.DXL_IDS:
            pos_result, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID {dxl_id}] 读取失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                continue
            elif dxl_error != 0:
                print(f"[ID {dxl_id}] 错误: {self.packetHandler.getRxPacketError(dxl_error)}")
                continue

            # Dynamixel 位置值是 0~4095，映射到 0~360°
            degree = (pos_result / 4095.0) * 360.0
            rad = np.radians(degree)  # 转换为弧度
            current_positions.append(rad)

        return current_positions

    def degree_to_position(self, degree):
        degree = degree % 360
        return int((degree / 360.0) * 4095)
    
    def send_joint_positions(self, degrees, check_response=False):
        """
        发送目标角度（单位：度）给舵机，自动切换至 Position 控制模式
        degrees: List[float]，每个关节的目标角度（0~360 度）
        """
        assert hasattr(self, "DXL_IDS"), "DXL_IDS 尚未初始化"
        assert len(degrees) == len(self.DXL_IDS), "角度与舵机ID数量不一致"

        ADDR_GOAL_POSITION = self.ADDR_GOAL_POSITION
        LEN_GOAL_POSITION = self.LEN_GOAL_POSITION
        POSITION_MODE = 3

        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        groupSyncWrite.clearParam()

        for dxl_id, degree in zip(self.DXL_IDS, degrees):
            # 设置模式（如果需要）
            self._set_control_mode_if_needed(dxl_id, POSITION_MODE)

            pos_val = self.degree_to_position(degree)
            param_goal_pos = [
                pos_val & 0xFF,
                (pos_val >> 8) & 0xFF,
                (pos_val >> 16) & 0xFF,
                (pos_val >> 24) & 0xFF
            ]
            success = groupSyncWrite.addParam(dxl_id, param_goal_pos)
            if not success and check_response:
                print(f"[ID {dxl_id}] ❌ 添加同步参数失败")

        result = groupSyncWrite.txPacket()

        if check_response:
            if result != COMM_SUCCESS:
                print("❌ 同步写入失败:", self.packetHandler.getTxRxResult(result))
            else:
                print("✅ 同步写入成功")

    def send_pwm(self, pwm_vals, check_response=False):
        """
        发送 PWM 控制信号到多个舵机
        pwm_vals: List[int]，单位为 [-885, 885]，与 self.DXL_IDS 一一对应
        自动切换至 PWM 控制模式
        """
        assert len(pwm_vals) == len(self.DXL_IDS), "PWM 数量必须与舵机 ID 数量一致"

        ADDR_GOAL_PWM = 100
        LEN_PWM = 2
        PWM_MODE = 16  # 0 = PWM 控制模式

        groupSyncWrite = self.pwmgroupWrite
        groupSyncWrite.clearParam()

        for dxl_id, pwm in zip(self.DXL_IDS, pwm_vals):
            # 自动检查并设置控制模式为 PWM
            self._set_control_mode_if_needed(dxl_id, PWM_MODE)

            pwm = int(np.clip(pwm, -885, 885))
            param_goal_pwm = [pwm & 0xFF, (pwm >> 8) & 0xFF]
            success = groupSyncWrite.addParam(dxl_id, param_goal_pwm)
            if not success and check_response:
                print(f"[ID {dxl_id}] ❌ 添加 PWM 参数失败")

        result = groupSyncWrite.txPacket()

        if check_response:
            if result != COMM_SUCCESS:
                print("❌ PWM 同步写入失败:", self.packetHandler.getTxRxResult(result))
            else:
                print("✅ PWM 同步写入成功")

    def send_torque(self, torque_vals, check_response=False):
        """
        输入力矩（单位 Nm），自动转换为 PWM 并发送给舵机
        torque_vals: List[float]，与 dxl_ids 一一对应
        """
        assert len(torque_vals) == len(self.DXL_IDS), "力矩数量与舵机 ID 不一致"

        #apply torque limits
        for i in range(len(torque_vals)):
            torque_vals[i] = np.clip(torque_vals[i], -self.torque_limit[i], self.torque_limit[i])

        pwm_vals = []
        for idx, tau in enumerate(torque_vals):
            max_tau = self.max_torque_per_joint[idx]
            pwm = (tau / max_tau) * self.max_pwm_val
            pwm_vals.append(int(np.clip(pwm, -self.max_pwm_val, self.max_pwm_val)))

        self.send_pwm(pwm_vals, check_response=check_response)

    def close(self):
        self.portHandler.closePort()

    def get_joint_state(self):
        # 一次性请求
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("SyncRead failed:", self.packetHandler.getTxRxResult(dxl_comm_result))
            return [0.0]*len(self.DXL_IDS), [0.0]*len(self.DXL_IDS)

        qpos, qvel = [], []

        for i, dxl_id in enumerate(self.DXL_IDS):
            if not self.groupSyncRead.isAvailable(dxl_id, self.ADDR_PRESENT_VELOCITY, 8):
                qpos.append(0.0)
                qvel.append(0.0)
                continue

            vel_bytes = self.groupSyncRead.getData(dxl_id, 128, 4).to_bytes(4,'little',signed=False)
            pos_bytes = self.groupSyncRead.getData(dxl_id, 132, 4).to_bytes(4,'little',signed=False)
            vel_val = int.from_bytes(vel_bytes,'little',signed=True)
            pos_val = int.from_bytes(pos_bytes,'little',signed=False)


            # 速度：带符号，单位 0.229 rpm/bit
            rpm = (vel_val / 1023.0) * 117.0
            rad_per_sec = rpm * 2 * np.pi / 60
            qvel.append(rad_per_sec)

            # 位置：无符号，0~4095 → 0~360deg
            degree = (pos_val / 4095.0) * 360.0 + self.angle_offset[i]
            qpos.append(np.radians(degree))

        return qpos, qvel

DEVICE_TYPE = setup_compute_device()

if sys.platform.startswith("linux"):
    device_path = "/dev/ttyUSB0"   # Ubuntu / Linux
elif sys.platform == "darwin":
    device_path = "/dev/tty.usbserial-FT9HDB5F"  # macOS
elif sys.platform == "win32":
    device_path = "COM7"  # Windows
else:
    raise RuntimeError(f"❌ 不支持的操作系统: {sys.platform}")

try:
    real_controller = RealRobotController( device_name=device_path )

except Exception as e:
    print(f"❌ 初始化 RealRobotController 失败: {e}")
    real_controller = None

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
        
    def running_cost(self, state, inputs, reference, steps):
        # ── 兼容 reference 既可能是一维 (17,) 也可能是二维 (H,17) ──
        if reference.ndim == 2:        # e.g. (horizon+1, 17)
            reference = reference[0]   # 只取当前时刻那一行 → (17,)
        # 解析状态
        pos       = state[0:3]
        quat      = state[3:7]
        q_joints  = state[7:9]
        vel       = state[9:12]
        omega     = state[12:15]
        dq_joints = state[15:17]

        # 归一化四元数
        quat = quat / (jnp.linalg.norm(quat) + 1e-10)

        cost = 0.0

        ref_ee_pos = reference[0:3]

        # 末端位置误差
        ee_pos   = compute_end_effector_position(pos, quat, q_joints[0], q_joints[1])
        ee_error = ee_pos - ref_ee_pos
        ee_err_norm = jnp.linalg.norm(ee_error)

        # compute drone pos error
        drone_error = ref_ee_pos - pos
        drone_err_norm = jnp.linalg.norm(drone_error)

        if USING_ARM_POSITION:
            # 1) 末端误差（远处中等，近处加大）——不归一化
            ee_weight = jnp.where(ee_err_norm < 0.12, 500.0, 180.0)  # 12cm 内强化
            cost += ee_weight * jnp.sum(jnp.where(ee_err_norm < 1.0, ee_err_norm**2, drone_err_norm**2))
        else:
            # 1) 末端误差（远处中等，近处加大）——不归一化
            ee_weight = jnp.where(drone_err_norm < 0.12, 500.0, 180.0)  # 12cm 内强化
            cost += ee_weight * jnp.sum(drone_err_norm**2)

        # 2) 平动速度（抑制飘）
        cost += 8.0 * jnp.sum(vel**2)

        # 3) 角速度（防晃）
        cost += 3.0 * jnp.sum(omega**2)

        # 4) 倾角（不强追姿态参考，只要别过度倾斜）
        R = quat2rotm(quat)
        tilt = 1.0 - R[2, 2]              # 水直对齐越好，值越小
        cost += 12.0 * (tilt**2)

        # 5) 关节速度（抑制手臂抖动）
        cost += 7.5 * jnp.sum(dq_joints**2)

        # 6) 输入正则（能量项）
        thrust_diff = inputs[0] - MASS_TOTAL * GRAVITY
        cost += 0.15 * (thrust_diff**2)         # 推力偏离悬停
        cost += 0.40 * jnp.sum(inputs[1:4]**2)  # 机体扭矩
        cost += 5.00 * jnp.sum(inputs[4:6]**2)  # 关节扭矩

        if True:
            cost += 150.0 * jnp.abs(q_joints[0] - reference[7])
            cost += 150.0 * jnp.abs(q_joints[1] - reference[8])

        # ========= 公共项（瘦身后版）=========
        # 对于 END_EFFECTOR_TRAJECTORY，不再额外追 ref_quat、不再加期望角速度/期望推力
        # 边界约束和关节极限可选：只留“硬碰硬”的一条，避免过多项相互打架
        joint_limit = 1.3
        j0_excess = jnp.maximum(jnp.abs(q_joints[0]) - joint_limit, 0.0)
        j1_excess = jnp.maximum(jnp.abs(q_joints[1]) - joint_limit, 0.0)
        cost += 1000.0 * (j0_excess**2 + j1_excess**2)

        # NaN/Inf 保护
        cost = jnp.where(jnp.isnan(cost), 1e6, cost)
        cost = jnp.where(jnp.isinf(cost), 1e6, cost)
        return jnp.clip(cost, 0.0, 1e6)

    def final_cost(self, state, reference, steps):
        # 允许 reference 为 (17,) 或 (H,17) 等
        if reference.ndim == 2:
            reference = reference[steps]      # 扁平化
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
    def arm_control_test(hover_pos, joint_targets, duration=10.0):
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
    def arm_swing_test(duration=20.0,
                    amp_deg=(20.0, 15.0),
                    freq_hz=(0.5, 0.8)):
        """飞机悬停，机械臂正弦摆动"""
        return {
            'name'      : 'Arm Swing Test',
            'task_type' : TaskType.ARM_CONTROL,   # 复用现有模式
            'swing_amp' : amp_deg,
            'swing_freq': freq_hz,
            'target_pos': jnp.array([0.0, 0.0, 1.5]),
            'duration'  : duration
        }

    
    @staticmethod
    def trajectory_test(waypoints=None, duration=20.0, pattern=None, **kwargs):
        """轨迹跟踪测试：支持多点或周期性预设路径"""
        # 若指定预设模式，则生成对应的 waypoints
        if pattern is not None:
            if pattern == "circle":
                # 圆轨迹参数：半径、中心、高度、点数、循环次数
                r = kwargs.get('radius', 1.0)             # 圆半径
                center = kwargs.get('center', (0.0, 0.0)) # 圆心在水平面的坐标
                height = kwargs.get('height', 1.5)        # 圆轨迹所在高度
                points = kwargs.get('points', 20)         # 单圈轨迹离散点数
                cycles = kwargs.get('cycles', 1)          # 重复圈数
                # 生成一个圆轨迹的离散点列表（首尾相接）
                base_points = []
                for k in range(points):
                    theta = 2 * np.pi * k / points
                    x = center[0] + r * np.cos(theta)
                    y = center[1] + r * np.sin(theta)
                    base_points.append([x, y, height])
                base_points.append(base_points[0])  # 闭合轨迹，重复起点
                # 按需求重复多圈
                waypoints = []
                for c in range(cycles):
                    # 最后一圈之后不再重复终点，以免出现重复点停留
                    segment = base_points if c == cycles - 1 else base_points[:-1]
                    waypoints.extend(segment)
            else:
                raise ValueError(f"未识别的轨迹模式: {pattern}")
        if waypoints is None:
            raise ValueError("必须提供 waypoints 列表或指定 pattern")
        return {
            'name': 'Trajectory Test',
            'task_type': TaskType.TRAJECTORY,
            'waypoints': jnp.array(waypoints, dtype=jnp.float32),
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
    robot_config.input_min = jnp.array([0., -1.0, -1.0, -1.0, -1.5, -1.5])
    robot_config.input_max = jnp.array([20., 1.0, 1.0, 1.0, 1.5, 1.5])
    
    # 初始状态（末端执行器模式使用非零关节角度）
    if scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        
        robot_config.q_init = jnp.array([
            0., 0., 1.5,      # 位置
            1., 0., 0., 0.,   # 四元数
            0.0, -0.0         # 初始关节角度（非零）
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

    elif scenario['task_type'] == TaskType.ARM_CONTROL and 'swing_amp' in scenario:
        # 机械臂周期摆动
        reference = generate_arm_swing_reference(
            num_iters   = config.sim_iterations,
            horizon     = config.MPC.horizon,
            dt          = config.MPC.dt,
            amp_deg     = scenario['swing_amp'],
            freq_hz     = scenario['swing_freq'],
            base_height = scenario['target_pos'][2],
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
        dynamics_fn = dynamics_step3
    
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
    elif scenario['task_type'] == TaskType.TRAJECTORY:
        results = analyze_trajectory_results(sim, scenario, config)
    elif scenario['task_type'] == TaskType.ARM_CONTROL:
        results = analyze_arm_control_results(sim, scenario, config)
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

def quat_mul(a, b):
    w1,x1,y1,z1 = a
    w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def run_with_visualization(sim, config, scenario):
    """带MuJoCo可视化运行"""
    print("\nRunning with MuJoCo visualization...")
    use_real = (real_controller is not None) or (USING_WIFI_RASPI is True)
    print(f"Using real robot controller: {use_real}")
    
    mj_model = mujoco.MjModel.from_xml_path("examples/drone_direct_control.xml")
    mj_data = mujoco.MjData(mj_model)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    # 设置相机
    viewer.cam.distance = 8.0
    viewer.cam.elevation = -90
    viewer.cam.azimuth = 0

    # 如果是末端执行器任务，更新目标球位置
    if scenario['task_type'] == TaskType.END_EFFECTOR_TRAJECTORY:
        # 获取目标末端执行器位置
        target_ee_pos = scenario['ee_waypoints'][0]
        # 查找current_target body的索引
        target_body_id = mj_model.body('current_target').id
        # 更新mocap body的位置
        mj_data.mocap_pos[0] = target_ee_pos
    
    try:        
        start_time = time.time()
        for i in range(config.sim_iterations):

            if i == 1:
                start_time = time.time()

            ctrl = sim.input_traj[i, :]
            current_state = sim.state_traj[i, :]
            arm_torques = ctrl[4:6]

            if USING_ROS2_PHASESPACE:
                newest = recv_latest_json(sock2)
                pos = quat = lin_vel = ang_vel = None
                if newest:
                    [x,z,y]  = np.asarray(newest["pos"], dtype=float) *0.001       # m
                    pos = [x,-y,z]
                    # print(f'{x:.2f} {-y:.2f} {z:.2f}')


                    [w,x,y,z] = np.asarray(newest["quat"], dtype=float)       # [w,x,y,z]  
                    q_fix = np.array([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])
                    quat = quat_mul([w,x,-z,y], q_fix)
                    # print(f'{x:.2f} {y:.2f} {z:.2f}')  


                    [x,z,y] = np.asarray(newest["lin_vel"], dtype=float) *0.001  # m/s
                    lin_vel = [x,y,z]
                    # print(f'{x:.2f} {y:.2f} {z:.2f}')


                    [x,z,y] = np.asarray(newest["ang_vel"], dtype=float) # rad/s
                    ang_vel = [x,-y,z]
                    # print(f'{x:.2f} {y:.2f} {z:.2f}')



            # Get real arm state
            if use_real is True:
                get_state_start = time.time()
                if USING_WIFI_RASPI:
                    values = recv_latest(sock, "ffffi")
                    if values is not None:
                        real_state = [list(values[:2]), list(values[2:4])]
                    else:
                        real_state = [current_state[7:9], current_state[15:17]]
                else:
                    real_state = real_controller.get_joint_state()
                get_state_end = time.time()
                print(f"Get state time: {(get_state_end - get_state_start)*1000:.2f} ms")
                q_arm  = jnp.asarray(real_state[0], dtype=jnp.float32)  # 形状 (2,)
                dq_arm = jnp.asarray(real_state[1], dtype=jnp.float32)  # 形状 (2,)
            
            # Get real Phasespace state
            if USING_ROS2_PHASESPACE and quat is not None:
                mj_data.qpos[0:3] = pos
                mj_data.qpos[3:7] = quat
            else:
                # 更新MuJoCo
                mj_data.qpos[0:3] = current_state[0:3]
                mj_data.qpos[3:7] = current_state[3:7]

            if mj_model.nq > 7:
                if use_real is True:
                    mj_data.qpos[7:9] = real_state[0]
                    sim.state_traj[i+1, :][7:9] = real_state[0]
                    sim.current_state = sim.current_state.at[7:9].set(q_arm)
                else:
                    mj_data.qpos[7:9] = current_state[7:9]


            # Add noise test
            if i == 30 and False:
                # 1) 修改 MuJoCo 内部速度
                mj_data.qvel[6:8] = np.array([2.0, 0.0])

                # 2) 修改仿真记录 —— 用 i+1 才会参与下一步积分
                sim.state_traj[i+1, 15:17] = np.array([2.0, 0.0], dtype=np.float32)

                # 3) 修改当前 JAX 状态；一定要“重新赋回”
                sim.current_state = sim.current_state.at[15:17].set(jnp.array([2.0, 0.0], dtype=jnp.float32))

            if USING_ROS2_PHASESPACE  and lin_vel is not None:
                mj_data.qvel[0:3] = lin_vel
                mj_data.qvel[3:6] = ang_vel
            else:
                # 更新MuJoCo
                mj_data.qvel[0:3] = current_state[9:12]
                mj_data.qvel[3:6] = current_state[12:15]
            if mj_model.nv > 6:
                if use_real is True:
                    mj_data.qvel[6:8] = real_state[1]
                    sim.state_traj[i+1, :][15:17] = real_state[1]
                    sim.current_state = sim.current_state.at[15:17].set(dq_arm)
                else:
                    mj_data.qvel[6:8] = current_state[15:17]



            sim_start = time.time()
            # SBMPC步进
            sim.step()
            sim_end = time.time()
            print(f"Sim step time: {(sim_end - sim_start)*1000:.2f} ms")



            # 发送控制到真实机器人
            if use_real is True:
                send_torque_start = time.time()
                if USING_WIFI_RASPI:
                    sock.sendto(struct.pack("ffi", arm_torques[0], arm_torques[1], i), (raspi_ip, raspi_port))
                else:
                    real_controller.send_torque(arm_torques) #0.25ms
                send_torque_end = time.time()
                print(f"Send torque time: {(send_torque_end - send_torque_start)*1000:.2f} ms")
            
            # # 检查NaN
            # if jnp.any(jnp.isnan(current_state)):
            #     print(f"\n⚠️ NaN detected at step {i}!")
            #     break


            if scenario['task_type'] == TaskType.TRAJECTORY:
                # 对于轨迹任务，更新目标球到第一个路径点
                target_ee_pos = sim.const_reference[i, 0][0:3]  # or reference[i, 0:3]
                mj_data.mocap_pos[1] = target_ee_pos
            
            mj_start = time.time()
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            mj_end = time.time()

            # print(f"Mujoco step time: {(mj_end - mj_start)*1000:.2f} ms")
            
            # 定期打印状态
            current_sim_time = i * config.MPC.dt

            
            # 实时同步
            err_time = current_sim_time - (time.time() - start_time)
            # print(f"now time = {time.time()}, start_time = {start_time}, current_time = {current_sim_time}, err_time = {err_time*1000}ms")
            if err_time < 0:
                print(f"⚠️ Warning: Simulation is lagging behind real time by {-err_time*1000}ms")
            else:
                time.sleep(err_time)

            # time.sleep(0.1)
            
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  # 激活 3-D 支持

def analyze_trajectory_results(sim, scenario, config):
    """
    画出实际轨迹 vs 参考轨迹（2-D、3-D），并给出误差统计。
    依赖：
        sim.state_traj                (T+1, 17)  实际状态轨迹
        sim.const_reference 或 scenario['reference'] (T, H+1, 17)  参考轨迹
    """
    print("\n" + "=" * 70)
    print("Trajectory Tracking Results")
    print("=" * 70)



    # ---------- 1. 取得实际轨迹 ----------
    actual_pos = np.asarray(sim.state_traj[:, 0:3])            # (T+1, 3)
    actual_pos = actual_pos[:-1]                               # 与参考长度对齐 (T, 3)
    sim_steps  = actual_pos.shape[0]
    time_vec   = np.arange(sim_steps) * config.MPC.dt          # (T,)

    # ------------- 若使用末端执行器位置作为轨迹 -------------
    if USING_ARM_POSITION:
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
        actual_pos =  ee_positions[:-1]    

    # ----------——————————————————————————

    # ---------- 2. 取得参考轨迹 ----------
    # 优先使用 simulation 内部的完整 reference
    if hasattr(sim, "const_reference"):
        ref_pos = np.asarray(sim.const_reference[:, 0, 0:3])   # (T, 3) 只取窗口第 0 步
    elif "reference" in scenario:
        ref_pos = np.asarray(scenario["reference"][:, 0, 0:3])
    elif "waypoints" in scenario:
        # 若只有稀疏 waypoints，则插值成与仿真步数相同的轨迹
        waypoints = np.asarray(scenario["waypoints"])
        seg_times = np.linspace(0, sim_steps * config.MPC.dt, waypoints.shape[0])
        ref_pos = np.empty_like(actual_pos)
        for k in range(3):      # x y z
            ref_pos[:, k] = np.interp(time_vec, seg_times, waypoints[:, k])
    else:
        raise ValueError("无法在 scenario / sim 中找到参考轨迹！")

    # ---------- 3. 误差统计 ----------
    errors = np.linalg.norm(actual_pos - ref_pos, axis=1)      # (T,)

    print(f"\nTracking Performance:")
    print(f"  Final error   : {errors[-1]:.4f} m")
    print(f"  Average error : {errors.mean():.4f} m")
    print(f"  Maximum error : {errors.max():.4f} m")
    print(f"  Std-dev error : {errors.std():.4f} m")

    # ---------- 4. 绘图 ----------
    fig = plt.figure(figsize=(30, 10))

    # ---- 4-1 3-D 轨迹 ----
    ax3d = fig.add_subplot(141, projection='3d')
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
              'r--',  linewidth=2, label='Reference')
    ax3d.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
              'b-',  linewidth=1.2, label='Actual')
    ax3d.set_title('3-D Trajectory')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend()
    ax3d.grid(True)

    # ---- 4-2 XY 平面 ----
    ax_xy = fig.add_subplot(142)
    ax_xy.plot(ref_pos[:, 0], ref_pos[:, 1], 'r--', label='Reference')
    ax_xy.plot(actual_pos[:, 0], actual_pos[:, 1], 'b-',  label='Actual')
    ax_xy.set_title('Trajectory in XY Plane')
    ax_xy.set_xlabel('X [m]')
    ax_xy.set_ylabel('Y [m]')
    ax_xy.axis('equal')
    ax_xy.legend()
    ax_xy.grid(True)

    # ---- 4-3 误差随时间 ----
    ax_err = fig.add_subplot(143)
    ax_err.plot(time_vec, errors, 'k-', label='‖pos_err‖')
    ax_err.axhline(0.05, color='g', ls='--', lw=0.8, label='5 cm')
    ax_err.axhline(0.10, color='r', ls='--', lw=0.8, label='10 cm')
    ax_err.set_title('Tracking Error vs Time')
    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('Error [m]')
    ax_err.legend()
    ax_err.grid(True)

    # ---- 4-4 位置分量 ----
    ax_pos = fig.add_subplot(144)
    ax_pos.plot(time_vec, actual_pos[:, 0], 'b',  label='X actual')
    ax_pos.plot(time_vec, actual_pos[:, 1], 'g',  label='Y actual')
    ax_pos.plot(time_vec, actual_pos[:, 2], 'r',  label='Z actual')
    ax_pos.plot(time_vec, ref_pos[:, 0], 'b--',   label='X ref')
    ax_pos.plot(time_vec, ref_pos[:, 1], 'g--',   label='Y ref')
    ax_pos.plot(time_vec, ref_pos[:, 2], 'r--',   label='Z ref')
    ax_pos.set_title('Position Components')
    ax_pos.set_xlabel('Time [s]')
    ax_pos.set_ylabel('Pos [m]')
    ax_pos.legend(ncol=2)
    ax_pos.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------- 5. 结果 dict ----------
    return {
        'final_error': float(errors[-1]),
        'avg_error'  : float(errors.mean()),
        'max_error'  : float(errors.max()),
        'std_error'  : float(errors.std())
    }


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

def generate_trajectory_reference(
    waypoints,       # list/ndarray, shape (N,3)
    num_iters,       # int: 仿真步数
    horizon,         # int: MPC 预测窗
    dt,              # float: 控制周期
    quat=jnp.array([1., 0., 0., 0.], dtype=jnp.float32)  # 姿态参考
):
    """
    Return shape: (num_iters, horizon+1, 17)
    """
    # ---------- 1) 预处理 ----------
    wp = jnp.asarray(waypoints, dtype=jnp.float32)        # (N,3)
    n_seg      = wp.shape[0] - 1
    total_time = num_iters * dt                           # 总时长
    seg_time   = total_time / n_seg                       # 每段时长

    total_steps = num_iters + horizon                     # 需要的最长 index
    t_all       = jnp.arange(total_steps, dtype=jnp.float32) * dt

    # ---------- 2) 逐时刻计算路径点 ----------
    seg_idx  = jnp.clip((t_all / seg_time).astype(jnp.int32), 0, n_seg - 1)
    alpha    = (t_all - seg_idx * seg_time) / seg_time    # 归一化 0~1
    start_wp = wp[seg_idx]                                # (total_steps,3)
    end_wp   = wp[seg_idx + 1]
    pos_all  = start_wp + alpha[:, None] * (end_wp - start_wp)  # (total_steps,3)

    # ---------- 3) 滑动窗口构造 (num_iters, horizon+1, 3) ----------
    def slice_window(i):
        return jax.lax.dynamic_slice(pos_all,  # 起点
                                     (i, 0),   # offset
                                     (horizon + 1, 3))  # size
    ref_pos = jax.vmap(slice_window)(jnp.arange(num_iters))  # (num_iters, h+1, 3)

    # ---------- 4) 拼 17 维完整状态 ----------
    zeros_2 = jnp.zeros((num_iters, horizon + 1, 2),  dtype=jnp.float32)   # 关节角
    zeros_6 = jnp.zeros((num_iters, horizon + 1, 6),  dtype=jnp.float32)   # 速度
    full_ref = jnp.concatenate(
        [
            ref_pos,                                                             # 0:3 位置
            jnp.broadcast_to(quat, (num_iters, horizon + 1, 4)),                 # 3:7 姿态
            zeros_2,                                                             # 7:9 关节角
            zeros_6,                                                             # 9:15 线/角速度
            zeros_2,                                                             # 15:17 关节角速度
        ],
        axis=-1,
    )
    return full_ref

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


def analyze_arm_control_results(sim, scenario, config):
    """
    飞机悬停 + 机械臂控制实验的结果分析：
    1) 末端执行器位置误差
    2) 关节角度跟踪误差
    """

    # ========== 0. 工具 ==========
    def fk_end_effector(state):
        pos  = state[0:3]
        quat = state[3:7] / (np.linalg.norm(state[3:7]) + 1e-10)
        q1, q2 = state[7], state[8]
        return np.asarray(compute_end_effector_position(pos, quat, q1, q2))

    # ========== 1. 真实轨迹 ==========
    states_np = np.asarray(sim.state_traj)[:-1]        # (T,17) 与参考对齐
    sim_steps = states_np.shape[0]
    time_vec  = np.arange(sim_steps) * config.MPC.dt   # (T,)

    ee_actual = np.vstack([fk_end_effector(s) for s in states_np])      # (T,3)
    drone_actual = states_np[:, 0:3]
    q_actual  = states_np[:, 7:9]                                        # (T,2)

    # ========== 2. 参考轨迹 ==========
    #   a) 优先 sim.const_reference
    #   b) 其次 scenario["reference"] (自生成的 arm_swing_reference)
    #   c) 最后 scenario["ee_waypoints"]（单点）或固定 0
    if hasattr(sim, "const_reference"):
        ref_full = np.asarray(sim.const_reference)[:, 0, :]  # (T,17)
    elif "reference" in scenario:
        ref_full = np.asarray(scenario["reference"])[:, 0, :]
    else:
        # 只给一个静态 ee_waypoints[0]
        tgt = np.asarray(scenario.get("ee_waypoints", [[0., 0., 1.5]]))[0]
        ref_full = np.tile(
            np.concatenate([tgt, [1,0,0,0], np.zeros(10)]), (sim_steps,1)
        )

    ee_ref = ref_full[:sim_steps, 0:3]   # (T,3)
    q_ref  = ref_full[:sim_steps, 7:9]   # (T,2)

    # ========== 3. 误差 ==========
    ee_err = ee_actual - ee_ref          # (T,3)
    ee_norm = np.linalg.norm(ee_err, axis=1)
    drone_norm = np.linalg.norm(drone_actual - ee_ref, axis=1)
    q_err   = q_actual - q_ref
    q_norm  = np.linalg.norm(q_err, axis=1)

    # ----------- 打印统计 -----------
    print("\n" + "="*70)
    print("Arm-Control Tracking Results")
    print("="*70)
    print(f"末端位置误差 (m)")
    print(f"  Final : {ee_norm[-1]:.4f}")
    print(f"  Avg   : {ee_norm.mean():.4f}")
    print(f"  Max   : {ee_norm.max():.4f}")
    print(f"  Std   : {ee_norm.std():.4f}")
    print(f"关节角误差 (rad)")
    print(f"  Final : {q_norm[-1]:.4f}")
    print(f"  Avg   : {q_norm.mean():.4f}")

    # ========== 4. 绘图 ==========
    fig = plt.figure(figsize=(14, 10))

    # 4-1 末端 3-D 轨迹
    ax3d = fig.add_subplot(221, projection='3d')
    # ax3d.plot(ee_ref[:,0], ee_ref[:,1], ee_ref[:,2], 'r--', lw=2, label='EE Ref')
    ax3d.plot(ee_actual[:,0], ee_actual[:,1], ee_actual[:,2], 'b-', lw=1.2, label='EE Actual')
    ax3d.set_title('End-Effector 3-D Trajectory')
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.legend(); ax3d.grid(True)

    # 4-2 XY 平面
    ax_xy = fig.add_subplot(222)
    ax_xy.plot(ee_ref[:,0], ee_ref[:,1], 'r--', label='Ref')
    ax_xy.plot(ee_actual[:,0], ee_actual[:,1], 'b-', label='Actual')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('EE Trajectory in XY')
    ax_xy.axis('equal'); ax_xy.grid(True); ax_xy.legend()

    # 4-3 末端误差随时间
    ax_err = fig.add_subplot(223)
    ax_err.plot(time_vec, drone_norm, 'k', label='‖ee_err‖')
    ax_err.axhline(0.05, color='g', ls='--', lw=0.8, label='5 cm')
    ax_err.set_xlabel('Time [s]'); ax_err.set_ylabel('Error [m]')
    ax_err.set_title('End-Effector Error')
    ax_err.grid(True); ax_err.legend()

    # 4-4 关节角度
    ax_q = fig.add_subplot(224)
    ax_q.plot(time_vec, q_actual[:,0], 'b',  label='q1 actual')
    ax_q.plot(time_vec, q_actual[:,1], 'g',  label='q2 actual')
    ax_q.plot(time_vec, q_ref[:,0],  'b--', label='q1 ref')
    ax_q.plot(time_vec, q_ref[:,1],  'g--', label='q2 ref')
    ax_q.set_xlabel('Time [s]'); ax_q.set_ylabel('Angle [rad]')
    ax_q.set_title('Joint Angles')
    ax_q.grid(True); ax_q.legend(ncol=2)

    plt.tight_layout(); plt.show()

    # ========== 5. 返回指标 ==========
    return {
        'ee_final' : float(ee_norm[-1]),
        'ee_avg'   : float(ee_norm.mean()),
        'ee_max'   : float(ee_norm.max()),
        'ee_std'   : float(ee_norm.std()),
        'q_final'  : float(q_norm[-1]),
        'q_avg'    : float(q_norm.mean())
    }

# ------------------------------------------------------------------
# 机械臂周期摆动参考：飞机坐标系不动，关节 q1,q2 = A·sin(ωt+φ)
# ------------------------------------------------------------------
def generate_arm_swing_reference(
    num_iters: int,
    horizon: int,
    dt: float,
    amp_deg: Tuple[float, float] = (20.0, 15.0),   # 摆幅（度）
    freq_hz: Tuple[float, float] = (0.5, 0.8),      # 频率（Hz）
    base_height: float = 1.5,                       # 悬停高度
) -> jnp.ndarray:
    """
    返回 shape = (num_iters, horizon+1, 17)
    只改 joint→ 7:9 ；位置固定 [0,0,base_height] & 姿态恒水平
    """
    A_rad  = jnp.radians(jnp.asarray(amp_deg,  dtype=jnp.float32))  # (2,)
    w_rad  = 2 * jnp.pi * jnp.asarray(freq_hz, dtype=jnp.float32)   # (2,)

    # 全局时间轴 t 全长 (T+horizon)
    total_len = num_iters + horizon
    t_all = jnp.arange(total_len, dtype=jnp.float32) * dt           # (T+H,)

    # 每时刻关节角   q(t) = A*sin(ω t)
    q_all = A_rad[None, :] * jnp.sin(t_all[:, None] * w_rad[None, :])   # (T+H, 2)

    # 把它滑窗拼成 (T, H+1, 2)
    def slice_window(i):
        return jax.lax.dynamic_slice(q_all, (i, 0), (horizon+1, 2))  # (H+1,2)
    ref_q = jax.vmap(slice_window)(jnp.arange(num_iters))            # (T,H+1,2)

    # ------- 拼 17 维完整参考 -------
    ref_state = jnp.zeros((num_iters, horizon+1, 17), dtype=jnp.float32)
    ref_state = ref_state.at[:, :, 0:3].set(jnp.array([0., 0., base_height]))   # 悬停
    ref_state = ref_state.at[:, :, 3:7].set(jnp.array([1., 0., 0., 0.]))        # 水平
    ref_state = ref_state.at[:, :, 7:9].set(ref_q)                              # 摆臂
    return ref_state

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
                               'ee_trajectory', 'all', 'hover_n_arm'],
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
            scenario = TestScenario.reach_point_test([1.0, 0.0, 2.0])
        elif args.test == 'arm':
            scenario = TestScenario.arm_control_test([0.0, 0.0, 1.5], [0.5, 0.7])
        elif args.test == 'ee_trajectory':
            # 定义末端执行器目标轨迹
            ee_target = [1, 1, 1]  # 单个目标点
            scenario = TestScenario.end_effector_trajectory_test(ee_target, duration=1500.0)
        elif args.test == 'trajectory':
            waypoints = [[0, 0, 1.5], [0, 0, 1.5], [1, 0, 1.5], [1, 1, 2.0], [0, 1, 2.0], [0, 0, 1.5], [0, 0, 1.5]]
            scenario = TestScenario.trajectory_test(waypoints, duration=10.0)
        elif args.test == 'hover_n_arm':
            scenario = TestScenario.arm_swing_test(duration=10.0,
                                           amp_deg=(30,30),
                                           freq_hz=(0.3,0.3))
        
        sim, results = run_test_with_diagnostics(scenario, args.step, args.visualize)
        
        if args.plot:
            plot_results(sim, scenario, None)
        


        # python .\examples\quadrotor_arm_test.py --test ee_trajectory --step 3 --visualize
        #192.168.0.172 Iot 5605
        #192.168.0.235 Hm-PC 5605
        #raspi_ip =  '192.168.0.206' 5606
        #env -u LD_LIBRARY_PATH python examples/quadrotor_arm_test.py --test ee_trajectory --step 3 --visualize
        # ros2 run phasespace_client psnode