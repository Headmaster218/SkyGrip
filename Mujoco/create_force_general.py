import numpy as np
import mujoco
import mujoco.viewer
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class MPPIParams:
    """MPPI算法参数"""
    horizon: int = 30          # 预测时域步数
    num_samples: int = 64      # 采样轨迹数
    lambda_: float = 1.0       # 温度参数
    noise_sigma: np.ndarray = None  # 控制噪声标准差
    dt: float = 0.02           # 控制时间步长
    
    # 代价函数权重
    w_pos: float = 30.0        # 位置权重
    w_vel: float = 10.0        # 速度权重
    w_att: float = 50.0        # 姿态权重
    w_omega: float = 5.0       # 角速度权重
    w_ee: float = 5.0          # 末端位置权重
    w_ctrl: float = 0.1        # 控制量权重
    w_smooth: float = 1.0      # 平滑性权重
    
    def __post_init__(self):
        if self.noise_sigma is None:
            # 控制噪声：[总推力, 3个扭矩, 2关节力矩]
            self.noise_sigma = np.array([
                1.0,                  # 总推力噪声 [N] - 降低避免剧烈变化
                0.02, 0.02, 0.01,    # 扭矩噪声 [Nm] - 降低以提高稳定性
                0.1, 0.1             # 关节力矩噪声 [Nm]
            ])


class DirectControlMPPIController:
    """直接力/力矩MPPI控制器"""
    
    def __init__(self, model_path: str, params: MPPIParams = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.params = params or MPPIParams()
        
        # 获取body ID
        self.base_id = self.model.body('base_link').id
        self.link1_id = self.model.body('Link_1').id
        self.link2_id = self.model.body('Link_2').id
        
        # 控制维度：1总推力 + 3扭矩 + 2机械臂关节
        self.nu = 6
        
        # 系统质量（base + Link_1 + Link_2）
        self.m_base = float(self.model.body('base_link').mass)
        self.m_link1 = float(self.model.body('Link_1').mass)
        self.m_link2 = float(self.model.body('Link_2').mass)
        self.total_mass = self.m_base + self.m_link1 + self.m_link2
        
        # 悬停推力（理论值，实际需要根据重心偏移调整）
        self.nominal_hover_thrust = self.total_mass * 9.81
        
        # 惯性矩（从模型中读取）
        self.I_base = self.model.body('base_link').inertia
        self.I_link1 = self.model.body('Link_1').inertia
        self.I_link2 = self.model.body('Link_2').inertia
        
        # MPPI缓存
        self.u_init = np.zeros((self.params.horizon, self.nu))
        self.u_prev = np.zeros(self.nu)
        self.noise = np.zeros((self.params.num_samples, self.params.horizon, self.nu))
        self.costs = np.zeros(self.params.num_samples)
        
        # 初始化控制序列（悬停状态）
        self.u_init[:, 0] = self.nominal_hover_thrust
        
        # 参考状态和轨迹
        self.x_ref = np.zeros(20)  # 20维状态向量
        self.x_ref[2] = 1.5  # 初始目标高度
        
        # 轨迹跟踪参数
        self.waypoints = []  # 路径点列表
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.08  # 到达路径点的容差（米），增大到8cm
        self.trajectory_start_time = 0.0
        
        # 机械臂测试参数
        self.arm_test_enabled = False
        self.arm_test_start_time = 0.0
        self.arm_test_mode = "sine"  # "sine", "step", "circle", "random"
        self.arm_test_amplitude = 0.3  # 摆动幅度（弧度）
        self.arm_test_frequency = 0.5  # 摆动频率（Hz）
        self.arm_test_phase = 0.0  # 相位差
        self.arm_reference = np.array([0.0, 0.0])  # 机械臂参考位置

        # 机械臂电机参数
        self.e_break = [0.235, 0.372]  # 电子制动系数
        self.stall_torque = [3.0, 1.5]  # 每个关节的额定力矩（Nm）
        self.full_speed = [77 * 2 * np.pi / 60 , 57 * 2 * np.pi / 60]  # 每个关节的最大速度（弧度/秒）
        
        # 控制限制
        self.thrust_limits = [0, 30]  # 总推力限制 [N]
        self.torque_limits = [-2, 2]  # 扭矩限制 [Nm]
        
        print(f"Direct Control MPPI Controller initialized:")
        print(f"  Total mass: {self.total_mass:.3f} kg")
        print(f"  Nominal hover thrust: {self.nominal_hover_thrust:.2f} N")
        print(f"  Control horizon: {self.params.horizon} steps")
        print(f"  Sample count: {self.params.num_samples}")
    
    def set_trajectory(self, waypoints: List[np.ndarray], start_delay: float = 2.0):
        """设置要跟踪的轨迹路径点
        
        Args:
            waypoints: 路径点列表，每个点是[x, y, z]坐标
            start_delay: 开始移动前的延迟时间（秒）
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.trajectory_start_time = start_delay
        
        # 更新可视化标记
        self.update_waypoint_markers()
        
        print(f"Trajectory set with {len(waypoints)} waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"  Waypoint {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]")
    
    def update_waypoint_markers(self):
        """更新路径点的可视化标记"""
        # 更新所有路径点标记
        for i in range(10):  # 最多支持10个路径点
            if i < len(self.waypoints):
                # 显示路径点
                try:
                    waypoint_id = self.model.body(f'waypoint_{i}').id
                    self.data.mocap_pos[waypoint_id] = self.waypoints[i]
                except:
                    pass  # 如果body不存在则跳过
            else:
                # 隐藏未使用的路径点标记
                try:
                    waypoint_id = self.model.body(f'waypoint_{i}').id
                    self.data.mocap_pos[waypoint_id] = [0, 0, -10]  # 移到视野外
                except:
                    pass
        
        # 初始化已到达标记（全部隐藏）
        for i in range(10):
            try:
                reached_id = self.model.body(f'reached_{i}').id
                self.data.mocap_pos[reached_id] = [0, 0, -10]
            except:
                pass
        
        # 更新当前目标标记
        if self.current_waypoint_idx < len(self.waypoints):
            try:
                target_id = self.model.body('current_target').id
                self.data.mocap_pos[target_id] = self.waypoints[self.current_waypoint_idx]
            except:
                pass
    
    def update_reference(self, current_time: float):
        """根据当前时间更新参考位置"""
        if current_time < self.trajectory_start_time:
            # 保持初始位置
            return
        
        if self.current_waypoint_idx >= len(self.waypoints):
            # 已完成所有路径点
            return
        
        # 更新目标为当前路径点
        self.x_ref[0:3] = self.waypoints[self.current_waypoint_idx]
    
    def set_arm_test(self, mode: str = "sine", amplitude: float = 0.3, 
                     frequency: float = 0.5, start_delay: float = 0.0,
                     phase_shift: float = 0.0):
        """设置机械臂摆动测试
        
        Args:
            mode: 测试模式 ("sine", "step", "circle", "random", "joint1", "joint2")
            amplitude: 摆动幅度（弧度）
            frequency: 摆动频率（Hz）
            start_delay: 开始延迟（秒）
            phase_shift: 两个关节之间的相位差（弧度）
        """
        self.arm_test_enabled = True
        self.arm_test_mode = mode
        self.arm_test_amplitude = amplitude
        self.arm_test_frequency = frequency
        self.arm_test_start_time = start_delay
        self.arm_test_phase = phase_shift
        
        print(f"\n=== Arm Test Configuration ===")
        print(f"  Mode: {mode}")
        print(f"  Amplitude: {amplitude:.2f} rad ({np.rad2deg(amplitude):.1f}°)")
        print(f"  Frequency: {frequency:.2f} Hz")
        print(f"  Phase shift: {phase_shift:.2f} rad ({np.rad2deg(phase_shift):.1f}°)")
        print(f"  Start delay: {start_delay:.1f} seconds")
    
    def get_arm_reference(self, current_time: float) -> np.ndarray:
        """根据测试模式计算机械臂参考角度
        
        Returns:
            [joint1_ref, joint2_ref] 期望的关节角度
        """
        if not self.arm_test_enabled:
            return np.array([0.0, 0.0])
        
        if current_time < self.arm_test_start_time:
            return np.array([0.0, 0.0])
        
        # 计算测试时间
        test_time = current_time - self.arm_test_start_time
        omega = 2 * np.pi * self.arm_test_frequency
        
        # 添加平滑启动（前2秒内逐渐增加幅度）
        ramp_time = 2.0  # 平滑启动时间
        if test_time < ramp_time:
            amplitude_scale = test_time / ramp_time  # 从0逐渐增加到1
        else:
            amplitude_scale = 1.0
        
        actual_amplitude = self.arm_test_amplitude * amplitude_scale
        
        if self.arm_test_mode == "sine":
            # 正弦波摆动
            joint1 = actual_amplitude * np.sin(omega * test_time)
            joint2 = actual_amplitude * np.sin(omega * test_time + self.arm_test_phase)
            
        elif self.arm_test_mode == "step":
            # 方波摆动（添加平滑过渡）
            period = 1.0 / self.arm_test_frequency
            phase = (test_time % period) / period
            
            # 使用tanh平滑过渡，避免突变
            transition_speed = 10.0
            joint1 = actual_amplitude * np.tanh(transition_speed * (phase - 0.5))
            phase2 = (phase + self.arm_test_phase/(2*np.pi)) % 1.0
            joint2 = actual_amplitude * np.tanh(transition_speed * (phase2 - 0.5))
            
        elif self.arm_test_mode == "circle":
            # 圆形运动（两个关节配合）
            joint1 = actual_amplitude * np.sin(omega * test_time)
            joint2 = actual_amplitude * np.cos(omega * test_time)
            
        elif self.arm_test_mode == "joint1":
            # 只摆动关节1
            joint1 = actual_amplitude * np.sin(omega * test_time)
            joint2 = 0.0
            
        elif self.arm_test_mode == "joint2":
            # 只摆动关节2
            joint1 = 0.0
            joint2 = actual_amplitude * np.sin(omega * test_time)
            
        elif self.arm_test_mode == "random":
            # 随机扰动（低频，平滑变化）
            # 使用低通滤波的随机信号
            if not hasattr(self, 'random_targets'):
                self.random_targets = np.array([0.0, 0.0])
                self.random_filter_state = np.array([0.0, 0.0])
            
            # 每0.5秒更新一次目标
            if int(test_time * 2) != int((test_time - 0.05) * 2):
                self.random_targets = actual_amplitude * (np.random.rand(2) - 0.5) * 2
            
            # 低通滤波平滑过渡
            alpha = 0.1  # 滤波系数
            self.random_filter_state = alpha * self.random_targets + (1 - alpha) * self.random_filter_state
            joint1 = self.random_filter_state[0]
            joint2 = self.random_filter_state[1]
            
        else:
            joint1 = 0.0
            joint2 = 0.0
        
        # 限制在关节范围内
        joint1 = np.clip(joint1, -0.6, 0.6)  # 更保守的限制
        joint2 = np.clip(joint2, -0.6, 0.6)
        
        return np.array([joint1, joint2])
    
    def check_waypoint_reached(self, current_pos: np.ndarray):
        """检查是否到达当前路径点"""
        if self.current_waypoint_idx >= len(self.waypoints):
            return
        
        target = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(current_pos - target)
        
        if distance < self.waypoint_tolerance:
            print(f"  -> Reached waypoint {self.current_waypoint_idx}: "
                  f"[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
            
            # 标记已到达的路径点（变成绿色小球）
            try:
                reached_id = self.model.body(f'reached_{self.current_waypoint_idx}').id
                self.data.mocap_pos[reached_id] = self.waypoints[self.current_waypoint_idx].copy()
            except:
                pass
            
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.waypoints):
                self.x_ref[0:3] = self.waypoints[self.current_waypoint_idx]
                
                # 更新当前目标标记
                try:
                    target_id = self.model.body('current_target').id
                    self.data.mocap_pos[target_id] = self.waypoints[self.current_waypoint_idx]
                except:
                    pass
                
                print(f"  -> Moving to waypoint {self.current_waypoint_idx}: "
                      f"[{self.x_ref[0]:.2f}, {self.x_ref[1]:.2f}, {self.x_ref[2]:.2f}]")
            else:
                print("  -> All waypoints reached! Hovering at final position.")
                # 隐藏当前目标标记
                try:
                    target_id = self.model.body('current_target').id
                    self.data.mocap_pos[target_id] = [0, 0, -10]
                except:
                    pass
    
    def get_state(self) -> np.ndarray:
        """获取当前状态向量[20维]"""
        # 位置和速度
        pos = self.data.qpos[0:3].copy()
        vel = self.data.qvel[0:3].copy()
        
        # 四元数和角速度
        quat = self.data.qpos[3:7].copy()
        omega = self.data.qvel[3:6].copy()
        
        # 机械臂状态
        q1 = self.data.qpos[7]
        q2 = self.data.qpos[8]
        dq1 = self.data.qvel[6]
        dq2 = self.data.qvel[7]
        
        # 欧拉角（用于控制）
        R = self.data.xmat[self.base_id].reshape(3, 3)
        euler = self.rotation_to_euler(R)
        
        return np.concatenate([pos, vel, quat, omega, euler, [q1, q2, dq1, dq2]])
    
    def compute_com_offset(self, q1: float, q2: float) -> np.ndarray:
        """计算由于机械臂运动造成的重心偏移（相对于base_link）"""
        # Link_1的质心位置（在其自身坐标系中）
        com_link1_local = np.array([-0.0016184, -7.0854E-06, -0.08892])
        # Link_2的质心位置（在其自身坐标系中）
        com_link2_local = np.array([-0.00195, 0, 0.079412])
        
        # 计算Link_1在base坐标系中的质心位置
        # Link_1绕X轴旋转q1
        R1 = np.array([[1, 0, 0],
                       [0, np.cos(q1), -np.sin(q1)],
                       [0, np.sin(q1), np.cos(q1)]])
        
        link1_offset = np.array([0, -2.5e-05, -0.038])
        com_link1_base = link1_offset + R1 @ com_link1_local
        
        # 计算Link_2在base坐标系中的质心位置
        # Link_2相对于Link_1的位置和旋转
        link2_offset_link1 = np.array([0, 0, -0.1308])
        link2_pos_base = link1_offset + R1 @ link2_offset_link1
        
        # Link_2的旋转（先绕Y轴转90度，再绕X轴转q2）
        R2_rel = np.array([[1, 0, 0],
                          [0, np.cos(q2), -np.sin(q2)],
                          [0, np.sin(q2), np.cos(q2)]])
        R2_90 = np.array([[0, 0, -1],
                         [0, 1, 0],
                         [1, 0, 0]])
        R2 = R1 @ R2_90 @ R2_rel
        com_link2_base = link2_pos_base + R2 @ com_link2_local
        
        # 计算总重心偏移
        total_com = (self.m_base * np.array([1.928E-06, 0.0086666, 0.027403]) +
                    self.m_link1 * com_link1_base +
                    self.m_link2 * com_link2_base) / self.total_mass
        
        return total_com
    
    def rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转欧拉角（ZYX顺序）"""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return np.array([x, y, z])
    
    def apply_direct_control(self, thrust: float, torques: np.ndarray):
        """直接应用推力和扭矩到base_link质心"""
        # 清空之前的力
        self.data.xfrc_applied[self.base_id] = 0
        
        # 限制输入
        thrust = np.clip(thrust, self.thrust_limits[0], self.thrust_limits[1])
        torques = np.clip(torques, self.torque_limits[0], self.torque_limits[1])
        
        # 获取机体姿态
        R_body = self.data.xmat[self.base_id].reshape(3, 3)
        
        # 推力方向（机体坐标系Z轴）
        thrust_body = np.array([0, 0, thrust])
        thrust_world = R_body @ thrust_body
        
        # 应用力和力矩
        self.data.xfrc_applied[self.base_id, 0:3] = thrust_world
        self.data.xfrc_applied[self.base_id, 3:6] = torques
        
        return thrust_world, torques
    
    def dynamics_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """简化的快速动力学模型用于MPPI预测"""
        next_state = state.copy()
        
        # 提取状态（避免复制）
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        q_arm = state[16:18]
        dq_arm = state[18:20]
        
        # 提取并限制控制输入
        thrust = np.clip(control[0], self.nominal_hover_thrust * 0.5, self.nominal_hover_thrust * 2.0)
        torques = np.clip(control[1:4], -1.0, 1.0)
        tau_arm = control[4:6]  # 机械臂暂不限制
        
        # 简化的旋转矩阵计算（只计算需要的元素）
        w, x, y, z = quat
        # 只计算R的第三列（Z轴）
        R_z = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x**2 + y**2)
        ])
        
        # 推力向量（世界坐标系）
        thrust_world = thrust * R_z
        
        # 加速度（简化计算）
        acc = thrust_world / self.total_mass - np.array([0, 0, 9.81])
        acc -= 0.1 * vel  # 简单阻尼
        
        # 角加速度（简化）
        alpha = torques / self.I_base - 0.5 * omega  # 简单阻尼
        
        # 机械臂动力学（忽略以加快计算）
        ddq_arm = tau_arm * 3.0 - dq_arm * 3.0
        
        # 简单欧拉积分
        next_state[0:3] = pos + vel * dt
        next_state[3:6] = vel + acc * dt
        next_state[10:13] = omega + alpha * dt
        next_state[16:18] = np.clip(q_arm + dq_arm * dt, -0.8, 0.8)
        next_state[18:20] = dq_arm + ddq_arm * dt
        
        # 简化的四元数更新（小角度近似）
        dq = 0.5 * np.array([
            -omega[0]*x - omega[1]*y - omega[2]*z,
            omega[0]*w + omega[2]*y - omega[1]*z,
            omega[1]*w - omega[2]*x + omega[0]*z,
            omega[2]*w + omega[1]*x - omega[0]*y
        ])
        next_quat = quat + dq * dt
        next_quat /= np.linalg.norm(next_quat)
        next_state[6:10] = next_quat
        
        # 更新欧拉角（简化）
        next_state[13] = np.arcsin(2*(next_quat[0]*next_quat[2] - next_quat[1]*next_quat[3]))  # roll
        next_state[14] = np.arcsin(-2*(next_quat[0]*next_quat[1] + next_quat[2]*next_quat[3])) # pitch
        next_state[15] = np.arctan2(2*(next_quat[0]*next_quat[3] + next_quat[1]*next_quat[2]),
                                    1 - 2*(next_quat[2]**2 + next_quat[3]**2))  # yaw
        
        return next_state
    
    def quat_to_rotation(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def compute_cost(self, trajectory: np.ndarray, controls: np.ndarray, 
                    target_ee: np.ndarray = None) -> float:
        """计算轨迹代价（优化版）"""
        cost = 0.0
        p = self.params
        
        for t in range(len(trajectory)):
            x = trajectory[t]
            u = controls[t] if t < len(controls) else np.zeros(self.nu)
            
            # 位置误差（分别惩罚各轴）
            pos_err = x[0:3] - self.x_ref[0:3]
            # 增加对水平位置误差的惩罚
            cost += p.w_pos * (2.0*pos_err[0]**2 + 2.0*pos_err[1]**2 + 3.0*pos_err[2]**2)
            
            # 速度惩罚（限制过快运动）
            vel = x[3:6]
            # 当接近目标时，更强地惩罚速度
            if np.linalg.norm(pos_err) < 0.1:
                vel_weight = 2.0
            else:
                vel_weight = 1.0
            cost += p.w_vel * vel_weight * (vel[0]**2 + vel[1]**2 + 2.0*vel[2]**2)
            
            # 姿态稳定（Roll和Pitch更重要）
            euler = x[13:16]
            cost += p.w_att * (3.0*euler[0]**2 + 3.0*euler[1]**2 + 0.3*euler[2]**2)
            
            # 角速度惩罚
            omega = x[10:13]
            cost += p.w_omega * (2.0*omega[0]**2 + 2.0*omega[1]**2 + omega[2]**2)
            
            # 控制代价（推力偏离悬停值的惩罚）
            thrust_err = (u[0] - self.nominal_hover_thrust) / self.nominal_hover_thrust
            # 减少对推力的惩罚，允许更大的推力变化
            cost += p.w_ctrl * (5.0*thrust_err**2 + 50.0*np.sum(u[1:4]**2) + np.sum(u[4:6]**2))
            
            # 平滑性（减少控制突变）
            if t > 0:
                du = u - controls[t-1]
                # 更强的平滑性要求
                cost += p.w_smooth * (0.05*du[0]**2 + 20.0*np.sum(du[1:4]**2) + np.sum(du[4:6]**2))
            
            # 额外惩罚：大姿态角时增加代价
            if abs(euler[0]) > 0.35 or abs(euler[1]) > 0.35:  # 约20度
                cost += 200.0  # 大惩罚避免翻转
            
            # 轨迹进度奖励（鼓励向目标移动）
            if t < len(trajectory) - 1:
                next_pos_err = trajectory[t+1][0:3] - self.x_ref[0:3]
                progress = np.linalg.norm(pos_err) - np.linalg.norm(next_pos_err)
                cost -= 10.0 * progress  # 奖励接近目标
        
        return cost
    
    def mppi_step(self, current_state: np.ndarray, 
                  target_ee: np.ndarray = None) -> np.ndarray:
        """执行一步MPPI优化（快速版）"""
        # 初始化控制序列（如果是第一次或偏差太大）
        if np.linalg.norm(current_state[0:3] - self.x_ref[0:3]) > 0.5:
            # 重新初始化以加快收敛
            self.u_init[:, 0] = self.nominal_hover_thrust * 1.1
            self.u_init[:, 1:4] = 0
        
        # 生成噪声扰动（使用更高效的方式）
        self.noise = np.random.randn(self.params.num_samples, self.params.horizon, self.nu)
        self.noise *= self.params.noise_sigma
        
        # 并行采样和评估
        for i in range(self.params.num_samples):
            # 扰动控制序列
            u_sample = self.u_init + self.noise[i]
            
            # 限制控制输入范围（更宽松的限制以允许快速响应）
            u_sample[:, 0] = np.clip(u_sample[:, 0], 
                                     self.nominal_hover_thrust * 0.6, 
                                     self.nominal_hover_thrust * 1.6)
            u_sample[:, 1:4] = np.clip(u_sample[:, 1:4], -0.5, 0.5)
            u_sample[:, 4:6] = np.clip(u_sample[:, 4:6], -2, 2)
            
            # 前向模拟（使用简化的动力学）
            trajectory = np.zeros((self.params.horizon + 1, 20))
            trajectory[0] = current_state
            
            for t in range(self.params.horizon):
                trajectory[t+1] = self.dynamics_step(trajectory[t], u_sample[t], self.params.dt)
            
            # 计算代价
            self.costs[i] = self.compute_cost(trajectory, u_sample, target_ee)
        
        # 处理无效代价
        valid_mask = np.isfinite(self.costs)
        if not np.any(valid_mask):
            # 所有样本都失败，返回安全控制
            return np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        # 只使用有效样本
        valid_costs = self.costs[valid_mask]
        valid_noise = self.noise[valid_mask]
        
        # 计算权重（softmax）
        costs_shifted = valid_costs - np.min(valid_costs)
        weights = np.exp(-costs_shifted / self.params.lambda_)
        weights /= np.sum(weights)
        
        # 加权平均更新控制序列
        weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * valid_noise, axis=0)
        self.u_init = self.u_init + weighted_noise
        
        # 提取第一个控制
        u_optimal = self.u_init[0].copy()
        
        # 应用安全限制和补偿
        # self.apply_safety_limits(u_optimal, current_state)
        
        # 平滑滤波（减少控制突变）
        alpha_filter = 0.6  # 降低滤波强度以提高响应
        if np.linalg.norm(self.u_prev) > 0:  # 不是第一次
            u_optimal = alpha_filter * u_optimal + (1 - alpha_filter) * self.u_prev
        
        # 滚动时域
        self.u_init[:-1] = self.u_init[1:]
        # 最后一步使用稳态控制
        self.u_init[-1] = np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        # 保存用于平滑性计算
        self.u_prev = u_optimal.copy()
        
        return u_optimal
    
    def apply_safety_limits(self, control: np.ndarray, state: np.ndarray):
        """应用安全限制和补偿（改进版）"""
        # 位置和速度
        pos_error = state[0:3] - self.x_ref[0:3]
        vel = state[3:6]
        euler = state[13:16]
        q_arm = state[16:18]
        dq_arm = state[18:20]  # 机械臂速度
        omega = state[10:13]  # 角速度
        
        # 基础推力保证 - 永远不低于70%悬停推力
        min_thrust = self.nominal_hover_thrust * 0.7
        max_thrust = self.nominal_hover_thrust * 1.6
        
        # 高度PD补偿（更强的响应）
        kp_z = 10.0  # 比例增益（从8.0增加到10.0）
        kd_z = 5.0   # 微分增益（从4.0增加到5.0）
        thrust_compensation = self.nominal_hover_thrust + kp_z * (-pos_error[2]) + kd_z * (-vel[2])
        
        # 水平位置补偿（增强以对抗机械臂扰动）
        kp_xy = 0.5  # 从0.3增加到0.5
        kd_xy = 0.3  # 从0.2增加到0.3
        
        # 通过倾斜来产生水平力
        desired_roll = kp_xy * pos_error[1] + kd_xy * vel[1]  # Y误差产生Roll
        desired_pitch = -kp_xy * pos_error[0] - kd_xy * vel[0]  # X误差产生Pitch
        
        # 限制期望倾角
        max_tilt = 0.25  # 从0.2增加到0.25（约14度）
        desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
        desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)
        
        # 姿态PD控制（增强）
        kp_att = 2.0  # 从1.5增加到2.0
        kd_att = 0.7  # 从0.5增加到0.7
        control[1] = kp_att * (desired_roll - euler[0]) - kd_att * omega[0]
        control[2] = kp_att * (desired_pitch - euler[1]) - kd_att * omega[1]
        
        # 混合原始控制和补偿
        alpha = 0.2  # 原始控制权重进一步降低（从0.3到0.2）
        control[0] = alpha * control[0] + (1 - alpha) * thrust_compensation
        
        # 计算重心偏移
        com_offset = self.compute_com_offset(q_arm[0], q_arm[1])
        
        # 静态重心偏移补偿（增强）
        com_compensation = 1.0 + 0.05 * np.linalg.norm(com_offset[:2])  # 从0.03增加到0.05
        control[0] *= com_compensation
        
        # 改进的动态补偿：机械臂运动产生的扰动
        if self.arm_test_enabled:
            # 机械臂速度产生的动态效应
            if np.linalg.norm(dq_arm) > 0.05:  # 降低阈值
                dynamic_compensation = 0.08 * np.linalg.norm(dq_arm)  # 从0.05增加到0.08
                control[0] *= (1.0 + dynamic_compensation)
            
            # 机械臂运动产生的反作用力矩补偿（增强）
            if np.linalg.norm(dq_arm) > 0.05:
                # 根据机械臂速度方向调整姿态补偿
                # Joint1主要影响Y方向，Joint2主要影响X方向
                arm_torque_compensation = 0.1 * dq_arm  # 从0.02增加到0.1
                
                # 更精确的耦合模型
                control[1] -= arm_torque_compensation[1] * 0.3  # Joint2速度影响Roll
                control[2] += arm_torque_compensation[0] * 0.3  # Joint1速度影响Pitch
            
            # 添加预测补偿（基于机械臂位置预测扰动）
            if abs(q_arm[0]) > 0.1 or abs(q_arm[1]) > 0.1:
                # 机械臂偏离中心位置时，预补偿姿态
                predictive_roll = -0.2 * q_arm[1]  # Joint2位置影响Roll
                predictive_pitch = 0.2 * q_arm[0]  # Joint1位置影响Pitch
                control[1] += predictive_roll
                control[2] += predictive_pitch
        
        # 大姿态角紧急处理
        max_angle = 0.35  # 约20度
        if abs(euler[0]) > max_angle or abs(euler[1]) > max_angle:
            # 紧急恢复模式
            control[0] = np.clip(control[0], self.nominal_hover_thrust, max_thrust)
            control[1] = np.clip(control[1] * 2.0, -1.0, 1.0)  # 加强Roll控制
            control[2] = np.clip(control[2] * 2.0, -1.0, 1.0)  # 加强Pitch控制
        
        # 严格的最终限制
        control[0] = np.clip(control[0], min_thrust, max_thrust)
        control[1:4] = np.clip(control[1:4], -1.0, 1.0)  # 扭矩限制略微增加
    
    def apply_control(self, control: np.ndarray, arm_reference: np.ndarray = None):
        """应用控制输入（修复版）
        
        Args:
            control: 控制向量 [推力, 3个扭矩, 2个关节力矩]
            arm_reference: 机械臂参考角度 [joint1_ref, joint2_ref]
        """
        # 应用推力和扭矩
        self.apply_direct_control(control[0], control[1:4])
        
        # 应用机械臂控制
        if arm_reference is not None:
            # PD控制器跟踪参考角度
            state = self.get_state()
            q_arm = state[16:18]  # 当前关节角度
            dq_arm = state[18:20]  # 当前关节速度
            
            # 检查状态是否有效
            if not np.all(np.isfinite(q_arm)) or not np.all(np.isfinite(dq_arm)):
                print(f"WARNING: Invalid arm state detected! q_arm={q_arm}, dq_arm={dq_arm}")
                self.data.ctrl[0] = 0.0
                self.data.ctrl[1] = 0.0
                return
            
            # 限制误差范围（抗饱和）
            error = arm_reference - q_arm
            max_error = 0.5  # 最大误差限制（约28度）
            error = np.clip(error, -max_error, max_error)
            
            # 自适应PD控制增益（根据误差大小调整）
            error_norm = np.linalg.norm(error)
            if error_norm > 0.3:  # 大误差时降低增益
                kp = 2.0
                kd = 0.5
            elif error_norm > 0.1:  # 中等误差
                kp = 4.0
                kd = 1.0
            else:  # 小误差时使用较高增益
                kp = 5.0
                kd = 1.5
            
            # PD控制
            tau = kp * error - kd * dq_arm
            
            # 严格的力矩限制
            tau = np.clip(tau, -1.5, 1.5)  # 降低到±1.5Nm
            
            # 额外的安全检查
            if np.any(np.abs(tau) > 10):
                print(f"WARNING: Excessive torque detected! tau={tau}")
                tau = np.clip(tau, -1.0, 1.0)
            
            self.data.ctrl[0] = tau[0]
            self.data.ctrl[1] = tau[1]
        else:
            # 使用MPPI计算的控制（也要限制）
            self.data.ctrl[0] = control[4] - \
                      (self.data.qvel[0] - (control[4] / self.stall_torque[0])* self.full_speed[0]) * self.e_break[0]
            self.data.ctrl[1] = control[5] - \
                      (self.data.qvel[1] - (control[5] / self.stall_torque[1])* self.full_speed[1]) * self.e_break[1]
    
    def run_simulation(self, duration: float = 30.0, visualize: bool = True):
        """运行仿真（修复版）"""
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
        
        control_freq = 20  # Hz
        control_dt = 1.0 / control_freq
        last_control_time = 0
        
        sim_start_time = self.data.time
        
        # 强制初始化机械臂位置为0
        self.data.qpos[7] = 0.0  # Joint_1
        self.data.qpos[8] = 0.0  # Joint_2
        self.data.qvel[6] = 0.0  # Joint_1 velocity
        self.data.qvel[7] = 0.0  # Joint_2 velocity
        
        # 执行前向动力学以更新所有状态
        mujoco.mj_forward(self.model, self.data)
        
        # 初始控制量
        current_control = np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        last_print_time = -0.5
        
        # 性能计时
        real_start_time = time.time()
        mppi_compute_time = 0
        mppi_count = 0
        
        # 性能统计
        position_errors = []
        arm_tracking_errors = []
        max_pos_error = 0.0
        max_arm_error = [0.0, 0.0]
        
        # 初始化路径点标记
        self.update_waypoint_markers()
        
        # 设置一个最小时间延迟，避免立即判断到达第一个路径点
        first_waypoint_check_time = 0.5
        
        # 机械臂测试状态
        arm_test_active = False
        arm_initialized = False  # 添加初始化标志
        
        try:
            while self.data.time - sim_start_time < duration:
                current_time = self.data.time
                
                # 更新参考位置（轨迹跟踪）
                self.update_reference(current_time)
                
                # 获取机械臂参考角度
                arm_reference = self.get_arm_reference(current_time)
                
                # 检查是否应该激活机械臂测试
                if self.arm_test_enabled and current_time >= self.arm_test_start_time and not arm_test_active:
                    # 在激活前，确保机械臂在初始位置
                    if not arm_initialized:
                        # 先将机械臂移动到初始位置（使用几个控制周期）
                        state = self.get_state()
                        q_arm = state[16:18]
                        if np.linalg.norm(q_arm) > 0.05:  # 如果不在初始位置
                            print(f"Initializing arm to zero position... Current: J1={np.rad2deg(q_arm[0]):.1f}°, J2={np.rad2deg(q_arm[1]):.1f}°")
                            # 应用归零控制
                            self.data.ctrl[0] = -2.0 * q_arm[0] - 0.5 * state[18]
                            self.data.ctrl[1] = -2.0 * q_arm[1] - 0.5 * state[19]
                            self.data.ctrl[0] = np.clip(self.data.ctrl[0], -1.0, 1.0)
                            self.data.ctrl[1] = np.clip(self.data.ctrl[1], -1.0, 1.0)
                        else:
                            arm_initialized = True
                    else:
                        arm_test_active = True
                        print(f"\n>>> ARM TEST STARTED at t={current_time:.1f}s <<<")
                        print(f"    Mode: {self.arm_test_mode}")
                        print(f"    Amplitude will ramp up over 2 seconds")
                        # 验证初始状态
                        state = self.get_state()
                        q_arm = state[16:18]
                        print(f"    Initial arm position: J1={np.rad2deg(q_arm[0]):.1f}°, J2={np.rad2deg(q_arm[1]):.1f}°")
                
                # MPPI控制更新
                if current_time - last_control_time >= control_dt:
                    mppi_start = time.time()
                    state = self.get_state()
                    
                    # 安全检查
                    if not np.all(np.isfinite(state)):
                        print(f"WARNING: Invalid state detected at t={current_time:.2f}s")
                        break
                    
                    # 检查是否到达路径点（增加时间延迟）
                    if current_time > first_waypoint_check_time:
                        self.check_waypoint_reached(state[0:3])
                    
                    current_control = self.mppi_step(state)
                    mppi_compute_time += time.time() - mppi_start
                    mppi_count += 1
                    last_control_time = current_time
                
                # 收集统计数据
                state = self.get_state()
                if np.all(np.isfinite(state)):  # 只在状态有效时收集
                    pos = state[0:3]
                    q_arm = state[16:18]
                    pos_error = np.linalg.norm(pos - self.x_ref[0:3])
                    position_errors.append(pos_error)
                    max_pos_error = max(max_pos_error, pos_error)
                    
                    if arm_test_active:
                        arm_error = np.abs(arm_reference - q_arm)
                        if np.all(np.isfinite(arm_error)):
                            arm_tracking_errors.append(arm_error)
                            max_arm_error[0] = max(max_arm_error[0], arm_error[0])
                            max_arm_error[1] = max(max_arm_error[1], arm_error[1])
                
                # 状态打印
                if current_time - last_print_time >= 0.5:  # 每0.5秒打印一次
                    if np.all(np.isfinite(state)):
                        euler = state[13:16]
                        vel = state[3:6]
                        dq_arm = state[18:20]
                        
                        real_time_factor = current_time / max(0.001, time.time() - real_start_time)
                        
                        # 显示当前目标
                        target_info = ""
                        if self.current_waypoint_idx < len(self.waypoints):
                            target = self.waypoints[self.current_waypoint_idx]
                            target_info = f" | WP[{self.current_waypoint_idx}]: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
                        else:
                            target_info = " | Hovering"
                        
                        # 机械臂信息
                        arm_info = ""
                        if arm_test_active:
                            arm_info = f" | ARM: J1={np.rad2deg(q_arm[0]):5.1f}° J2={np.rad2deg(q_arm[1]):5.1f}° | Ref: J1={np.rad2deg(arm_reference[0]):5.1f}° J2={np.rad2deg(arm_reference[1]):5.1f}°"
                        
                        print(f"t={current_time:5.1f}s | "
                              f"Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                              f"RPY: [{np.rad2deg(euler[0]):5.1f}°, {np.rad2deg(euler[1]):5.1f}°] | "
                              f"T: {current_control[0]:5.1f}N | "
                              f"Err: {pos_error:.3f}m"
                              f"{target_info}"
                              f"{arm_info}")
                        
                        last_print_time = current_time
                
                # 施加控制（包括机械臂控制）
                if arm_test_active:
                    self.apply_control(current_control, arm_reference)
                elif not arm_initialized and self.arm_test_enabled:
                    # 归零过程中只控制推力和姿态，机械臂由上面的归零逻辑控制
                    self.apply_direct_control(current_control[0], current_control[1:4])
                else:
                    self.apply_control(current_control)
                
                mujoco.mj_step(self.model, self.data)
                
                if visualize:
                    viewer.sync()
        
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        except Exception as e:
            print(f"\nSimulation stopped due to error: {e}")
        
        # 性能统计（只统计有效数据）
        print("\n" + "="*60)
        print("=== Performance Statistics ===")
        print("="*60)
        
        if mppi_count > 0:
            print(f"\nComputation Performance:")
            print(f"  Average MPPI compute time: {mppi_compute_time/mppi_count*1000:.1f} ms")
            print(f"  Real-time factor: {(self.data.time - sim_start_time)/(time.time()-real_start_time):.2f}x")
        
        if position_errors and len(position_errors) > 0:
            # 过滤掉无效值
            valid_errors = [e for e in position_errors if np.isfinite(e) and e < 100]
            if valid_errors:
                avg_pos_error = np.mean(valid_errors)
                print(f"\nPosition Control Performance:")
                print(f"  Average position error: {avg_pos_error:.4f} m")
                print(f"  Maximum position error: {min(max_pos_error, 100):.4f} m")
                if np.isfinite(position_errors[-1]):
                    print(f"  Final position error: {position_errors[-1]:.4f} m")
        
        if arm_tracking_errors and len(arm_tracking_errors) > 0:
            # 过滤掉无效值
            valid_arm_errors = [e for e in arm_tracking_errors if np.all(np.isfinite(e)) and np.all(e < 10)]
            if valid_arm_errors:
                avg_arm_error = np.mean(valid_arm_errors, axis=0)
                print(f"\nArm Tracking Performance:")
                print(f"  Average tracking error:")
                print(f"    Joint 1: {np.rad2deg(avg_arm_error[0]):.2f}°")
                print(f"    Joint 2: {np.rad2deg(avg_arm_error[1]):.2f}°")
                print(f"  Maximum tracking error:")
                print(f"    Joint 1: {np.rad2deg(min(max_arm_error[0], 10)):.2f}°")
                print(f"    Joint 2: {np.rad2deg(min(max_arm_error[1], 10)):.2f}°")
        
        # 轨迹完成统计
        if self.current_waypoint_idx >= len(self.waypoints):
            print(f"\nSuccessfully completed all {len(self.waypoints)} waypoints!")
        else:
            print(f"\nCompleted {self.current_waypoint_idx}/{len(self.waypoints)} waypoints")

def main():
    # 创建模型

    # 初始化MPPI控制器 - 大幅优化参数以提高速度
    params = MPPIParams(
        horizon=5,              # 减少预测步数
        num_samples=200,         # 大幅减少采样数！从2000减到200
        lambda_=2.0,             # 增加温度参数
        w_pos=80.0,              # 位置权重
        w_att=60.0,              # 姿态权重
        w_omega=15.0,            # 角速度惩罚
        w_vel=10.0,              # 速度权重
        w_ctrl=0.01,             # 控制权重
        w_smooth=1.0,            # 平滑性权重
        noise_sigma=np.array([0.8, 0.015, 0.015, 0.008, 0.05, 0.05])  # 调整噪声
    )
    
    controller = DirectControlMPPIController("drone_direct_control.xml", params)
    
    # ========== 选择测试模式 ==========
    # 模式1：完整轨迹跟踪 + 机械臂测试
    TEST_MODE = "hover_with_arm"  # 可选: "full_trajectory_with_arm", "hover_with_arm", "trajectory_only"
    
    if TEST_MODE == "full_trajectory_with_arm":
        # 设置轨迹
        waypoints = [
            np.array([0.0, 0.0, 1.5]),   # 初始悬停位置
            np.array([0.3, 0.0, 1.5]),   # 向前移动0.3m
            np.array([0.5, 0.2, 1.5]),   # 继续向前并向右
            np.array([0.5, 0.5, 1.5]),   # 向右移动到角落
            np.array([0.5, 0.5, 1.8]),   # 上升0.3m
            np.array([0.0, 0.5, 1.8]),   # 向后移动
            np.array([0.0, 0.0, 1.8]),   # 回到原点上方（悬停测试位置）
        ]
        controller.set_trajectory(waypoints, start_delay=2.0)
        
        # 设置机械臂测试（在完成轨迹后开始）
        controller.set_arm_test(
            mode="sine",           # 测试模式: "sine", "step", "circle", "joint1", "joint2", "random"
            amplitude=0.4,         # 摆动幅度（弧度，约23度）
            frequency=0.5,         # 摆动频率（Hz）
            start_delay=20.0,      # 20秒后开始（给足时间完成轨迹）
            phase_shift=np.pi/2    # 两个关节90度相位差
        )
        
        simulation_time = 45.0
        
    elif TEST_MODE == "hover_with_arm":
        # 只悬停 + 机械臂测试
        waypoints = [
            np.array([0.0, 0.0, 1.5]),   # 只在原点悬停
        ]
        controller.set_trajectory(waypoints, start_delay=0.0)
        
        # 设置机械臂测试（更保守的参数）
        controller.set_arm_test(
            mode="sine",           # 先用正弦波测试，更平滑
            amplitude=0.9,         # 降低幅度（约17度）
            frequency=0.1,         # 降低频率，更慢的运动
            start_delay=1.0,       # 3秒后开始
            phase_shift=np.pi/2    # 90度相位差
        )
        
        simulation_time = 30.0
        
    elif TEST_MODE == "trajectory_only":
        # 只做轨迹跟踪，不测试机械臂
        waypoints = [
            np.array([0.0, 0.0, 1.5]),
            np.array([0.3, 0.0, 1.5]),
            np.array([0.5, 0.2, 1.5]),
            np.array([0.5, 0.5, 1.5]),
            np.array([0.5, 0.5, 2.0]),
            np.array([0.0, 0.5, 2.0]),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 1.5]),
        ]
        controller.set_trajectory(waypoints, start_delay=2.0)
        simulation_time = 30.0
    
    # 设置初始参考位置
    controller.x_ref[0:3] = waypoints[0]
    
    # 运行仿真
    print("\n" + "="*60)
    print("=== Starting Simulation ===")
    print("="*60)
    print(f"\nTest Mode: {TEST_MODE}")
    print("\nVisualization Legend:")
    print("  - Large GREEN sphere: Current target waypoint")
    print("  - Small BLUE spheres: Future waypoints") 
    print("  - Small GREEN spheres: Reached waypoints")
    
    if controller.arm_test_enabled:
        print("\n=== Arm Stability Test Enabled ===")
        print("The arm will start moving after the drone reaches hovering position.")
        print("Watch how the drone maintains stability during arm movements!")
    
    print("\n" + "-"*60 + "\n")
    
    controller.run_simulation(duration=simulation_time, visualize=True)


if __name__ == "__main__":
    main()