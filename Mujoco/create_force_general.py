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

def create_model_with_direct_control(save_path: str = "drone_direct_control.xml"):
    """创建直接力/力矩控制的模型"""
    model_xml = """
    <mujoco model="drone_arm_system">
        <compiler angle="radian" meshdir="." />
        <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
        
        <asset>
            <mesh name="base_link" file="base_link.STL"/>
            <mesh name="Link_1" file="Link_1.STL"/>
            <mesh name="Link_2" file="Link_2.STL"/>
        </asset>
        
        <actuator>
            <!-- 机械臂关节控制 -->
            <motor joint="Joint_1" name="act1" gear="1" ctrlrange="-6 6"/>
            <motor joint="Joint_2" name="act2" gear="1" ctrlrange="-4 4"/>
        </actuator>
        
        <worldbody>
            <!-- base_link作为无人机本体 -->
            <body name="base_link" pos="0 0 1.5">
                <freejoint name="drone_free_joint"/>
                
                <!-- 惯性参数 -->
                <inertial mass="1.0" 
                         pos="1.928E-06 0.0086666 0.027403" 
                         diaginertia="0.00409 0.0055803 0.0094981"/>
                
                <!-- 机身几何体 -->
                <geom type="mesh" mesh="base_link" rgba="0.7 0.7 0.9 1"/>
                
                <!-- 可视化推力箭头位置 -->
                <site name="thrust_point" pos="0 0 0" size="0.02" rgba="0 1 1 0.8" type="sphere"/>
                
                <!-- 机械臂连接 -->
                <body name="Link_1" pos="0 -2.5e-05 -0.038">
                    <inertial pos="-0.0016184 -7.0854E-06 -0.08892" 
                             mass="0.08" 
                             diaginertia="3e-5 2.5e-5 1.2e-5"/>
                    <joint name="Joint_1" pos="0 0 0" axis="1 0 0" 
                          range="-0.8 0.8" damping="0.5" frictionloss="0.1"/>
                    <geom type="mesh" mesh="Link_1" rgba="1 0.95 0.9 1"/>
                    
                    <body name="Link_2" pos="0 0 -0.1308" quat="0 -1 0 0">
                        <inertial pos="-0.00195 0 0.079412" 
                                 mass="0.105" 
                                 diaginertia="2e-5 2e-5 2e-5"/>
                        <joint name="Joint_2" pos="0 0 0" axis="1 0 0" 
                              range="-0.8 0.8" damping="0.5" frictionloss="0.1"/>
                        <geom type="mesh" mesh="Link_2" rgba="1 1 1 1"/>
                        
                        <!-- 末端执行器标记 -->
                        <site name="end_effector" pos="0 0 0.16" size="0.02" rgba="1 0 0 1"/>
                    </body>
                </body>
            </body>
            
            <!-- 地面 -->
            <geom name="ground" type="plane" size="10 10 0.1" rgba="0.3 0.3 0.3 1"/>
            
            <!-- 当前目标位置标记（大绿球） -->
            <body name="current_target" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.05" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
            </body>
            
            <!-- 路径点标记（小球） -->
            <body name="waypoint_0" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_1" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_2" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_3" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_4" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_5" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_6" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_7" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_8" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            <body name="waypoint_9" pos="0 0 1.5" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.5 0.5 1 0.5" contype="0" conaffinity="0"/>
            </body>
            
            <!-- 已到达路径点标记（用于显示已经通过的点） -->
            <body name="reached_0" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_1" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_2" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_3" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_4" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_5" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_6" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_7" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_8" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
            <body name="reached_9" pos="0 0 -10" mocap="true">
                <geom type="sphere" size="0.015" rgba="0 0.8 0 0.7" contype="0" conaffinity="0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    with open(save_path, 'w') as f:
        f.write(model_xml)
    
    print(f"Model saved to {save_path}")
    return save_path


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
        self.m_base = 1.0
        self.m_link1 = 0.08
        self.m_link2 = 0.105
        self.total_mass = self.m_base + self.m_link1 + self.m_link2
        
        # 悬停推力（理论值，实际需要根据重心偏移调整）
        self.nominal_hover_thrust = self.total_mass * 9.81
        
        # 惯性矩（从XML读取）
        self.I_base = np.array([0.00409, 0.0055803, 0.0094981])
        self.I_link1 = np.array([3e-5, 2.5e-5, 1.2e-5])
        self.I_link2 = np.array([2e-5, 2e-5, 2e-5])
        
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
        self.apply_safety_limits(u_optimal, current_state)
        
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
        """应用安全限制和补偿（优化版）"""
        # 位置和速度
        pos_error = state[0:3] - self.x_ref[0:3]
        vel = state[3:6]
        euler = state[13:16]
        q_arm = state[16:18]
        omega = state[10:13]  # 角速度
        
        # 基础推力保证 - 永远不低于70%悬停推力
        min_thrust = self.nominal_hover_thrust * 0.7
        max_thrust = self.nominal_hover_thrust * 1.6
        
        # 高度PD补偿（更强的响应）
        kp_z = 8.0  # 比例增益
        kd_z = 4.0  # 微分增益
        thrust_compensation = self.nominal_hover_thrust + kp_z * (-pos_error[2]) + kd_z * (-vel[2])
        
        # 水平位置补偿（添加水平控制）
        kp_xy = 0.3
        kd_xy = 0.2
        # 通过倾斜来产生水平力
        desired_roll = kp_xy * pos_error[1] + kd_xy * vel[1]  # Y误差产生Roll
        desired_pitch = -kp_xy * pos_error[0] - kd_xy * vel[0]  # X误差产生Pitch
        
        # 限制期望倾角
        max_tilt = 0.2  # 约11.5度
        desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
        desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)
        
        # 姿态PD控制
        kp_att = 1.5
        kd_att = 0.5
        control[1] = kp_att * (desired_roll - euler[0]) - kd_att * omega[0]
        control[2] = kp_att * (desired_pitch - euler[1]) - kd_att * omega[1]
        
        # 混合原始控制和补偿（更多权重给补偿）
        alpha = 0.3  # 原始控制权重降低
        control[0] = alpha * control[0] + (1 - alpha) * thrust_compensation
        
        # 考虑重心偏移的额外补偿（温和）
        com_offset = self.compute_com_offset(q_arm[0], q_arm[1])
        com_compensation = 1.0 + 0.03 * np.linalg.norm(com_offset[:2])
        control[0] *= com_compensation
        
        # 大姿态角紧急处理
        max_angle = 0.35  # 约20度
        if abs(euler[0]) > max_angle or abs(euler[1]) > max_angle:
            # 紧急恢复模式
            control[0] = np.clip(control[0], self.nominal_hover_thrust, max_thrust)
            control[1] = np.clip(control[1] * 2.0, -1.0, 1.0)  # 加强Roll控制
            control[2] = np.clip(control[2] * 2.0, -1.0, 1.0)  # 加强Pitch控制
        
        # 严格的最终限制
        control[0] = np.clip(control[0], min_thrust, max_thrust)
        control[1:4] = np.clip(control[1:4], -0.8, 0.8)  # 扭矩限制
    
    def apply_control(self, control: np.ndarray):
        """应用控制输入"""
        # 应用推力和扭矩
        self.apply_direct_control(control[0], control[1:4])
        
        # 应用机械臂控制
        self.data.ctrl[0] = control[4]  # Joint_1
        self.data.ctrl[1] = control[5]  # Joint_2
    
    def run_simulation(self, duration: float = 30.0, visualize: bool = True):
        """运行仿真"""
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
        
        control_freq = 20  # Hz
        control_dt = 1.0 / control_freq
        last_control_time = 0
        
        sim_start_time = self.data.time
        
        # 初始控制量
        current_control = np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        last_print_time = -0.5
        
        # 性能计时
        real_start_time = time.time()
        mppi_compute_time = 0
        mppi_count = 0
        
        # 初始化路径点标记
        mujoco.mj_forward(self.model, self.data)  # 确保数据更新
        self.update_waypoint_markers()
        
        # 设置一个最小时间延迟，避免立即判断到达第一个路径点
        first_waypoint_check_time = 0.5
        
        try:
            while self.data.time - sim_start_time < duration:
                current_time = self.data.time
                
                # 更新参考位置（轨迹跟踪）
                self.update_reference(current_time)
                
                # MPPI控制更新
                if current_time - last_control_time >= control_dt:
                    mppi_start = time.time()
                    state = self.get_state()
                    
                    # 检查是否到达路径点（增加时间延迟）
                    if current_time > first_waypoint_check_time:
                        self.check_waypoint_reached(state[0:3])
                    
                    current_control = self.mppi_step(state)
                    mppi_compute_time += time.time() - mppi_start
                    mppi_count += 1
                    last_control_time = current_time
                
                # 状态打印
                if current_time - last_print_time >= 0.5:  # 每0.5秒打印一次
                    state = self.get_state()
                    pos = state[0:3]
                    euler = state[13:16]
                    vel = state[3:6]
                    pos_error = np.linalg.norm(pos - self.x_ref[0:3])
                    
                    real_time_factor = current_time / max(0.001, time.time() - real_start_time)
                    
                    # 显示当前目标
                    target_info = ""
                    if self.current_waypoint_idx < len(self.waypoints):
                        target = self.waypoints[self.current_waypoint_idx]
                        target_info = f" | WP[{self.current_waypoint_idx}]: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
                    else:
                        target_info = " | All waypoints reached!"
                    
                    print(f"t={current_time:5.1f}s | "
                          f"Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                          f"Vel: [{vel[0]:4.2f}, {vel[1]:4.2f}, {vel[2]:4.2f}] | "
                          f"RPY: [{np.rad2deg(euler[0]):5.1f}°, {np.rad2deg(euler[1]):5.1f}°, {np.rad2deg(euler[2]):5.1f}°] | "
                          f"T: {current_control[0]:5.1f}N | "
                          f"Err: {pos_error:.3f}m"
                          f"{target_info}")
                    
                    last_print_time = current_time
                
                # 施加控制
                self.apply_control(current_control)
                mujoco.mj_step(self.model, self.data)
                
                if visualize:
                    viewer.sync()
        
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        
        # 性能统计
        if mppi_count > 0:
            print(f"\n=== Performance Stats ===")
            print(f"Average MPPI compute time: {mppi_compute_time/mppi_count*1000:.1f} ms")
            print(f"Real-time factor: {(self.data.time - sim_start_time)/(time.time()-real_start_time):.2f}x")
            
            # 轨迹完成统计
            if self.current_waypoint_idx >= len(self.waypoints):
                print(f"Successfully completed all {len(self.waypoints)} waypoints!")
            else:
                print(f"Completed {self.current_waypoint_idx}/{len(self.waypoints)} waypoints")

def main():
    # 创建模型
    model_path = create_model_with_direct_control("drone_direct_control.xml")
    
    # 初始化MPPI控制器 - 大幅优化参数以提高速度
    params = MPPIParams(
        horizon=15,              # 减少预测步数
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
    
    controller = DirectControlMPPIController(model_path, params)
    
    # 设置轨迹（从原点移动到目标点）- 调整路径点让它们更合理
    waypoints = [
        np.array([0.0, 0.0, 1.5]),   # 初始悬停位置
        np.array([0.3, 0.0, 1.5]),   # 向前移动0.3m
        np.array([0.5, 0.2, 1.5]),   # 继续向前并向右
        np.array([0.5, 0.5, 1.5]),   # 向右移动到角落
        np.array([0.5, 0.5, 1.8]),   # 上升0.3m
        np.array([0.3, 0.5, 2.0]),   # 继续上升并向后
        np.array([0.0, 0.5, 2.0]),   # 向后移动到原点上方
        np.array([0.0, 0.3, 2.0]),   # 开始向左移动
        np.array([0.0, 0.0, 2.0]),   # 回到原点上方
        np.array([0.0, 0.0, 1.5]),   # 下降到初始高度
    ]
    
    # 设置轨迹，2秒后开始移动
    controller.set_trajectory(waypoints, start_delay=2.0)
    
    # 设置初始参考位置
    controller.x_ref[0:3] = waypoints[0]
    
    # 运行仿真
    print("\n=== Starting Simulation ===")
    print("Visualization Legend:")
    print("  - Large GREEN sphere: Current target waypoint")
    print("  - Small BLUE spheres: Future waypoints") 
    print("  - Small GREEN spheres: Reached waypoints")
    print("\nThe drone will hover for 2 seconds, then follow the trajectory.")
    print("Watch the drone navigate through all waypoints!\n")
    
    controller.run_simulation(duration=40.0, visualize=True)  # 增加时间到40秒


if __name__ == "__main__":
    main()