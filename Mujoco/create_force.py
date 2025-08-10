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
    w_vel: float = 10.0         # 速度权重
    w_att: float = 50.0        # 姿态权重
    w_omega: float = 5.0       # 角速度权重
    w_ee: float = 5.0         # 末端位置权重
    w_ctrl: float = 0.1        # 控制量权重
    w_smooth: float = 1.0      # 平滑性权重
    
    def __post_init__(self):
        if self.noise_sigma is None:
            # 控制噪声：[4个旋翼推力, 2关节力矩]
            self.noise_sigma = np.array([
                1.0, 1.0, 1.0, 1.0,   # 旋翼推力噪声 [N]
                0.3, 0.3              # 关节力矩噪声 [Nm]
            ])

                # <!-- 定义4个旋翼力作用点（Site） -->
                # <site name="rotor1" pos="0.105 0.0825 0.03" size="0.015" rgba="1 0 0 0.8" type="sphere"/>
                # <site name="rotor2" pos="-0.105 0.0825 0.03" size="0.015" rgba="0 1 0 0.8" type="sphere"/>
                # <site name="rotor3" pos="-0.105 -0.0825 0.03" size="0.015" rgba="0 0 1 0.8" type="sphere"/>
                # <site name="rotor4" pos="0.105 -0.0825 0.03" size="0.015" rgba="1 1 0 0.8" type="sphere"/>
                


def create_model_with_rotors(save_path: str = "drone_with_rotors.xml"):
    """创建带有旋翼作用点的模型"""
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
                
                <!-- 定义4个旋翼力作用点（Site） -->
                <site name="rotor1" pos="0.105 0.0825 0.03" size="0.015" rgba="1 0 0 0.8" type="sphere"/>
                <site name="rotor2" pos="-0.105 0.0825 0.03" size="0.015" rgba="0 1 0 0.8" type="sphere"/>
                <site name="rotor3" pos="-0.105 -0.0825 0.03" size="0.015" rgba="0 0 1 0.8" type="sphere"/>
                <site name="rotor4" pos="0.105 -0.0825 0.03" size="0.015" rgba="1 1 0 0.8" type="sphere"/>
                
                <!-- 可视化旋翼盘 -->
                <geom name="rotor1_visual" type="cylinder" pos="0.105 0.0825 0.03" 
                      size="0.05 0.002" rgba="0.3 0.3 0.3 0.6" contype="0" conaffinity="0"/>
                <geom name="rotor2_visual" type="cylinder" pos="-0.105 0.0825 0.03" 
                      size="0.05 0.002" rgba="0.3 0.3 0.3 0.6" contype="0" conaffinity="0"/>
                <geom name="rotor3_visual" type="cylinder" pos="-0.105 -0.0825 0.03" 
                      size="0.05 0.002" rgba="0.3 0.3 0.3 0.6" contype="0" conaffinity="0"/>
                <geom name="rotor4_visual" type="cylinder" pos="0.105 -0.0825 0.03" 
                      size="0.05 0.002" rgba="0.3 0.3 0.3 0.6" contype="0" conaffinity="0"/>
                
                <!-- 机械臂连接 -->
                <body name="Link_1" pos="0 -2.5e-05 -0.038">
                    <inertial pos="-0.0016184 -7.0854E-06 -0.08892" 
                             mass="0.08" 
                             diaginertia="3e-5 2.5e-5 1.2e-5"/>
                    <joint name="Joint_1" pos="0 0 0" axis="1 0 0" 
                          range="-0.8 0.8" damping="0.05" frictionloss="0.1"/>
                    <geom type="mesh" mesh="Link_1" rgba="1 0.95 0.9 1"/>
                    
                    <body name="Link_2" pos="0 0 -0.1308" quat="0 -1 0 0">
                        <inertial pos="-0.00195 0 0.079412" 
                                 mass="0.105" 
                                 diaginertia="2e-5 2e-5 2e-5"/>
                        <joint name="Joint_2" pos="0 0 0" axis="1 0 0" 
                              range="-0.8 0.8" damping="0.05" frictionloss="0.1"/>
                        <geom type="mesh" mesh="Link_2" rgba="1 1 1 1"/>
                        
                        <!-- 末端执行器标记 -->
                        <site name="end_effector" pos="0 0 0.16" size="0.02" rgba="1 0 0 1"/>
                    </body>
                </body>
            </body>
            
            <!-- 地面 -->
            <geom name="ground" type="plane" size="10 10 0.1" rgba="0.3 0.3 0.3 1"/>
            
            <!-- 目标位置标记 -->
            <body name="target" pos="0.3 0 1.2" mocap="true">
                <geom type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    with open(save_path, 'w') as f:
        f.write(model_xml)
    
    print(f"Model saved to {save_path}")
    return save_path


class QuadrotorMPPIController:
    """四旋翼MPPI控制器"""
    
    def __init__(self, model_path: str, params: MPPIParams = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.params = params or MPPIParams()
        
        # 获取body ID
        self.base_id = self.model.body('base_link').id
        
        
        # 控制维度：4旋翼推力 + 2机械臂关节
        self.nu = 6
        
        # 系统质量
        self.total_mass = 1.3  # base + Link_1 + Link_2
        self.hover_thrust_per_rotor = self.total_mass * 9.81 /4
        
        # MPPI缓存
        self.u_init = np.zeros((self.params.horizon, self.nu))
        self.u_prev = np.zeros(self.nu)
        self.noise = np.zeros((self.params.num_samples, self.params.horizon, self.nu))
        self.costs = np.zeros(self.params.num_samples)
        
        # 初始化控制序列（悬停推力）
        self.u_init[:, 0:4] = self.hover_thrust_per_rotor
        
        # 参考状态
        self.x_ref = np.zeros(19)  # 19维状态向量
        self.x_ref[2] = 1.5  # 目标高度
        
        # 力矩系数
        self.km = 0.02  # 旋翼反扭矩系数
        
        print(f"MPPI Controller initialized:")
        print(f"  Total mass: {self.total_mass:.3f} kg")
        print(f"  Hover thrust per rotor: {self.hover_thrust_per_rotor:.2f} N")
        print(f"  Control horizon: {self.params.horizon} steps")
        print(f"  Sample count: {self.params.num_samples}")
    
    def get_state(self) -> np.ndarray:
        """获取当前状态向量[19维]"""
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
    
    def apply_rotor_forces(self, rotor_thrusts: np.ndarray):
        """修复版：应用4个旋翼推力"""
        # 清空之前的力
        self.data.xfrc_applied[self.base_id] = 0
        
        # 限制推力范围
        rotor_thrusts = np.clip(rotor_thrusts, 0, 20)
        
        # 获取机体姿态
        R_body = self.data.xmat[self.base_id].reshape(3, 3)
        
        # 计算总推力（机体坐标系Z方向）
        total_thrust_body = np.array([0, 0, np.sum(rotor_thrusts)])
        total_thrust_world = R_body @ total_thrust_body
        
        # 修正的力矩计算
        # 四旋翼标准配置：前右(1), 前左(2), 后左(3), 后右(4)
        arm_length = 0.148  # 旋翼臂长度 sqrt(0.105^2 + 0.0825^2)
        
        # Roll力矩（绕X轴）：左右推力差
        roll_torque = arm_length * 0.0825/arm_length * (rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[0] - rotor_thrusts[3])
        
        # Pitch力矩（绕Y轴）：前后推力差  
        pitch_torque = arm_length * 0.105/arm_length * (rotor_thrusts[0] + rotor_thrusts[1] - rotor_thrusts[2] - rotor_thrusts[3])
        
        # Yaw力矩（绕Z轴）：反扭矩，假设1,3顺时针，2,4逆时针
        yaw_torque = self.km * (rotor_thrusts[0] - rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[3])
        
        torque_body = np.array([roll_torque, pitch_torque, yaw_torque])
        
        # 应用力和力矩
        self.data.xfrc_applied[self.base_id, 0:3] = total_thrust_world
        self.data.xfrc_applied[self.base_id, 3:6] = torque_body
        
        return total_thrust_world, torque_body
    
    def dynamics_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """简化动力学模型用于MPPI预测"""
        next_state = state.copy()
        
        # 提取状态
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        euler = state[13:16]
        q_arm = state[16:18]
        dq_arm = state[18:20]
        
        # 提取控制输入
        rotor_thrusts = control[0:4]
        tau_arm = control[4:6]
        
        # 限制控制输入
        rotor_thrusts = np.clip(rotor_thrusts, 0, 20)
        tau_arm = np.clip(tau_arm, -6, 6)
        
        # 四元数转旋转矩阵
        R = self.quat_to_rotation(quat)
        
        # 推力动力学
        total_thrust = np.sum(rotor_thrusts)
        thrust_body = np.array([0, 0, total_thrust])
        thrust_world = R @ thrust_body
        
        # 加速度
        acc = thrust_world / self.total_mass + np.array([0, 0, -9.81])
        
        # 力矩动力学（简化）
        arm_length = 0.148
        roll_torque = arm_length * 0.6 * (rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[0] - rotor_thrusts[3])
        pitch_torque = arm_length * 0.7 * (rotor_thrusts[0] + rotor_thrusts[1] - rotor_thrusts[2] - rotor_thrusts[3])
        yaw_torque = self.km * (rotor_thrusts[0] - rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[3])
        
        torque_body = np.array([roll_torque, pitch_torque, yaw_torque])
        
        # 角加速度（简化惯性）
        I_diag = np.array([0.01, 0.01, 0.015])
        alpha = torque_body / I_diag
        
        # 机械臂动力学（简化）
        ddq_arm = tau_arm * 5.0 - dq_arm * 2.0
        
        # 积分更新
        next_state[0:3] = pos + vel * dt
        next_state[3:6] = vel + acc * dt
        next_state[13:16] = euler + omega * dt  # 欧拉角积分（简化）
        next_state[10:13] = omega + alpha * dt
        next_state[16:18] = q_arm + dq_arm * dt
        next_state[18:20] = dq_arm + ddq_arm * dt
        
        # 机械臂关节限位
        next_state[16:18] = np.clip(next_state[16:18], -0.8, 0.8)
        
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
        """计算轨迹代价"""
        cost = 0.0
        p = self.params
        
        for t in range(len(trajectory)):
            x = trajectory[t]
            u = controls[t] if t < len(controls) else np.zeros(self.nu)
            
            # 位置误差
            pos_err = x[0:3] - self.x_ref[0:3]
            cost += p.w_pos * np.sum(pos_err**2)
            
            # 速度惩罚
            cost += p.w_vel * np.sum(x[3:6]**2)
            
            # 姿态稳定（欧拉角）
            euler_err = x[13:16]  # 期望姿态为[0,0,0]
            cost += p.w_att * (euler_err[0]**2 + euler_err[1]**2 + 0.1*euler_err[2]**2)
            
            # 角速度惩罚
            cost += p.w_omega * np.sum(x[10:13]**2)
            
            # 控制代价
            rotor_err = u[0:4] - self.hover_thrust_per_rotor
            cost += p.w_ctrl * (np.sum(rotor_err**2) + 0.1*np.sum(u[4:6]**2))
            
            # 平滑性
            if t > 0:
                du = u - controls[t-1] if t > 0 else u - self.u_prev
                cost += p.w_smooth * np.sum(du**2)
        
        return cost
    
    def mppi_step(self, current_state: np.ndarray, 
                  target_ee: np.ndarray = None) -> np.ndarray:
        """执行一步MPPI优化"""
        # 生成噪声扰动
        for i in range(self.params.num_samples):
            self.noise[i] = np.random.randn(self.params.horizon, self.nu) * self.params.noise_sigma
        
        # 并行采样和评估
        for i in range(self.params.num_samples):
            # 扰动控制序列
            u_sample = self.u_init + self.noise[i]
            
            # 限制控制输入范围
            u_sample[:, 0:4] = np.clip(u_sample[:, 0:4], 0, 20)  # 旋翼推力
            u_sample[:, 4:6] = np.clip(u_sample[:, 4:6], -6, 6)  # 关节力矩
            
            # 前向模拟
            trajectory = [current_state]
            for t in range(self.params.horizon):
                next_state = self.dynamics_step(trajectory[-1], u_sample[t], self.params.dt)
                trajectory.append(next_state)
            
            # 计算代价
            self.costs[i] = self.compute_cost(np.array(trajectory), u_sample, target_ee)
        
        # 计算权重（softmax）
        costs_shifted = self.costs - np.min(self.costs)
        weights = np.exp(-costs_shifted / self.params.lambda_)
        weights /= np.sum(weights)
        
        # 加权平均更新控制序列
        self.u_init = np.sum(weights[:, np.newaxis, np.newaxis] * 
                            (self.u_init + self.noise), axis=0)
        
        # 提取第一个控制
        u_optimal = self.u_init[0].copy()
        
        # 滚动时域
        self.u_init[:-1] = self.u_init[1:]
        self.u_init[-1] = np.array([self.hover_thrust_per_rotor]*4 + [0, 0])
        
        # 保存用于平滑性计算
        self.u_prev = u_optimal.copy()
        
        return u_optimal
    
    def apply_control(self, control: np.ndarray):
        """应用控制输入"""
        # 应用旋翼推力
        self.apply_rotor_forces(control[0:4])
        
        # 应用机械臂控制
        self.data.ctrl[0] = control[4]  # Joint_1
        self.data.ctrl[1] = control[5]  # Joint_2
    
    def run_simulation(self, duration: float = 30.0, visualize: bool = True):
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
        
        control_freq = 20  # Hz
        control_dt = 1.0 / control_freq
        last_control_time = 0
        
        start_time = time.time()
        
        # 保存当前控制量
        current_control = np.array([self.hover_thrust_per_rotor]*4 + [0, 0])
        
        try:
            while time.time() - start_time < duration:
                current_time = self.data.time
                
                # MPPI控制更新
                if current_time - last_control_time >= control_dt:
                    state = self.get_state()
                    current_control = self.mppi_step(state)  # 更新控制量
                    last_control_time = current_time
                    
                    # 打印状态信息
                    if int(current_time) % 3 == 0:
                        pos = state[0:3]
                        euler = state[13:16]
                        rotor_thrusts = current_control[0:4]
                        print(f"t={current_time:.1f}s | "
                            f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                            f"Thrusts: {rotor_thrusts}")
                
                # 每次步进前都施加力！
                self.apply_control(current_control)
                
                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                
                if visualize:
                    viewer.sync()
        
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")

def main():
    # 创建模型
    model_path = create_model_with_rotors("drone_mppi_model.xml")
    
    # 初始化MPPI控制器
    params = MPPIParams(
        horizon=2,
        num_samples=8000,
        lambda_=0.5,
        w_pos=15.0,
        w_att=80.0,
        w_vel=2.0,
        w_ctrl=0.05
    )
    
    controller = QuadrotorMPPIController(model_path, params)
    
    # 设置目标
    controller.x_ref[0:3] = [0.0, 0.0, 1.5]  # 目标位置
    
    # 运行仿真
    controller.run_simulation(duration=9999999.0, visualize=True)


if __name__ == "__main__":
    main()