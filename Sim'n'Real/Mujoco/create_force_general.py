import numpy as np
import mujoco
import mujoco.viewer
from typing import List
import time
from dataclasses import dataclass

@dataclass
class MPPIParams:
    """优化的MPPI参数"""
    horizon: int = 10              # 预测时域步数（减少计算量）
    num_samples: int = 50          # 采样轨迹数（减少计算量）
    lambda_: float = 1.0           # 温度参数
    dt: float = 0.05               # 控制时间步长（增大以减少仿真次数）
    
    # 简化的代价函数权重
    w_pos: float = 100.0           # 位置权重
    w_vel: float = 15.0            # 速度权重
    w_att: float = 30.0            # 姿态权重
    w_omega: float = 10.0          # 角速度权重
    w_ctrl: float = 0.001          # 控制量权重（减小以允许更大控制）
    w_smooth: float = 0.1          # 平滑性权重（减小以提高响应速度）
    
    def __post_init__(self):
        # 控制噪声标准差 [推力, 3个扭矩, 2个关节力矩]
        self.noise_sigma = np.array([
            2.0,                       # 推力噪声（增大探索）
            0.03, 0.03, 0.02,         # 扭矩噪声
            0.1, 0.1                  # 关节力矩噪声
        ])


class PureMPPIController:
    """纯MPPI控制器 - 无任何补偿"""
    
    def __init__(self, model_path: str, params: MPPIParams = None):
        """初始化控制器"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.params = params or MPPIParams()
        
        # 创建仿真数据池
        print(f"Creating {self.params.num_samples} simulation instances...")
        self.sim_data_pool = []
        for i in range(self.params.num_samples):
            self.sim_data_pool.append(mujoco.MjData(self.model))
        
        # 获取body ID
        try:
            self.base_id = self.model.body('base_link').id
        except:
            self.base_id = 1
            print("Warning: Could not find base_link, using default ID")
        
        # 基本参数
        self.total_mass = 1.185  # kg
        self.nominal_hover_thrust = self.total_mass * 9.81
        self.nu = 6  # 控制维度
        
        # 控制限制
        self.thrust_limits = [0, 30]
        self.torque_limits = [-1, 1]
        self.joint_limits = [-2, 2]
        
        # MPPI缓存
        self.u_init = np.zeros((self.params.horizon, self.nu))
        self.u_init[:, 0] = self.nominal_hover_thrust  # 初始化为悬停推力
        self.u_prev = np.zeros(self.nu)
        self.noise = np.zeros((self.params.num_samples, self.params.horizon, self.nu))
        self.costs = np.zeros(self.params.num_samples)
        
        # 参考状态（只跟踪位置）
        self.target_pos = np.array([0, 0, 1.5])
        
        # 轨迹跟踪
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.1
        
        print(f"Pure MPPI Controller initialized:")
        print(f"  Mass: {self.total_mass:.3f} kg")
        print(f"  Hover thrust: {self.nominal_hover_thrust:.2f} N")
        print(f"  Horizon: {self.params.horizon} steps")
        print(f"  Samples: {self.params.num_samples}")
        print(f"  Control dt: {self.params.dt:.3f} s")
    
    def get_state(self, data=None):
        """获取系统状态"""
        if data is None:
            data = self.data
        
        # 位置和速度
        pos = data.qpos[0:3].copy()
        vel = data.qvel[0:3].copy()
        
        # 四元数和角速度
        quat = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        
        # 关节状态
        q_joints = data.qpos[7:9].copy() if len(data.qpos) > 8 else np.zeros(2)
        dq_joints = data.qvel[6:8].copy() if len(data.qvel) > 7 else np.zeros(2)
        
        # 计算欧拉角（用于代价函数）
        R = data.xmat[self.base_id].reshape(3, 3)
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        yaw = np.arctan2(R[1,0], R[0,0])
        
        return {
            'pos': pos,
            'vel': vel,
            'quat': quat,
            'omega': omega,
            'euler': np.array([roll, pitch, yaw]),
            'q_joints': q_joints,
            'dq_joints': dq_joints
        }
    
    def simulate_step(self, sim_data, control, dt):
        """使用MuJoCo仿真一个控制步"""
        # 清除之前的力
        sim_data.xfrc_applied[:] = 0
        
        # 限制控制输入
        thrust = np.clip(control[0], self.thrust_limits[0], self.thrust_limits[1])
        torques = np.clip(control[1:4], self.torque_limits[0], self.torque_limits[1])
        joint_torques = np.clip(control[4:6], self.joint_limits[0], self.joint_limits[1])
        
        # 应用推力（body坐标系转世界坐标系）
        R = sim_data.xmat[self.base_id].reshape(3, 3)
        thrust_body = np.array([0, 0, thrust])
        thrust_world = R @ thrust_body
        
        # 应用力和扭矩
        sim_data.xfrc_applied[self.base_id, 0:3] = thrust_world
        sim_data.xfrc_applied[self.base_id, 3:6] = torques
        
        # 应用关节控制
        if len(sim_data.ctrl) >= 2:
            sim_data.ctrl[0] = joint_torques[0]
            sim_data.ctrl[1] = joint_torques[1]
        
        # 执行仿真步进
        n_steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, sim_data)
        
        return self.get_state(sim_data)
    
    def compute_trajectory_cost(self, trajectory, controls):
        """计算轨迹代价（纯MPPI代价函数）"""
        cost = 0.0
        
        for t in range(len(trajectory)):
            state = trajectory[t]
            
            # 位置误差
            pos_error = state['pos'] - self.target_pos
            cost += self.params.w_pos * np.sum(pos_error**2)
            
            # 速度惩罚（希望稳定）
            cost += self.params.w_vel * np.sum(state['vel']**2)
            
            # 姿态惩罚（希望水平）
            euler = state['euler']
            cost += self.params.w_att * (euler[0]**2 + euler[1]**2 + 0.1*euler[2]**2)
            
            # 角速度惩罚（希望稳定）
            cost += self.params.w_omega * np.sum(state['omega']**2)
            
            # 控制惩罚
            if t < len(controls):
                u = controls[t]
                # 推力偏离悬停的惩罚
                thrust_error = (u[0] - self.nominal_hover_thrust) / self.nominal_hover_thrust
                cost += self.params.w_ctrl * (thrust_error**2 + 10*np.sum(u[1:4]**2) + np.sum(u[4:6]**2))
                
                # 控制平滑性
                if t > 0:
                    du = u - controls[t-1]
                    cost += self.params.w_smooth * (0.01*du[0]**2 + np.sum(du[1:4]**2) + 0.1*np.sum(du[4:6]**2))
            
            # 大倾角惩罚
            if abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5:
                cost += 500.0
            
            # 进步奖励（鼓励向目标移动）
            if t > 0:
                prev_error = np.linalg.norm(trajectory[t-1]['pos'] - self.target_pos)
                curr_error = np.linalg.norm(pos_error)
                progress = prev_error - curr_error
                cost -= 20.0 * progress  # 奖励进步
        
        return cost
    
    def mppi_step(self, current_state):
        """执行一步MPPI优化（纯MPPI，无补偿）"""
        # 重新初始化（如果偏差太大）
        pos_error = np.linalg.norm(current_state['pos'] - self.target_pos)
        if pos_error > 1.0:
            self.u_init[:, 0] = self.nominal_hover_thrust * 1.1
            self.u_init[:, 1:4] = 0
            self.u_init[:, 4:6] = 0
        
        # 生成噪声扰动
        self.noise = np.random.randn(self.params.num_samples, self.params.horizon, self.nu)
        self.noise *= self.params.noise_sigma
        
        # 保存当前仿真状态
        saved_state = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'ctrl': self.data.ctrl.copy(),
            'time': self.data.time
        }
        
        # 评估每个样本
        valid_samples = 0
        for i in range(self.params.num_samples):
            # 获取仿真数据
            sim_data = self.sim_data_pool[i]
            
            # 恢复初始状态
            sim_data.qpos[:] = saved_state['qpos']
            sim_data.qvel[:] = saved_state['qvel']
            sim_data.ctrl[:] = saved_state['ctrl']
            sim_data.time = saved_state['time']
            mujoco.mj_forward(self.model, sim_data)
            
            # 生成扰动控制序列
            u_sample = self.u_init + self.noise[i]
            
            # 基本限制（软限制，允许MPPI探索）
            u_sample[:, 0] = np.clip(u_sample[:, 0], 
                                     self.nominal_hover_thrust * 0.3, 
                                     self.nominal_hover_thrust * 2.0)
            u_sample[:, 1:4] = np.clip(u_sample[:, 1:4], -0.5, 0.5)
            u_sample[:, 4:6] = np.clip(u_sample[:, 4:6], -1.0, 1.0)
            
            # 仿真轨迹
            trajectory = []
            trajectory.append(current_state)
            
            try:
                for t in range(self.params.horizon):
                    next_state = self.simulate_step(sim_data, u_sample[t], self.params.dt)
                    trajectory.append(next_state)
                    
                    # 检查状态有效性
                    if not np.all(np.isfinite(next_state['pos'])):
                        self.costs[i] = 1e6
                        break
                else:
                    # 计算代价
                    self.costs[i] = self.compute_trajectory_cost(trajectory, u_sample)
                    valid_samples += 1
            except:
                self.costs[i] = 1e6
        
        # 检查是否有有效样本
        if valid_samples == 0:
            print("WARNING: No valid samples in MPPI")
            return np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        # 找到有效样本
        valid_mask = self.costs < 1e5
        if not np.any(valid_mask):
            return np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        valid_costs = self.costs[valid_mask]
        valid_noise = self.noise[valid_mask]
        
        # 计算softmax权重
        min_cost = np.min(valid_costs)
        costs_shifted = valid_costs - min_cost
        weights = np.exp(-costs_shifted / self.params.lambda_)
        weights /= np.sum(weights)
        
        # 加权平均更新控制序列
        weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * valid_noise, axis=0)
        self.u_init = self.u_init + weighted_noise
        
        # 提取第一个控制（这就是纯MPPI输出）
        u_optimal = self.u_init[0].copy()
        
        # 硬限制（确保安全）
        u_optimal[0] = np.clip(u_optimal[0], 
                              self.nominal_hover_thrust * 0.5,
                              self.nominal_hover_thrust * 1.8)
        u_optimal[1:4] = np.clip(u_optimal[1:4], -0.5, 0.5)
        u_optimal[4:6] = np.clip(u_optimal[4:6], -1.5, 1.5)
        
        # 时域滚动
        self.u_init[:-1] = self.u_init[1:]
        self.u_init[-1] = np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        # 保存用于下次迭代
        self.u_prev = u_optimal.copy()
        
        return u_optimal
    
    def apply_control(self, control):
        """应用控制到实际系统"""
        # 清除之前的力
        self.data.xfrc_applied[:] = 0
        
        # 应用推力
        R = self.data.xmat[self.base_id].reshape(3, 3)
        thrust_body = np.array([0, 0, control[0]])
        thrust_world = R @ thrust_body
        
        self.data.xfrc_applied[self.base_id, 0:3] = thrust_world
        self.data.xfrc_applied[self.base_id, 3:6] = control[1:4]
        
        # 应用关节控制
        if len(self.data.ctrl) >= 2:
            self.data.ctrl[0] = control[4]
            self.data.ctrl[1] = control[5]
    
    def set_trajectory(self, waypoints: List[np.ndarray]):
        """设置轨迹路径点"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        if len(waypoints) > 0:
            self.target_pos = waypoints[0]
        
        print(f"Trajectory set with {len(waypoints)} waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"  Waypoint {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]")
    
    def update_target(self, current_pos):
        """更新目标路径点"""
        if len(self.waypoints) == 0:
            return
        
        # 检查是否到达当前路径点
        distance = np.linalg.norm(current_pos - self.target_pos)
        if distance < self.waypoint_tolerance:
            print(f"  -> Reached waypoint {self.current_waypoint_idx}")
            self.current_waypoint_idx += 1
            
            # 更新到下一个路径点
            if self.current_waypoint_idx < len(self.waypoints):
                self.target_pos = self.waypoints[self.current_waypoint_idx]
                print(f"  -> New target: [{self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}]")
            else:
                print("  -> All waypoints completed!")
    
    def run_simulation(self, duration=20.0, visualize=True):
        """运行仿真主循环"""
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
        
        # 控制参数
        control_freq = 20  # Hz
        control_dt = 1.0 / control_freq
        last_control_time = 0
        
        # 初始化状态
        self.data.qpos[2] = 1.5  # 初始高度
        if len(self.data.qpos) > 8:
            self.data.qpos[7] = 0  # 关节1
            self.data.qpos[8] = 0  # 关节2
        mujoco.mj_forward(self.model, self.data)
        
        # 初始控制
        current_control = np.array([self.nominal_hover_thrust, 0, 0, 0, 0, 0])
        
        # 统计
        sim_start_time = self.data.time
        real_start_time = time.time()
        mppi_compute_time = 0
        mppi_count = 0
        position_errors = []
        
        print("\n" + "="*60)
        print("Starting Pure MPPI Simulation (No Compensation)")
        print("="*60 + "\n")
        
        last_print_time = 0
        
        try:
            while self.data.time - sim_start_time < duration:
                current_time = self.data.time
                
                # MPPI控制更新
                if current_time - last_control_time >= control_dt:
                    # 获取当前状态
                    state = self.get_state()
                    
                    # 更新目标
                    self.update_target(state['pos'])
                    
                    # MPPI计算
                    mppi_start = time.time()
                    current_control = self.mppi_step(state)
                    mppi_compute_time += time.time() - mppi_start
                    mppi_count += 1
                    
                    last_control_time = current_time
                    
                    # 记录误差
                    pos_error = np.linalg.norm(state['pos'] - self.target_pos)
                    position_errors.append(pos_error)
                
                # 应用控制
                self.apply_control(current_control)
                
                # 仿真步进
                mujoco.mj_step(self.model, self.data)
                
                # 打印状态
                if current_time - last_print_time >= 0.5:
                    state = self.get_state()
                    pos = state['pos']
                    euler = state['euler']
                    pos_error = np.linalg.norm(pos - self.target_pos)
                    
                    print(f"t={current_time:5.1f}s | "
                          f"Pos=[{pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}] | "
                          f"R/P=[{np.rad2deg(euler[0]):5.1f}°,{np.rad2deg(euler[1]):5.1f}°] | "
                          f"T={current_control[0]:5.1f}N | "
                          f"Err={pos_error:.3f}m")
                    
                    last_print_time = current_time
                
                if visualize:
                    viewer.sync()
        
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            if visualize:
                viewer.close()
        
        # 打印统计
        print("\n" + "="*60)
        print("Performance Statistics")
        print("="*60)
        
        if mppi_count > 0:
            avg_mppi_time = mppi_compute_time / mppi_count * 1000
            real_time_factor = (self.data.time - sim_start_time) / (time.time() - real_start_time)
            
            print(f"\nComputation Performance:")
            print(f"  Average MPPI time: {avg_mppi_time:.1f} ms")
            print(f"  Real-time factor: {real_time_factor:.2f}x")
            
            if avg_mppi_time > 50:
                print(f"  Note: MPPI is slow. Consider:")
                print(f"    - Reducing horizon (current: {self.params.horizon})")
                print(f"    - Reducing samples (current: {self.params.num_samples})")
                print(f"    - Increasing dt (current: {self.params.dt})")
        
        if position_errors:
            avg_error = np.mean(position_errors)
            max_error = np.max(position_errors)
            final_error = position_errors[-1]
            
            print(f"\nTracking Performance:")
            print(f"  Average error: {avg_error:.4f} m")
            print(f"  Maximum error: {max_error:.4f} m")
            print(f"  Final error: {final_error:.4f} m")
            
            if self.current_waypoint_idx >= len(self.waypoints):
                print(f"  Status: All {len(self.waypoints)} waypoints completed!")
            else:
                print(f"  Status: {self.current_waypoint_idx}/{len(self.waypoints)} waypoints completed")


def test_pure_mppi():
    """测试纯MPPI控制器"""
    print("\n" + "="*60)
    print("Pure MPPI Controller Test")
    print("="*60)
    print("This is a pure MPPI implementation:")
    print("  - No compensation functions")
    print("  - No manual PD control")
    print("  - Only MPPI optimization")
    print("  - Direct MuJoCo simulation for prediction")
    
    # 创建控制器
    params = MPPIParams(
        horizon=10,
        num_samples=50,
        lambda_=1.0,
        dt=0.05,
        w_pos=100.0,
        w_vel=15.0,
        w_att=30.0,
        w_omega=10.0,
        w_ctrl=0.001,
        w_smooth=0.1
    )
    
    controller = PureMPPIController("drone_direct_control.xml", params)
    
    # 设置轨迹
    waypoints = [
        np.array([0.0, 0.0, 1.5]),   # 悬停
        np.array([0.5, 0.0, 1.5]),   # 前进
        np.array([0.5, 0.5, 1.5]),   # 右转
        np.array([0.0, 0.5, 1.5]),   # 后退
        np.array([0.0, 0.0, 1.5]),   # 返回
    ]
    controller.set_trajectory(waypoints)
    
    # 运行仿真
    controller.run_simulation(duration=30.0, visualize=True)


def test_hovering():
    """测试悬停性能"""
    print("\n" + "="*60)
    print("Pure MPPI Hovering Test")
    print("="*60)
    
    params = MPPIParams(
        horizon=8,
        num_samples=30,
        lambda_=0.5,
        dt=0.05,
        w_pos=200.0,  # 增加位置权重
        w_vel=20.0,
        w_att=50.0,
        w_omega=15.0,
        w_ctrl=0.0001,  # 减小控制惩罚
        w_smooth=0.05
    )
    
    controller = PureMPPIController("drone_direct_control.xml", params)
    
    # 只设置一个悬停点
    waypoints = [np.array([0.0, 0.0, 1.5])]
    controller.set_trajectory(waypoints)
    
    # 运行仿真
    controller.run_simulation(duration=10.0, visualize=True)


if __name__ == "__main__":
    # 选择测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hover":
        test_hovering()
    else:
        test_pure_mppi()