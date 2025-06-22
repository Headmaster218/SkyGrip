import time
import numpy as np
from bridge.unified_robot_controller import UnifiedRobotController
import pybullet as p

controller = UnifiedRobotController("mppi.json", use_sim=True, use_real=True)

angle_slider1 = p.addUserDebugParameter("Angle1", -1.6, 1.6, 0)
angle_slider2 = p.addUserDebugParameter("Angle2", -1.6, 1.6, 0)

dt = 0.001  # 控制周期

class SmoothPositionController:
    def __init__(self, n_joints, max_vel=10.0, accel=4.0, dt=0.001):
        self.n = n_joints
        self.max_vel = max_vel
        self.accel = accel
        self.dt = dt
        self.pos = np.zeros(n_joints)  # 当前角度
        self.vel = np.zeros(n_joints)
        self.target = np.zeros(n_joints)

    def set_target(self, target):
        self.target = np.array(target)

    def update(self):
        next_pos = np.zeros(self.n)
        for i in range(self.n):
            delta = self.target[i] - self.pos[i]
            direction = np.sign(delta)

            # 计算制动距离
            stop_dist = (self.vel[i] ** 2) / (2 * self.accel)

            if abs(delta) < 1e-4:
                self.vel[i] = 0.0
                next_pos[i] = self.target[i]
                continue

            # 减速阶段
            if abs(delta) < stop_dist:
                self.vel[i] -= direction * self.accel * self.dt
            else:
                self.vel[i] += direction * self.accel * self.dt

            # 限制速度
            self.vel[i] = np.clip(self.vel[i], -self.max_vel, self.max_vel)

            # 更新位置
            self.pos[i] += self.vel[i] * self.dt

        return self.pos.tolist()


smooth_controller = SmoothPositionController(n_joints=2, max_vel=1.0, accel=2.0, dt=0.002)

# 初始化目标
prev_target = [0.0, 0.0]
smooth_controller.set_target(prev_target)

while True:
    # 读取目标角度
    new_target = [
        np.clip(p.readUserDebugParameter(angle_slider1), -1.6, 1.6),
        np.clip(p.readUserDebugParameter(angle_slider2), -1.6, 1.6)
    ]

    # 如果目标变化，更新插值目标
    if not np.allclose(new_target, prev_target, atol=1e-4):
        smooth_controller.set_target(new_target)
        prev_target = new_target

    # 获取平滑插值结果
    smooth_angles = smooth_controller.update()

    controller.send_joint_targets(smooth_angles)
    # time.sleep(dt)

