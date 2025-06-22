import time
import numpy as np
from bridge.unified_robot_controller import UnifiedRobotController
import pybullet as p

controller = UnifiedRobotController("mppi.json", use_sim=True, use_real=False)

# 控制周期（秒）
dt = 0.001
freq = 0.5  # 频率（Hz）


# 添加滑块
kp_slider = p.addUserDebugParameter("Kp", 0, 1000, 1000)
kd_slider = p.addUserDebugParameter("Kd", 0, 1, 0.4)

kp_slider2 = p.addUserDebugParameter("Kp2", 0, 50, 6.5)
kd_slider2 = p.addUserDebugParameter("Kd2", 0, 5, 0)

angleslider1 = p.addUserDebugParameter("Angle1", -1.5, 1.5, 0)
angleslider2 = p.addUserDebugParameter("Angle2", -1.5, 1.5, 0)

# 控制循环
t0 = time.time()
while True:

    # 读取滑块参数
    kp_val = p.readUserDebugParameter(kp_slider)
    kd_val = p.readUserDebugParameter(kd_slider)

    kp_val2 = p.readUserDebugParameter(kp_slider2)
    kd_val2 = p.readUserDebugParameter(kd_slider2)

    # 应用新参数
    controller.sim.bot[0].servo_motor_model.set_motor_gains(kp=[kp_val,kp_val2 ], kd=[kd_val,kd_val2])

    t = time.time() - t0
    angle1 = 1 * np.sin(2 * np.pi * freq * t)  # 正弦波：-0.5 ~ +0.5 rad
    angle2 = -1 * np.sin(2 * np.pi * freq * t)  # 可设为不同相位或幅值

    angle1 = 0
    angle2 = 0

    angle1 = p.readUserDebugParameter(angleslider1)
    angle2 = p.readUserDebugParameter(angleslider2)

    # limit angles to [-1.6, 1.6]
    angle1 = np.clip(angle1, -1.6, 1.6)
    angle2 = np.clip(angle2, -1.6, 1.6)
    controller.send_joint_targets([angle1, angle2])
    time.sleep(dt)
