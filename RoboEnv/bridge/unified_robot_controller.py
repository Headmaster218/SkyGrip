from simulation_and_control.sim.pybullet_robot_interface import SimInterface
from mppi_real_driver import RealRobotController
from simulation_and_control.controllers.servo_motor import MotorCommands
import numpy as np
import time

class UnifiedRobotController:
    def __init__(self, config_path, use_sim=True, use_real=True):
        self.use_sim = use_sim
        self.use_real = use_real
        self.last_real_send_time = 0.0  # 上次发送给实物的时间
        self.prev_angles = [0.0, 0.0]  # 上次发送的角度
        self.real_ctrl_dt = 0.033  # 实物控制周期

        if self.use_sim:
            self.sim = SimInterface(config_path)
        else:
            self.sim = None

        if self.use_real:
            self.real = RealRobotController()
        else:
            self.real = None

    def get_joint_positions(self):
        """
        返回当前舵机位置（单位：弧度），长度与 self.DXL_IDS 一致
        """
        real_offset = np.radians(np.array([115.0, 135.0]) )       # 实物环境的偏移（deg）
        raw_pos =  self.real.get_joint_positions()
        return raw_pos - real_offset

    def send_joint_targets(self, q_des):
        from simulation_and_control.controllers.servo_motor import MotorCommands
        import numpy as np

        # 设置偏移
        sim_offset = np.array([0,0])          # 仿真环境的偏移（rad）
        real_offset = np.array([115.0, 135.0])        # 实物环境的偏移（deg）

        # 给仿真传入的目标值（弧度）
        sim_q_des = np.array(q_des) + sim_offset
        ctrl_value = np.array([[q, 0.0] for q in sim_q_des])  # velocity 设为 0
        control_list = ["position"] * len(q_des)
        cmd = MotorCommands(ctrl_value=ctrl_value, control_list=control_list)

        if self.sim is not None:
            self.sim.Step(cmd)

        now = time.time()
        if now - self.last_real_send_time >= self.real_ctrl_dt:
            if self.real is not None:
                deg_q_des = np.degrees(q_des) + real_offset
                self.real.send_joint_positions(deg_q_des.tolist())
                self.last_real_send_time = now




