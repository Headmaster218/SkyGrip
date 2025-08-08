import mujoco
from mujoco import viewer
import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK
import time

model = mujoco.MjModel.from_xml_path("mjmodel.xml")
data = mujoco.MjData(model)

class RealRobotController:
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=115200, dxl_ids=[0, 1]):
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.DXL_IDS = dxl_ids

        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116  # 4 bytes
        self.LEN_GOAL_POSITION = 4
        self.ADDR_PRESENT_POSITION = 132  # 4 bytes
        # self.TORQUE_ENABLE = 1

        self.angle_offset = [-115,180]
        self.max_torque_per_joint = [4.5, 1]
        self.max_pwm_val = 855

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
        # 当前模式缓存（ID → mode），mode：0=PWM, 1=Current, 3=Position 等
        self.current_mode_map = {dxl_id: None for dxl_id in self.DXL_IDS}


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

        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_PWM, LEN_PWM)
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

        pwm_vals = []
        for idx, tau in enumerate(torque_vals):
            max_tau = self.max_torque_per_joint[idx]
            pwm = (tau / max_tau) * self.max_pwm_val
            pwm_vals.append(int(np.clip(pwm, -self.max_pwm_val, self.max_pwm_val)))

        self.send_pwm(pwm_vals, check_response=check_response)

    def close(self):
        self.portHandler.closePort()

    def get_joint_state(self):

        """
        获取当前所有关节的角度（弧度）和速度（弧度/秒），用于仿真同步
        返回：
            qpos: List[float] 角度（rad）
            qvel: List[float] 速度（rad/s）—— 若舵机不支持读取速度，则为零向量
        """
        qpos = []
        qvel = []

        for i, dxl_id in enumerate(self.DXL_IDS):
            # --- 读取角度 ---
            pos_result, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID {dxl_id}] 读取角度失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
                qpos.append(0.0)
            elif dxl_error != 0:
                print(f"[ID {dxl_id}] 错误: {self.packetHandler.getRxPacketError(dxl_error)}")
                qpos.append(0.0)
            else:
                degree = (pos_result / 4095.0) * 360.0 + self.angle_offset[i]
                qpos.append(np.radians(degree))

            # --- 读取速度（若支持） ---
            # Dynamixel XL-320 / X 系列一般使用地址 128
            vel_result, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, dxl_id, 128  # PRESENT_VELOCITY
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                qvel.append(0.0)
            else:
                # Dynamixel 的速度单位是：0 ~ 1023 对应 0 ~ 最大转速（通常为 ~117 RPM）
                # 这里假设最大值对应 117 RPM = 12.25 rad/s，映射线性计算
                rpm = (vel_result / 1023.0) * 117.0
                rad_per_sec = rpm * 2 * np.pi / 60
                qvel.append(rad_per_sec)

        return qpos, qvel

class SynchronizedTorqueController:
    def __init__(self, real_robot_controller, mujoco_model, mujoco_data, enable_real=True, enable_sim=True):
        """
        real_robot_controller: 实例化后的 RealRobotController
        mujoco_model, mujoco_data: MuJoCo 模型与数据对象
        enable_real, enable_sim: 控制是否向现实 / 仿真发送力矩指令
        """
        self.real_robot = real_robot_controller
        self.model = mujoco_model
        self.data = mujoco_data

    def send_torque(self, torque_vals, check_response=False, enable_sim=True, enable_real=True):
        """
        同步向仿真和现实发送力矩控制指令
        torque_vals: List[float]，单位为 Nm
        """
        assert isinstance(torque_vals, (list, tuple, np.ndarray)), "输入应为列表/数组"

        # 发送给仿真
        if enable_sim:
            for i in range(len(torque_vals)):
                self.data.ctrl[i] = torque_vals[i]

        # 发送给现实
        if enable_real:
            assert len(torque_vals) == len(self.real_robot.DXL_IDS), "力矩数量必须与舵机 ID 一致"
            self.real_robot.send_torque(torque_vals, check_response=check_response)


def apply_real_state_to_sim(data, joint_qpos, joint_qvel):
    data.qpos[:len(joint_qpos)] = joint_qpos
    data.qvel[:len(joint_qvel)] = joint_qvel
    mujoco.mj_forward(model, data)  # 更新派生量

try:
    real_controller = RealRobotController()

except Exception as e:
    print(f"❌ 初始化 RealRobotController 失败: {e}")
    real_controller = None

sync_ctrl = SynchronizedTorqueController(
    real_robot_controller=real_controller,
    mujoco_model=model,
    mujoco_data=data,
    enable_real=True,
    enable_sim=False
)



with viewer.launch_passive(model, data) as v:
    step_counter = 0
    torque_val = np.array([4, 1])  # 初始力矩

    while v.is_running():
        # 每 N 步反向力矩
        if step_counter % 30 == 0:
            torque_val = -torque_val
            # apply_real_state_to_sim(data, real_controller.get_joint_state()[0], real_controller.get_joint_state()[1])

        # 控制仿真 & 实物
        sync_ctrl.send_torque(torque_val, enable_real=False, enable_sim=True)

        # 推进仿真一帧
        mujoco.mj_step(model, sync_ctrl.data)

        # 可视化刷新
        v.sync()
        step_counter += 1
        time.sleep(0.01)  # 控制帧率，避免过快

