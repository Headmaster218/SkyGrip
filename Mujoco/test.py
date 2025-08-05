import mujoco
from mujoco import viewer
import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK

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
        self.TORQUE_ENABLE = 0

        self.angle_offset = [-120,180]

        # 初始化串口和协议处理器
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # 打开串口
        if not self.portHandler.openPort():
            raise RuntimeError("❌ 串口打开失败")
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("❌ 设置波特率失败")

        # 启用力矩
        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID {dxl_id}] 通信失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID {dxl_id}] 错误: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"[ID {dxl_id}] 力矩启用 ✅")

        # 初始化同步写对象
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION
        )


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
        degrees: List[float], 每个关节的角度（单位：度），必须与 self.DXL_IDS 一一对应
        使用 SyncWrite 同步发送角度指令，提高效率
        
        check_response: bool, 是否检查通信返回值，默认不检查以提高速度
        """
        assert hasattr(self, "DXL_IDS"), "DXL_IDS 尚未初始化"
        assert len(degrees) == len(self.DXL_IDS), "角度与舵机ID数量不一致"

        self.groupSyncWrite.clearParam()

        for dxl_id, degree in zip(self.DXL_IDS, degrees):
            pos_val = self.degree_to_position(degree)
            param_goal_pos = [
                pos_val & 0xFF,
                (pos_val >> 8) & 0xFF,
                (pos_val >> 16) & 0xFF,
                (pos_val >> 24) & 0xFF
            ]
            success = self.groupSyncWrite.addParam(dxl_id, param_goal_pos)
            if not success and check_response:
                print(f"[ID {dxl_id}] ❌ 添加同步参数失败")

        result = self.groupSyncWrite.txPacket()

        if check_response:
            if result != COMM_SUCCESS:
                print("❌ 同步写入失败:", self.packetHandler.getTxRxResult(result))
            else:
                print("✅ 同步写入成功")


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



def apply_real_state_to_sim(data, joint_qpos, joint_qvel):

    data.qpos[:len(joint_qpos)] = joint_qpos

    data.qvel[:len(joint_qvel)] = joint_qvel

    mujoco.mj_forward(model, data)  # 更新派生量

real_controller = RealRobotController()

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # 获取实际关节状态
        joint_qpos, joint_qvel = real_controller.get_joint_state()
        
        # 将实际状态应用到仿真中
        apply_real_state_to_sim(data, joint_qpos, joint_qvel)
        # mujoco.mj_step(model, data)
        v.sync()
