from dynamixel_sdk import *  # Dynamixel SDK
import numpy as np
class RealRobotController:
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=57600, dxl_ids=[0, 1]):
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.DXL_IDS = dxl_ids

        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116  # 4 bytes
        self.LEN_GOAL_POSITION = 4
        self.ADDR_PRESENT_POSITION = 132  # 4 bytes
        self.TORQUE_ENABLE = 1

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
