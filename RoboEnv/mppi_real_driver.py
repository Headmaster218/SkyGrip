# my_real_driver.py

from dynamixel_sdk import *  # Dynamixel SDK

class RealRobotController:
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=57600, dxl_ids=[0, 1]):
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.DXL_IDS = dxl_ids

        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.TORQUE_ENABLE = 1

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            raise RuntimeError("❌ 串口打开失败")
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("❌ 设置波特率失败")

        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID {dxl_id}] 通信失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID {dxl_id}] 错误: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"[ID {dxl_id}] 力矩启用 ✅")

    def degree_to_position(self, degree):
        degree = degree % 360
        return int((degree / 360.0) * 4095)

    def send_joint_positions(self, degrees):
        """
        degrees: List[float], 每个关节的角度（单位：度），必须与 self.DXL_IDS 一一对应
        """
        assert hasattr(self, "DXL_IDS"), "DXL_IDS 尚未初始化"
        assert len(degrees) == len(self.DXL_IDS), "角度与舵机ID数量不一致"

        print(f"发送角度: {degrees}")
        
        for dxl_id, degree in zip(self.DXL_IDS, degrees):
            pos_val = int(self.degree_to_position(degree))  # 确保为Python int
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, pos_val)

            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID {dxl_id}] 写入失败: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"[ID {dxl_id}] 错误: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"[ID {dxl_id}] ✅ 成功发送角度 {degree}° => pos {pos_val}")

    def close(self):
        self.portHandler.closePort()
