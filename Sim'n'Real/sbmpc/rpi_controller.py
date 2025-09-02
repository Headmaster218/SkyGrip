from dynamixel_sdk import *  # Dynamixel SDK
import socket
import numpy as np
import struct

host_ip =  '192.168.0.235'
host_port = 5605
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5606))   # 监听本机 5005 端口
def recv_latest(sock, fmt="ffi", bufsize=1024):
    """
    从UDP缓冲区取出所有数据，只返回 i 最大的那条
    :param sock: 已经 bind 的 UDP socket
    :param fmt: struct 格式串，默认 "ffi" (float, float, int)
    :param bufsize: 单个 UDP 包最大字节数
    :return: (unpacked_data, addr) 或 None
    """
    size = struct.calcsize(fmt)
    newest = None
    max_i = -float("inf")

    sock.setblocking(False)  # 设置非阻塞
    while True:
        try:
            data, addr = sock.recvfrom(bufsize)
            values = struct.unpack(fmt, data[:size])
            *others, i = values
            if i > max_i:
                max_i = i
                newest = values
        except BlockingIOError:
            break  # 缓冲区已空

    sock.setblocking(True)   # 恢复阻塞
    return newest



class RealRobotController:
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=1000000, dxl_ids=[1, 2]):
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.DXL_IDS = dxl_ids

        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116  # 4 bytes
        self.LEN_GOAL_POSITION = 4
        self.ADDR_PRESENT_POSITION = 132  # X 系列 Present Position
        self.ADDR_PRESENT_VELOCITY = 128  # X 系列 Present Velocity
        self.LEN_4 = 4
        # self.TORQUE_ENABLE = 1

        self.angle_offset = [-115,-180]
        self.max_torque_per_joint = [4.5, 1]
        self.max_pwm_val = 855

        self.e_break = [0.235, 0.372]  # 电子制动系数
        self.stall_torque = [3.0, 1.5]  # 每个关节的额定力矩（Nm）
        self.torque_limit = [0.5, 0.2]  # 每个关节的力矩限制（Nm）
        self.full_speed = [77 * 2 * np.pi / 60 , 57 * 2 * np.pi / 60]  # 每个关节的最大速度（弧度/秒）


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

        # 初始化 GroupSyncRead
        self.groupSyncRead = GroupSyncRead(
            self.portHandler, self.packetHandler,
            self.ADDR_PRESENT_VELOCITY, 8
        )
        for dxl_id in self.DXL_IDS:
            if not self.groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"GroupSyncRead addParam failed: ID={dxl_id}")

        # 当前模式缓存（ID → mode），mode：0=PWM, 1=Current, 3=Position 等
        self.current_mode_map = {dxl_id: None for dxl_id in self.DXL_IDS}
        self.pwmgroupWrite = GroupSyncWrite(self.portHandler, self.packetHandler, 100, 2)


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

        groupSyncWrite = self.pwmgroupWrite
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

        #apply torque limits
        for i in range(len(torque_vals)):
            torque_vals[i] = np.clip(torque_vals[i], -self.torque_limit[i], self.torque_limit[i])

        pwm_vals = []
        for idx, tau in enumerate(torque_vals):
            max_tau = self.max_torque_per_joint[idx]
            pwm = (tau / max_tau) * self.max_pwm_val
            pwm_vals.append(int(np.clip(pwm, -self.max_pwm_val, self.max_pwm_val)))

        self.send_pwm(pwm_vals, check_response=check_response)

    def close(self):
        self.portHandler.closePort()

    def get_joint_state(self):
        # 一次性请求
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("SyncRead failed:", self.packetHandler.getTxRxResult(dxl_comm_result))
            return [0.0]*len(self.DXL_IDS), [0.0]*len(self.DXL_IDS)

        qpos, qvel = [], []

        for i, dxl_id in enumerate(self.DXL_IDS):
            if not self.groupSyncRead.isAvailable(dxl_id, self.ADDR_PRESENT_VELOCITY, 8):
                qpos.append(0.0)
                qvel.append(0.0)
                continue

            vel_bytes = self.groupSyncRead.getData(dxl_id, 128, 4).to_bytes(4,'little',signed=False)
            pos_bytes = self.groupSyncRead.getData(dxl_id, 132, 4).to_bytes(4,'little',signed=False)
            vel_val = int.from_bytes(vel_bytes,'little',signed=True)
            pos_val = int.from_bytes(pos_bytes,'little',signed=False)


            # 速度：带符号，单位 0.229 rpm/bit
            rpm = (vel_val / 1023.0) * 117.0
            rad_per_sec = rpm * 2 * np.pi / 60
            qvel.append(rad_per_sec)

            # 位置：无符号，0~4095 → 0~360deg
            degree = (pos_val / 4095.0) * 360.0 + self.angle_offset[i]
            qpos.append(np.radians(degree))

        return qpos, qvel



real_controller = RealRobotController( device_name='/dev/ttyUSB0' )

i = 0

while True:
	i+=1
	qpos, qvel = real_controller.get_joint_state()
	sock.sendto(struct.pack("ffffi", qpos[0], qpos[1], qvel[0], qvel[1], i), (host_ip, host_port))
	values = recv_latest(sock, "ffi")
	if values is not None:
		real_controller.send_torque([values[0],values[1]])
		print('success control'+str(i))
