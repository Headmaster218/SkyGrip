from dynamixel_sdk import *  # Dynamixel SDK
import socket
import numpy as np
import struct
import serial
import struct
import time
import numpy as np
import time


# === 串口配置 ===
SERIAL_PORT = '/dev/ttyACM0'     # 根据你的系统改成对应串口，比如 '/dev/ttyACM0' 或 'COM3'
BAUDRATE = 1000000


# === 串口通信函数 ===
def send_pwm(ser, t, r, p, y):
    # 注意：STM32 是大端还是小端？默认我们用大端（如果你用的是 `>HHHH`）
    packet = struct.pack('>HHHH', t, r, p, y)
    ser.write(packet)


# === 常量 ===
K_F = 0.1667        # N / PWM  ← 16.96 g × 9.81e-3
K_M = 6.80e-4       # N·m / PWM

# 机架坐标 (m)
a, b = 0.1116, 0.0856
x_pos  = np.array([ +a,  +a, -a, -a ])   # RF, LF, RB, LB
y_pos  = np.array([ +b,  -b, +b, -b ])
spin   = np.array([ -1,  +1, +1, -1 ])   # CW, CCW, CCW, CW

def pwm_from_wrench_N_fullmodel(total_thrust_N,
                                 tau_roll, tau_pitch, tau_yaw,
                                 k_f_slope=0.1667,
                                 thrust_bias=-23.29,  # N
                                 k_m=6.80e-4,
                                 x=x_pos, y=y_pos, s=spin):
    """
    输出更贴近实际拟合模型的 PWM。
    """
    M = np.vstack((
        k_f_slope * np.ones(4),
        k_f_slope * y,
        k_f_slope * x,
        k_m * s
    ))

    # 修正总推力：减去偏置项再均分
    T_corrected = total_thrust_N - 4 * thrust_bias
    wrench = np.array([T_corrected, tau_roll, tau_pitch, tau_yaw])
    pwm = np.linalg.solve(M, wrench)
    pwm = np.clip(pwm, 110, 200)  # 限制 PWM 在 110 到 200 之间
    return pwm




host_ip =  '192.168.0.172' #IoT
# host_ip =  '192.168.0.235' #Hm-PC
host_port = 5605
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5606))   # 监听本机 5005 端口


def recv_latest(sock, fmt="ffffffi", bufsize=1024):
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

    def _init_groups(self):
        # 抽出来，便于重连后复用
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION
        )
        self.groupSyncRead = GroupSyncRead(
            self.portHandler, self.packetHandler, self.ADDR_PRESENT_VELOCITY, 8
        )
        for dxl_id in self.DXL_IDS:
            self.groupSyncRead.addParam(dxl_id)
        self.current_mode_map = {dxl_id: None for dxl_id in self.DXL_IDS}
        self.pwmgroupWrite = GroupSyncWrite(self.portHandler, self.packetHandler, 100, 2)

    def _reconnect(self, retry_delay=1.0, max_try=10):
        """关闭端口并重连，返回是否成功"""
        try:
            self.portHandler.closePort()
        except Exception:
            pass
        for k in range(max_try):
            try:
                if self.portHandler.openPort() and self.portHandler.setBaudRate(self.BAUDRATE):
                    self._init_groups()
                    print(f"✅ 串口重连成功（第{ k+1 }次）")
                    return True
            except Exception as e:
                print("重连异常：", e)
            time.sleep(retry_delay)
        print("❌ 串口重连失败")
        return False


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
            if self._reconnect():
                return self.get_joint_state()   # 重试一次
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

miss_count = 0  # 连续未收到消息的次数

while True:
    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1) as ser:
        start_time = time.time()
        i += 1
        start_time1 = time.time()
        qpos, qvel = real_controller.get_joint_state()
        # print(f'get_joint_state time:{time.time()-start_time1}') #~10ms

        start_time1 = time.time()
        sock.sendto(struct.pack("ffffi", qpos[0], qpos[1], qvel[0], qvel[1], i), (host_ip, host_port))

        values = recv_latest(sock, "ffffffi")
        # print(f'network time:{time.time()-start_time1}')
        if values is not None:
            # 收到消息，正常控制
            start_time1 = time.time()
            real_controller.send_torque([values[0], values[1]])
            print(f'send_torque time:{time.time()-start_time1}')

            start_time1 = time.time()
            pwm_cmd = pwm_from_wrench_N_fullmodel(
                total_thrust_N=values[2],    # ≈ 600 g 悬停
                tau_roll=values[3],
                tau_pitch=-values[4],
                tau_yaw=-values[5]
            )
            print(f'compute time:{time.time()-start_time1}')

            start_time1 = time.time()
            send_pwm(ser, int(pwm_cmd[0]), int(pwm_cmd[1]), int(pwm_cmd[2]), int(pwm_cmd[3]))
            print(f'send_pwm time:{time.time()-start_time1}')

            miss_count = 0  # 重置未收到计数
            # print('success control ' + str(i))
            print(f'One loop time:{time.time()-start_time}')
        else:
            # 没收到消息
            miss_count += 1
            # print("No command for {miss_count} cycles, torque set to 0")
            if miss_count >= 50:
                real_controller.send_torque([0.0, 0.0])  # 停止两个电机
                send_pwm(ser, 110,110,110,110)
                if miss_count == 100:
                    print("No command for 50 cycles, torque set to 0")
                if miss_count % 5000 == 0:
                    print(f"Prevent Beep at {miss_count}")
                    send_pwm(ser, 127,127,127,127)
                    # time.sleep(0.01)
                    send_pwm(ser, 110,110,110,110)


