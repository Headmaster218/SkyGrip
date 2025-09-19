import time, struct, socket, collections
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

# ---------------- 用户可配置 ----------------
UDP_PORT      = 5604          # 上位机→本机，drone_cmd
SERIAL_PORT   = '/dev/ttyACM0'
BAUDRATE      = 1_000_000
LOOP_PERIOD   = 0.005         # 200 Hz
LP_FC         = 8             # 一阶 IIR 截止频率 (Hz)
FAIL_TIMEOUT  = 0.10          # 无包 >0.10 s→failsafe
BUF_LEN       = 5             # 历史帧深度
# ------------------------------------------------

@dataclass
class Cmd:
    thrust: float
    roll  : float
    pitch : float
    yaw   : float
    ts    : float         # 秒级时间戳

# ---------- 占位函数：按需替换 ----------

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

def init_serial(port, baud):
    import serial
    return serial.Serial(port, baud, timeout=0)

# === 串口通信函数 ===
def send_pwm(ser, t, r, p, y):
    # 注意：STM32 是大端还是小端？默认我们用大端（如果你用的是 `>HHHH`）
    packet = struct.pack('>HHHH', t, r, p, y)
    ser.write(packet)
# ----------------------------------------

def LowPass(fc, dt):
    a, y = np.exp(-2*np.pi*fc*dt), None
    def filt(x):
        nonlocal y
        y = x if y is None else a*y + (1-a)*x
        return y
    return filt

def predict(buf, t_now):
    n = len(buf)
    if n == 0 or (t_now - buf[-1].ts) > FAIL_TIMEOUT:
        return 0., 0., 0., 0.
    if n == 1:
        c = buf[-1]
        return c.thrust, c.roll, c.pitch, c.yaw

    t  = np.array([c.ts for c in buf])
    vec = lambda k: np.array([getattr(c, k) for c in buf])
    def interp(arr):
        if n >= 3:
            return CubicSpline(t, arr, extrapolate=True)(t_now)
        return np.interp(t_now, t, arr)
    return (interp(vec('thrust')),
            interp(vec('roll')),
            interp(vec('pitch')),
            interp(vec('yaw')))



def main():
    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.setblocking(False)

    # Serial
    ser = init_serial(SERIAL_PORT, BAUDRATE)

    # 缓冲区
    buf = collections.deque(maxlen=BUF_LEN)
    lp  = LowPass(LP_FC, LOOP_PERIOD)

    fmt = 'ffffi'
    pkt_sz = struct.calcsize(fmt)

    # ===== 运行前放在主循环外 =====
    first_packet   = True          # 用于一次性计算时钟偏移
    time_offset    = 0.0           # 发送端 ↔︎ 本地时钟差 (sec)
    last_cmd_ts    = 0.0           # 最近一帧“本地接收时刻”
    active         = False         # 是否进入激活控制
    FAIL_TIMEOUT   = 0.20          # 200 ms 无包 → 失活
    SAFE_PWM       = np.full(4, 110, dtype=np.uint16)  # 失活输出
    # ===============================================


    print("Plane controller started (200 Hz loop)")
    while True:
        loop_t0 = time.perf_counter()

        # ≡ 收 UDP 指令 (非阻塞) ----------------
        try:
            data, _ = sock.recvfrom(128)
            if len(data) >= pkt_sz:
                th, r, p, y, idx = struct.unpack(fmt, data[:pkt_sz])  # idx = 发送端时间戳(ms)
                
                # --- 时钟同步：首帧计算 offset ---
                if first_packet:
                    time_offset = loop_t0 - (idx / 1000.0)
                    first_packet = False
                    print(f"[INFO] time offset = {time_offset:.3f} s")

                # --- 修正时间戳后写入缓冲 ---
                ts_local = (idx / 1000.0) + time_offset      # 转换到本地时钟域
                buf.append(Cmd(th, r, p, y, ts_local))

                # --- 更新激活状态 ---
                last_cmd_ts = loop_t0
                active = True
        except BlockingIOError:
            pass  # UDP 缓冲区为空

        # ≡ 检查是否超时失活 ----------------------
        if active and (loop_t0 - last_cmd_ts) > FAIL_TIMEOUT:
            print("[WARN] no control packet for 200 ms → failsafe")
            active = False
            first_packet = True  # 重新同步时钟
            buf.clear()                 # 清空旧轨迹，等待下一次激活

        # ≡ 三次样条 + 低通 ----------------------
        if active:
            # thrust, roll, pitch, yaw = predict(buf, loop_t0)
            thrust, roll, pitch, yaw = lp(np.array([thrust, roll, pitch, yaw]))
        else:
            thrust = roll = pitch = yaw = 0.0

        # ≡ 计算 PWM 并发送 ----------------------
        if active:
            pwm = pwm_from_wrench_N_fullmodel(
                total_thrust_N = thrust,
                tau_roll  = roll,
                tau_pitch = -pitch,
                tau_yaw   = -yaw
            ).astype(np.uint16)
        else:
            pwm = SAFE_PWM                 # 失活时恒定安全 PWM

        send_pwm(ser, pwm[0], pwm[1], pwm[2], pwm[3])

        # ≡ 维持 200 Hz --------------------------
        time.sleep(max(0.0, LOOP_PERIOD - (time.perf_counter() - loop_t0)))

if __name__ == '__main__':
    main()
