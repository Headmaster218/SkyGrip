# rigid_bridge.py
import rclpy
from rclpy.node import Node
from phasespace_msgs.msg import Rigid

import socket, json, math
from collections import deque
import numpy as np

# ========= 配置 =========
UDP_IP = "127.0.0.1"     # 跨机转发就改成对端 IP
UDP_PORT = 5005
WINDOW_LEN = 10          # 滑动窗口长度：10 包 ≈ 10ms（在 1kHz 下）
TIME_SCALE = 1e-6        # Rigid.time 单位→秒的比例（μs→s）
EPS = 1e-12

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --------- 四元数工具 ----------
def q_conj(q):  # [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def q_mul(q1, q2):
    # Hamilton product
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_rel(q_from, q_to):
    # 相对旋转 q_rel = q_from^{-1} ⊗ q_to
    return q_mul(q_conj(q_from), q_to)

def quat_to_rotvec(q_rel):
    # 把相对四元数转成旋转向量 θ*axis（3D）
    # q = [w, x, y, z]，angle = 2*atan2(|v|, w), axis = v/|v|
    w, vx, vy, vz = q_rel
    v = np.array([vx, vy, vz], dtype=float)
    v_norm = np.linalg.norm(v)
    # 处理数值边界
    if v_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(v_norm, max(-1.0, min(1.0, w)))
    axis = v / v_norm
    return angle * axis  # 旋转向量

# --------- 速度估计 ----------
def lin_vel_via_regression(times, positions):
    """
    对每个坐标分量做线性回归，斜率即速度。
    times: (N,) 秒
    positions: (N,3)
    """
    N = len(times)
    if N < 2:
        return np.zeros(3, dtype=float)
    t = np.asarray(times, dtype=float)
    p = np.asarray(positions, dtype=float)  # N x 3
    t_center = t.mean()
    dt = t - t_center
    denom = float(np.sum(dt*dt))
    if denom < EPS:
        return np.zeros(3, dtype=float)
    # 对每个维度求 slope
    # slope = sum( dt * p ) / sum( dt^2 )
    slope = (dt[:, None] * p).sum(axis=0) / denom
    return slope

def ang_vel_via_average(times, quats):
    """
    利用窗口内相邻样本的相对四元数，求每段平均角速度，再对窗口加权平均。
    times: (N,) 秒
    quats: (N,4) [w,x,y,z]（已做号一致性处理）
    """
    N = len(times)
    if N < 2:
        return np.zeros(3, dtype=float)
    omega_sum = np.zeros(3, dtype=float)
    weight_sum = 0.0
    for k in range(1, N):
        dt = times[k] - times[k-1]
        if dt <= 1e-6:
            continue
        q_rel = quat_rel(quats[k-1], quats[k])
        # 规范化，避免数值漂移
        q_rel = q_rel / max(EPS, np.linalg.norm(q_rel))
        rotvec = quat_to_rotvec(q_rel)  # θ * axis
        omega_k = rotvec / dt           # 平均角速度
        # 以时间段长度作为权重（也可以统一平均）
        omega_sum += omega_k * dt
        weight_sum += dt
    if weight_sum < EPS:
        return np.zeros(3, dtype=float)
    return omega_sum / weight_sum

class RigidBridge(Node):
    def __init__(self):
        super().__init__('rigid_bridge')

        # 滑动窗口缓存：每项是 dict(t: 秒, pos: (3,), quat: (4,))
        self.buf = deque(maxlen=WINDOW_LEN)

        self.sub = self.create_subscription(
            Rigid,
            '/phasespace_body_skygrip',
            self.cb,
            50  # QoS 队列
        )

    def cb(self, msg: Rigid):
        # 时间戳（转秒）
        t = float(msg.time) * TIME_SCALE

        # 位置、四元数（按 [w,x,y,z] 存；并做符号一致性处理，避免跳变）
        pos = np.array([msg.x, msg.y, msg.z], dtype=float)
        quat = np.array([msg.qw, msg.qx, msg.qy, msg.qz], dtype=float)

        # 与上一帧保持同向（q 与 -q 表示同一姿态，但数值不连续）
        if self.buf:
            if np.dot(quat, self.buf[-1]["quat"]) < 0.0:
                quat = -quat

        self.buf.append({"t": t, "pos": pos, "quat": quat})

        # 线速度（回归斜率）
        times = [e["t"] for e in self.buf]
        positions = [e["pos"] for e in self.buf]
        lin_vel = lin_vel_via_regression(times, positions)

        # 角速度（相邻段平均 + 加权平均）
        quats = [e["quat"] for e in self.buf]
        ang_vel = ang_vel_via_average(times, quats)

        # 最新一帧（用于输出 pos/quat）
        last = self.buf[-1]

        data = {
            "time": int(msg.time),                    # 原始时间戳（未缩放）
            "pos":  last["pos"].tolist(),            # [x, y, z]
            "quat": last["quat"].tolist(),           # [w, x, y, z]
            "lin_vel": lin_vel.tolist(),             # [vx, vy, vz] (m/s)
            "ang_vel": ang_vel.tolist()              # [wx, wy, wz] (rad/s)
        }

        sock.sendto(json.dumps(data).encode('utf-8'), (UDP_IP, UDP_PORT))

def main():
    rclpy.init()
    node = RigidBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
