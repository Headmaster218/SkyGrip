import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import sys

# ---------- 工具函数 ----------
def parse_name(fname):
    """
    解析 EE_Square_{h}_{s}.npz -> (h, s)
    """
    m = re.search(r"EE_Square_(\d+)_(\d+)\.npz$", fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))  # (h, s)

def load_runs(pattern, label_mode="samples"):
    """
    读取并组织数据
    pattern: 例如 "EE_Square_80_*.npz" 或 "EE_Square_*_5000.npz"
    label_mode: "samples" 时用 s 作为标签；"horizon" 时用 h 作为标签
    return: dict[label] -> (time, act, ref)
    """
    files = sorted(glob.glob(pattern))
    runs = {}
    for f in files:
        h, s = parse_name(f)
        if h is None:
            continue
        data = np.load(f)
        time = data["time"]
        act = data["actual_pos"]
        ref = data["ref_pos"]
        if label_mode == "samples":
            label = f"s={s}"
        else:
            label = f"h={h}"
        runs[label] = (time, act, ref)
    return runs

def pick_reference(runs):
    """
    若存在至少一个 run，取其 ref 作为全局 reference。
    如果需要，也可以检测不同 run 的 ref 是否一致。
    """
    if not runs:
        return None
    # 取第一个
    first_label = next(iter(runs))
    _, _, ref = runs[first_label]
    # 可选一致性检查（如需严格可打开）
    # for lb, (_, _, r) in runs.items():
    #     if r.shape != ref.shape or np.max(np.abs(r - ref)) > 1e-9:
    #         return None  # 表示不一致，调用方自行处理
    return ref

def plot_group(runs, title="Group Plot"):
    if not runs:
        raise ValueError("没有匹配到任何 npz 文件，请检查文件名和路径。")

    # 取全局参考（假设一致）；如返回 None，可在每条曲线内分别画各自 ref
    ref_global = pick_reference(runs)

    fig = plt.figure(figsize=(18, 5))

    # 1) 3D 轨迹
    ax3d = fig.add_subplot(141, projection="3d")
    for label, (t, act, ref) in runs.items():
        ax3d.plot(act[:,0], act[:,1], act[:,2], label=label)
    if ref_global is not None:
        ax3d.plot(ref_global[:,0], ref_global[:,1], ref_global[:,2], "k--", linewidth=2, label="Reference")
    ax3d.set_title("3D Trajectory"); ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.legend(); ax3d.grid(True)

    # 2) XY 平面
    ax_xy = fig.add_subplot(142)
    for label, (t, act, ref) in runs.items():
        ax_xy.plot(act[:,0], act[:,1], label=label)
    if ref_global is not None:
        ax_xy.plot(ref_global[:,0], ref_global[:,1], "k--", linewidth=2, label="Reference")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("XY Plane"); ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
    ax_xy.legend(); ax_xy.grid(True)

    # 3) 误差 vs 时间
    ax_err = fig.add_subplot(143)
    for label, (t, act, ref) in runs.items():
        # 若 ref_global 存在，则统一对齐它；否则用各自 ref
        base_ref = ref_global if ref_global is not None else ref
        # 安全裁剪到相同长度
        T = min(len(t), len(act), len(base_ref))
        errors = np.linalg.norm(act[:T] - base_ref[:T], axis=1)
        ax_err.plot(t[:T], errors, label=label)
    ax_err.axhline(0.05, ls="--", lw=0.8, label="5 cm")
    ax_err.axhline(0.10, ls="--", lw=0.8, label="10 cm")
    ax_err.set_title("Tracking Error"); ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("Error [m]")
    ax_err.legend(); ax_err.grid(True)

    # 4) 分量对比（实际）
    ax_pos = fig.add_subplot(144)
    # 只画实际分量，避免图例过多；参考分量用虚线三条即可
    for label, (t, act, ref) in runs.items():
        ax_pos.plot(t, act[:,0],  label=f"X {label}")
        ax_pos.plot(t, act[:,1],  label=f"Y {label}")
        ax_pos.plot(t, act[:,2],  label=f"Z {label}")
    if ref_global is not None:
        ax_pos.plot(t, ref_global[:,0], "k--", label="X ref")
        ax_pos.plot(t, ref_global[:,1], "k--", label="Y ref")
        ax_pos.plot(t, ref_global[:,2], "k--", label="Z ref")
    ax_pos.set_title("Position Components"); ax_pos.set_xlabel("Time [s]"); ax_pos.set_ylabel("Pos [m]")
    ax_pos.legend(ncol=2, fontsize=8); ax_pos.grid(True)

    fig.suptitle("Trajectory Tracking Results", fontsize=18, y=0.98)
    plt.subplots_adjust(top=0.87, wspace=0.27, hspace=0.4, bottom = 0.1, left= 0, right= 0.98)

    plt.show()

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 组1：固定 h = 80，不同 s
    runs_samples = load_runs("./results/EE_Square_80_*.npz", label_mode="samples")
    if not runs_samples:
        print("未找到匹配文件：EE_Square_80_*.npz。请确认文件是否在当前目录，或修改路径。", file=sys.stderr)
    else:
        plot_group(runs_samples, title="EE_Square – Horizon=80, varying Samples")

    # 组2：固定 s = 5000，不同 h
    runs_horizon = load_runs("./results/EE_Square_*_5000.npz", label_mode="horizon")
    if not runs_horizon:
        print("未找到匹配文件：EE_Square_*_5000.npz。请确认文件是否在当前目录，或修改路径。", file=sys.stderr)
    else:
        plot_group(runs_horizon, title="EE_Square – Samples=5000, varying Horizon")
