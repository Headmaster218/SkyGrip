#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import sys
import os

# ---------- 工具函数 ----------
def parse_param(fname):
    """
    从文件名中解析参数串：
    EE_Square_<param>.npz -> <param>（如 'M=15' 或 '80_5000'）
    """
    base = os.path.basename(fname)
    m = re.search(r"EE_Square_(.+)\.npz$", base)
    return m.group(1) if m else None

def load_runs(pattern):
    """
    读取并组织数据：
    返回 dict[label] -> (time, act, ref)，label 为参数串
    """
    files = sorted(glob.glob(pattern))
    runs = {}
    for f in files:
        param = parse_param(f)
        if not param:
            continue
        data = np.load(f)
        time = data["time"]
        act  = data["actual_pos"]
        ref  = data["ref_pos"]
        label = str(param)
        # 若重名，追加序号防止覆盖
        k = label
        idx = 1
        while k in runs:
            idx += 1
            k = f"{label}#{idx}"
        runs[k] = (time, act, ref)
    return runs

def pick_reference(runs):
    """
    若存在至少一个 run，取其 ref 作为全局 reference。
    """
    if not runs:
        return None
    first_label = next(iter(runs))
    _, _, ref = runs[first_label]
    return ref

def plot_group(runs, title="Group Plot"):
    if not runs:
        raise ValueError("没有匹配到任何 npz 文件，请检查文件名和路径。")

    ref_global = pick_reference(runs)
    first_label = next(iter(runs))
    t0, _, _ = runs[first_label]  # 用第一条的时间作为参考时间轴（绘参考分量时会裁剪）

    fig = plt.figure(figsize=(18, 5))

    # 1) 3D 轨迹
    ax3d = fig.add_subplot(141, projection="3d")
    for label, (t, act, ref) in runs.items():
        ax3d.plot(act[:,0], act[:,1], act[:,2], label=label)
    if ref_global is not None:
        ax3d.plot(ref_global[:,0], ref_global[:,1], ref_global[:,2], "k--", linewidth=2, label="Reference")
    ax3d.set_title("3D Trajectory"); ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.legend(); ax3d.grid(True)
    # 坐标范围如需固定可在此设置
    ax3d.set_xlim(-0.1, 1.1); ax3d.set_ylim(-0.1, 1.1); ax3d.set_zlim(1, 2)

    # 2) XY 平面
    ax_xy = fig.add_subplot(142)
    for label, (t, act, ref) in runs.items():
        ax_xy.plot(act[:,0], act[:,1], label=label)
    if ref_global is not None:
        ax_xy.plot(ref_global[:,0], ref_global[:,1], "k--", linewidth=2, label="Reference")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("XY Plane"); ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
    ax_xy.legend(); ax_xy.grid(True)
    ax_xy.set_xlim(-0.1, 1.1); ax_xy.set_ylim(-0.1, 1.1)

    # 3) 误差 vs 时间
    ax_err = fig.add_subplot(143)
    for label, (t, act, ref) in runs.items():
        base_ref = ref_global if ref_global is not None else ref
        T = min(len(t), len(act), len(base_ref))
        errors = np.linalg.norm(act[:T] - base_ref[:T], axis=1)
        ax_err.plot(t[:T], errors, label=label)
    ax_err.axhline(0.05, ls="--", lw=0.8, label="5 cm")
    # ax_err.axhline(0.10, ls="--", lw=0.8, label="10 cm")
    ax_err.set_title("Tracking Error"); ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("Error [m]")
    ax_err.legend(); ax_err.grid(True)
    ax_err.set_ylim(-0.01, 0.06)

    # 4) 分量对比（实际）
    ax_pos = fig.add_subplot(144)
    for label, (t, act, ref) in runs.items():
        ax_pos.plot(t, act[:,0],  label=f"X {label}")
        ax_pos.plot(t, act[:,1],  label=f"Y {label}")
        ax_pos.plot(t, act[:,2],  label=f"Z {label}")
    if ref_global is not None:
        Tref = min(len(t0), len(ref_global))
        ax_pos.plot(t0[:Tref], ref_global[:Tref,0], "k--", label="X ref")
        ax_pos.plot(t0[:Tref], ref_global[:Tref,1], "k--", label="Y ref")
        ax_pos.plot(t0[:Tref], ref_global[:Tref,2], "k--", label="Z ref")
    ax_pos.set_title("Position Components"); ax_pos.set_xlabel("Time [s]"); ax_pos.set_ylabel("Pos [m]")
    ax_pos.legend(ncol=2, fontsize=8); ax_pos.grid(True)
    ax_pos.set_ylim(-0.1, 2.1)

    fig.suptitle(title, fontsize=18, y=0.98)
    plt.subplots_adjust(top=0.87, wspace=0.27, hspace=0.4, bottom=0.1, left=0, right=0.98)
    plt.show()

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 一次性读取 ./results 下所有 EE_Square_<param>.npz，并按 <param> 作为标签绘图
    runs_all = load_runs("EE_Square_*.npz")
    if not runs_all:
        print("未找到匹配文件：./results/EE_Square_*.npz。请确认路径。", file=sys.stderr)
    else:
        plot_group(runs_all, title="MPPI vs RS-MPC vs Myopic")
