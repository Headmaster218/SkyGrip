#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 配置 =========
PATTERN   = "EE_Square_*.npz"
CSV_OUT   = "EE_Square_rmse.csv"
FIG_OUT   = "EE_Square_rmse_heatmap.png"
GRID_SIZE = 100   # 拟合网格分辨率

# ========= 工具函数 =========
def parse_name(fname):
    m = re.search(r"EE_Square_(\d+)_(\d+)\.npz$", fname)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def compute_rmse(act, ref):
    T = min(len(act), len(ref))
    if T == 0: return np.nan
    e = act[:T] - ref[:T]
    return np.sqrt(np.mean(np.sum(e**2, axis=1)))

def fit_surface_poly2(H, S, Z):
    """二次多项式拟合"""
    X = np.column_stack([np.ones_like(H), H, S, H**2, S**2, H*S])
    coef, *_ = np.linalg.lstsq(X, Z, rcond=None)
    def predict(Hg, Sg):
        Xg = np.stack([np.ones_like(Hg), Hg, Sg, Hg**2, Sg**2, Hg*Sg], axis=-1)
        return np.tensordot(Xg, coef, axes=([-1],[0]))
    return predict


SEARCH_DIR = "./results"

import os
# ========= 1. 读取并算 RMSE =========
rows = []
files = sorted(glob.glob(os.path.join(SEARCH_DIR, PATTERN)))
if not files:
    sys.exit(f"❌ 未找到文件 {PATTERN}")

for f in files:
    h, s = parse_name(f)
    if h is None: continue
    data = np.load(f)
    rmse = compute_rmse(data["actual_pos"], data["ref_pos"])
    rows.append({"h": h, "s": s, "rmse": float(rmse)})

df = pd.DataFrame(rows).sort_values(["h","s"]).reset_index(drop=True)
print(df)
df.to_csv(CSV_OUT, index=False)
print(f"✅ RMSE 表已保存: {CSV_OUT}")

# ========= 2. 拟合并生成热力图 =========
H, S, Z = df["h"].to_numpy(), df["s"].to_numpy(), df["rmse"].to_numpy()
hmin, hmax = H.min(), H.max()
smin, smax = S.min(), S.max()
Hg, Sg = np.meshgrid(np.linspace(hmin,hmax,GRID_SIZE), np.linspace(smin,smax,GRID_SIZE))

predict = fit_surface_poly2(H,S,Z)
Zg = predict(Hg,Sg)

# ========= 3. 画二维彩色渐变图 =========
fig, ax = plt.subplots(figsize=(8,6))
cmap = ax.pcolormesh(Hg, Sg, Zg, shading="auto", cmap="viridis")
sc = ax.scatter(H, S, c=Z, cmap="viridis", edgecolor="k", s=40)  # 原始点
fig.colorbar(cmap, ax=ax, label="RMSE [m]")

ax.set_xlabel("Horizon (h)")
ax.set_ylabel("Samples (s)")
ax.set_title("RMSE Heatmap over (h, s)")

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
print(f"✅ 热力图已保存: {FIG_OUT}")
plt.show()
