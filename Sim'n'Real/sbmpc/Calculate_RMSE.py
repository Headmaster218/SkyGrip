#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, re, sys, os
import numpy as np
import pandas as pd

# ========= 配置 =========
PATTERN   = "EE_Square_*.npz"
CSV_OUT   = "EE_Square_rmse_compare.csv"
SEARCH_DIR = "./"

# ========= 工具函数 =========
def parse_name(fname):
    """解析文件名中的参数部分，例如 EE_Square_M=15.npz -> 'M=15'"""
    m = re.search(r"EE_Square_(.+)\.npz$", os.path.basename(fname))
    return m.group(1) if m else None

def compute_rmse(act, ref):
    T = min(len(act), len(ref))
    if T == 0: return np.nan
    e = act[:T] - ref[:T]
    return np.sqrt(np.mean(np.sum(e**2, axis=1)))

# ========= 1. 读取并算 RMSE =========
rows = []
files = sorted(glob.glob(os.path.join(SEARCH_DIR, PATTERN)))
if not files:
    sys.exit(f"❌ 未找到文件 {PATTERN} in {SEARCH_DIR}")

for f in files:
    param = parse_name(f)
    if param is None: 
        continue
    data = np.load(f)
    rmse = compute_rmse(data["actual_pos"], data["ref_pos"])
    rows.append({"param": param, "rmse": float(rmse)})

df = pd.DataFrame(rows).sort_values("param").reset_index(drop=True)
print(df)
df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"✅ RMSE 表已保存: {CSV_OUT}")
