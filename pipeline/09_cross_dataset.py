#!/usr/bin/env python3
"""
Step 9: 跨实验效应量汇总 (横向对比 exp2/3/4/5)

目的: 汇总各实验的主效应量 (LMM condition 系数 + 多宇宙 Hedges' dz),
      生成跨实验对比表, 供 fig5 绘图。

产出:
  - data/cross_dataset_effects.csv  各实验的主效应 (β, SE, p, dz)

用法: python 09_cross_dataset.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from common import BASE

# 所有实验的输出都在 pipeline/pipeline_data/data/<experimentX>/ 下 (由 common.py 的 OUT 定义)。
# 用 common.BASE (= pipeline/pipeline_data/) 统一解析, 避免脚本自身相对路径假设。
OUT = BASE / "data"

EXPERIMENTS = {
    "2": ("experiment2", "incidental learning", "attentive vs distracted"),
    "3": ("experiment3", "intentional learning", "attentive vs distracted"),
    "4": ("experiment4", "dual-attention", "attentive1 vs attentive2"),
    "5": ("experiment5", "monetary incentive", "attentive vs incentive"),
}

rows = []
for exp_id, (dirname, design, contrast) in EXPERIMENTS.items():
    lmm_path = OUT / dirname / "lmm_coefs.csv"
    mv_path = OUT / dirname / "multiverse_effects.csv"

    row = {
        "experiment": exp_id,
        "name": dirname,
        "design": design,
        "contrast": contrast,
        "lmm_beta": None,
        "lmm_se": None,
        "lmm_p": None,
        "lmm_ci_low": None,
        "lmm_ci_high": None,
        "n_subjects": None,
        "median_dz": None,
        "dz_min": None,
        "dz_max": None,
        "n_pipelines": None,
        "n_sig_pipelines": None,
    }

    # LMM 主效应 (condition 系数)
    if lmm_path.is_file():
        with open(lmm_path, encoding="utf-8") as f:
            coefs = list(csv.DictReader(f))
        cond_rows = [c for c in coefs if "condition" in c.get("term", "")]
        if cond_rows:
            c = cond_rows[0]
            row["lmm_beta"] = c.get("Estimate")
            row["lmm_se"] = c.get("Std. Error")
            row["lmm_p"] = c.get("Pr(>|t|)")
            # 95% CI (z 近似)
            try:
                b = float(c["Estimate"]); se = float(c["Std. Error"])
                row["lmm_ci_low"] = round(b - 1.96 * se, 4)
                row["lmm_ci_high"] = round(b + 1.96 * se, 4)
            except (ValueError, KeyError):
                row["lmm_ci_low"] = None
                row["lmm_ci_high"] = None

    # 多宇宙效应量分布
    if mv_path.is_file():
        with open(mv_path, encoding="utf-8") as f:
            effects = list(csv.DictReader(f))
        dzs = [float(e["hedges_dz"]) for e in effects
               if e.get("hedges_dz") not in (None, "", "None")]
        ps = [float(e["p_val"]) for e in effects
              if e.get("p_val") not in (None, "", "None")]
        if effects:
            row["n_subjects"] = effects[0].get("n_subjects")
        if dzs:
            row["median_dz"] = round(float(np.median(dzs)), 4)
            row["dz_min"] = round(float(np.min(dzs)), 4)
            row["dz_max"] = round(float(np.max(dzs)), 4)
        row["n_pipelines"] = len(effects)
        row["n_sig_pipelines"] = int(sum(1 for p in ps if p < 0.05))

    rows.append(row)

OUT.mkdir(parents=True, exist_ok=True)
fn = list(rows[0].keys())
with open(OUT / "cross_dataset_effects.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader()
    w.writerows(rows)

print("跨实验效应量汇总:")
for r in rows:
    print(f"  exp{r['experiment']} ({r['design']:20s}): "
          f"LMM β={r['lmm_beta']} (p={r['lmm_p']}), "
          f"median dz={r['median_dz']}, pipelines={r['n_pipelines']}")
print(f"\n→ 已写入 data/cross_dataset_effects.csv")
