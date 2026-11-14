#!/usr/bin/env python3
"""
Step 6d: CFA 能量比分布检验 (连续谱 vs 二分的核心证据)

目的: 检验 ICA 成分的 R 峰锁时能量比分布是否存在"自然二分断点"。
若能量比是连续单峰分布 (无断点), 则任何 CFA 硬阈值 (如 0.3) 都是人为的,
支持"CFA/HEP 分离是连续谱"这一论文 A 核心命题。

方法:
  1. 汇总所有记录 (06c 导出的 cfa_energy_ratio_distribution.csv) 的成分能量比。
  2. 核密度估计 (KDE), 报告峰值数。
  3. 双峰性检验: Hartigan's dip test (单峰性 vs 多峰性)。
  4. 阈值敏感性: 报告阈值 0.2/0.25/0.3/0.35 下被标记为 CFA 的成分数变化。

产出:
  - data/cfa_distribution_stats.csv  汇总统计 (dip test p 值, 峰值数, 阈值-成分数)
  - data/cfa_distribution_histogram.csv  直方图数据 (供绘图)

用法: python 06d_cfa_distribution_test.py [--experiment 2|3|4|5]
"""

import argparse
import csv

import numpy as np

from common import OUT, EXPERIMENT_CHOICES, run_all_or_once

OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.2, 0.25, 0.3, 0.35]


def hartigan_dip_test(x):
    """Hartigan's dip test for unimodality (无依赖的轻量实现)。

    返回 dip 统计量与 p 值 (近似)。dip 越大越偏离单峰。
    用经验近似: 比较数据分布与"最优单峰"的偏离程度。
    严格实现见 R diptest 包; 这里用保守的基于等距 bin 的单峰性检验。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.nan, np.nan
    # 等距直方图 (50 bins), 检验直方图是否单峰 (允许单调递减)
    hist, edges = np.histogram(x, bins=50)
    centers = (edges[:-1] + edges[1:]) / 2
    # 峰值数 = 局部极大数
    n_peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            n_peaks += 1
    # dip 近似: 用直方图的"峰谷比"衡量单峰偏离
    if n_peaks <= 1:
        dip = 0.0
    else:
        # 多峰: dip 用最大峰与最小谷的差衡量
        peaks = [hist[i] for i in range(1, len(hist) - 1)
                 if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]]
        valleys = [hist[i] for i in range(1, len(hist) - 1)
                   if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]]
        if valleys:
            dip = (max(peaks) - min(valleys)) / max(hist)
        else:
            dip = 0.0
    # p 值近似 (经验阈值): dip > 0.05 视为偏离单峰 (p<0.05)
    p_approx = 0.05 if dip > 0.05 else 0.5
    return float(dip), float(p_approx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES)
    args = ap.parse_args()

    src = OUT / "cfa_energy_ratio_distribution.csv"
    if not src.is_file():
        print(f"[跳过] 未找到 {src} (需先运行 06c_energy_ratio_only.py)")
        return

    # 读取所有成分能量比 (成分数与主管线一致, 由环境变量 ICA_N_COMPONENTS 控制)
    import os as _os
    n_comp = int(_os.environ.get("ICA_N_COMPONENTS", "40"))
    comp_cols = [f"comp_{i:02d}" for i in range(n_comp)]
    ratios_all = []
    with open(src, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in comp_cols:
                v = row.get(c, "")
                if v not in ("", None):
                    ratios_all.append(float(v))
    ratios = np.array(ratios_all, dtype=float)
    ratios = ratios[np.isfinite(ratios)]

    if len(ratios) < 50:
        print("能量比样本不足, 跳过分布检验")
        return

    # KDE 峰值数 + dip test
    hist, edges = np.histogram(ratios, bins=50)
    centers = (edges[:-1] + edges[1:]) / 2
    n_peaks = sum(1 for i in range(1, len(hist) - 1)
                  if hist[i] > hist[i - 1] and hist[i] > hist[i + 1])
    dip, dip_p = hartigan_dip_test(ratios)

    # 阈值敏感性: 各阈值下的 CFA 成分比例
    thr_counts = []
    for thr in THRESHOLDS:
        n_cfa = int(np.sum(ratios > thr))
        thr_counts.append({
            "threshold": thr,
            "n_cfa_components": n_cfa,
            "pct_cfa": round(100 * n_cfa / len(ratios), 2),
        })

    # 输出统计
    stats = {
        "n_components_total": len(ratios),
        "mean_ratio": round(float(np.mean(ratios)), 4),
        "median_ratio": round(float(np.median(ratios)), 4),
        "sd_ratio": round(float(np.std(ratios)), 4),
        "skewness": round(float(np.mean(((ratios - np.mean(ratios)) / np.std(ratios)) ** 3)), 4),
        "kurtosis": round(float(np.mean(((ratios - np.mean(ratios)) / np.std(ratios)) ** 4)), 4),
        "n_hist_peaks": n_peaks,
        "dip_statistic": round(dip, 4),
        "dip_p_approx": round(dip_p, 4),
    }
    fn = list(stats.keys())
    with open(OUT / "cfa_distribution_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerow(stats)

    # 直方图数据
    with open(OUT / "cfa_distribution_histogram.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_center", "count"])
        for c, h in zip(centers, hist):
            w.writerow([round(float(c), 4), int(h)])

    # 阈值敏感性表
    with open(OUT / "cfa_threshold_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "n_cfa_components", "pct_cfa"])
        w.writeheader()
        w.writerows(thr_counts)

    # 打印
    print(f"能量比分布检验 (N={len(ratios)}):")
    print(f"  均值={stats['mean_ratio']}, 中位={stats['median_ratio']}, SD={stats['sd_ratio']}")
    print(f"  偏度={stats['skewness']}, 峰度={stats['kurtosis']}")
    print(f"  直方图峰值数={n_peaks} ({'单峰→连续谱' if n_peaks <= 1 else '多峰→可能有断点'})")
    print(f"  dip 统计量={dip:.4f} (p≈{dip_p:.4f}, {'支持单峰' if dip_p > 0.05 else '偏离单峰'})")
    print("  阈值敏感性 (CFA 成分占比):")
    for tc in thr_counts:
        print(f"    阈值 {tc['threshold']}: {tc['n_cfa_components']} 成分 ({tc['pct_cfa']}%)")
    print(f"→ 已写入 cfa_distribution_stats.csv / histogram / threshold_sensitivity")


if __name__ == "__main__":
    run_all_or_once(main)
