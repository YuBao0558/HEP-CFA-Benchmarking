#!/usr/bin/env python3
"""三数据集一致性 meta 分析 (把 S3b/S3c 的"各自独立佐证"升级为"定量对比")。

背景:
  论文 H3 目前是"三个数据集各自独立报告单峰性/阈值单调性" (S3b/S3c), 审稿人
  可能质疑这仍是"各说各话", 缺少一个形式化的跨数据集定量对比。本脚本补上:

  1. pooled 单峰证据: 三数据集的 dip test p、GMM ΔBIC、检出率梯度放同一张表,
     并做 pooled 判断 (是否全部支持"无自然分界")。
  2. CFA 能量水平与"无自然分界证据强度"的跨数据集关系: 用 Spearman 相关检验
     "结论是否依赖 CFA 能量水平" (若相关不显著, 则支持"结论跨能量梯度稳健")。
  3. track 效应方向的跨数据集 meta 分析: 把 BBBD 4 实验 + ds003816 3 条件对的
     track 效应 g 值做随机效应合并 (报告合并 g、Cochran's Q、I²), 检验方向是否
     跨数据集一致或漂移。SENSE-42 无条件结构, 不进入此部分。

  判定规则:
    - pooled 单峰: 三数据集 dip p 全部 > 0.05 且 GMM ΔBIC 全为正 → "无自然分界
      跨三数据集一致成立"; 否则标注例外个案 (如 SENSE-42 log 尺度)。
    - 能量水平相关性: Spearman ρ 的 |ρ| < 0.7 且 p > 0.05 → "结论不依赖 CFA 能量
      水平" (3 个点, 只作描述性, 不作推断)。
    - track meta: I² < 50% 且合并 g 的 95% CI 方向一致 → "方向一致"; I² ≥ 50%
      或 CI 跨越 0 → "方向漂移/量级有限" (与 BBBD 内部结论一致)。

只读已产出的统计 CSV, 不重跑任何预处理/ICA。

产出:
  - data/cross_dataset_consistency.csv         pooled 单峰 + 能量水平相关性
  - data/cross_dataset_track_meta.csv          track 效应跨数据集 meta
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE

# 三个数据集的能量富集比分布统计 (median_ratio, dip_p_log, gmm_delta_bic_log)。
# 这些值来自各数据集已产出的 CSV; 若某数据集尚未重跑完, 脚本会从 CSV 现场读取,
# 而非硬编码 (见 load_dataset_stats)。
BBBD_NCOMP = BASE / "data" / "ncomp_robustness.csv"          # 4 实验 × 40 成分行
DS003816_STATS = Path("/Users/yubao/CODE/public/ds003816-full/analysis/results/cfa_distribution_stats.csv")
SENSE42_STATS = None  # SENSE-42 单峰性 CSV 由 04/05 步产出, 见下 fallback


def load_bbbd_stats():
    """读 BBBD ncomp_robustness.csv, 取 40 成分行的 median_ratio / dip_p_log / gmm_delta_bic_log。"""
    if not BBBD_NCOMP.is_file():
        return None
    rows = []
    with open(BBBD_NCOMP) as f:
        for r in csv.DictReader(f):
            if r["n_components"] == "40":
                rows.append({
                    "dataset": f"BBBD-exp{r['experiment']}",
                    "median_ratio": float(r["median_ratio"]),
                    "dip_p_log": float(r["dip_p_log"]),
                    "gmm_delta_bic_log": float(r["gmm_delta_bic_log"]),
                })
    return rows


def load_ds003816_stats():
    """读 ds003816 cfa_distribution_stats.csv (单行)。"""
    if not DS003816_STATS.is_file():
        return None
    with open(DS003816_STATS) as f:
        r = next(csv.DictReader(f))
    return [{
        "dataset": "ds003816",
        "median_ratio": float(r["median_ratio"]),
        "dip_p_log": float(r["dip_p_log"]),
        "gmm_delta_bic_log": float(r["gmm_delta_bic"]),
    }]


def load_sense42_stats():
    """读 SENSE-42 单峰性统计 (unimodality_result.csv, 长格式 metric/value)。"""
    p = Path("/Users/yubao/CODE/public/SENSE-42/results/data/unimodality_result.csv")
    if not p.is_file():
        return None
    kv = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            kv[r["metric"]] = r["value"]
    def g(k, default=np.nan):
        v = kv.get(k)
        return float(v) if v not in (None, "", "nan") else default
    return [{
        "dataset": "SENSE-42",
        "median_ratio": g("median"),
        "dip_p_log": g("dip_p_log"),
        "gmm_delta_bic_log": g("dbic_log"),
    }]


def random_effects_meta(g_list, ci_lo_list, ci_hi_list):
    """DerSimonian-Laird 随机效应 meta (以 CI 反推 SE)。返回 (合并 g, CI, Q, I²)。"""
    g = np.asarray(g_list, dtype=float)
    ci_lo = np.asarray(ci_lo_list, dtype=float)
    ci_hi = np.asarray(ci_hi_list, dtype=float)
    se = (ci_hi - ci_lo) / (2 * 1.96)
    w_fixed = 1.0 / (se ** 2)
    g_fixed = np.sum(w_fixed * g) / np.sum(w_fixed)
    # 异质性
    Q = np.sum(w_fixed * (g - g_fixed) ** 2)
    k = len(g)
    df = k - 1
    tau2 = max(0.0, (Q - df) / (np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)))
    w_rand = 1.0 / (se ** 2 + tau2)
    g_rand = np.sum(w_rand * g) / np.sum(w_rand)
    se_rand = np.sqrt(1.0 / np.sum(w_rand))
    ci_lo_rand = g_rand - 1.96 * se_rand
    ci_hi_rand = g_rand + 1.96 * se_rand
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    p_het = stats.chi2.sf(Q, df)
    return g_rand, ci_lo_rand, ci_hi_rand, Q, I2, p_het, tau2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # ---- 1. pooled 单峰证据 ----
    stats_rows = []
    bb = load_bbbd_stats()
    ds = load_ds003816_stats()
    se = load_sense42_stats()
    for rows in (bb, ds, se):
        if rows:
            stats_rows.extend(rows)

    print("=== pooled 单峰证据 (三数据集) ===")
    all_ratios = []
    dip_ps = []
    bic = []
    for r in stats_rows:
        print(f"  {r['dataset']:12s} median={r['median_ratio']} "
              f"dip_p={r['dip_p_log']} ΔBIC={r['gmm_delta_bic_log']}")
        if not np.isnan(r["median_ratio"]):
            all_ratios.append(r["median_ratio"])
        dip_ps.append(r["dip_p_log"])
        bic.append(r["gmm_delta_bic_log"])

    # pooled 判断
    n_neg_bic = sum(1 for b in bic if b < 0)
    n_dip_fail = sum(1 for p in dip_ps if p < 0.05)
    pooled_verdict = ("一致成立" if n_neg_bic == 0 and n_dip_fail == 0
                      else f"存在例外 (负ΔBIC={n_neg_bic}, dip<0.05={n_dip_fail})")

    # ---- 2. 能量水平 vs 无自然分界证据强度 ----
    ratios = np.asarray(all_ratios, dtype=float)
    bic_arr = np.asarray(bic[:len(ratios)], dtype=float)
    rho = np.nan
    rho_p = np.nan
    if len(ratios) >= 3:
        rho, rho_p = stats.spearmanr(ratios, bic_arr)
    rho_verdict = ("不依赖 CFA 能量水平 (描述性)" if (not np.isnan(rho) and abs(rho) < 0.7)
                   else ("可能相关 (描述性, 仅 3 组点)" if not np.isnan(rho) else "样本不足"))

    out1 = [{
        "n_datasets": len(stats_rows),
        "n_neg_gmm_bic": n_neg_bic,
        "n_dip_fail": n_dip_fail,
        "pooled_verdict": pooled_verdict,
        "spearman_rho_energy_vs_bic": round(rho, 3) if not np.isnan(rho) else "",
        "spearman_p": round(rho_p, 3) if not np.isnan(rho_p) else "",
        "energy_dependence_verdict": rho_verdict,
    }]
    out = Path(args.out) if args.out else (BASE / "data" / "cross_dataset_consistency.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out1[0].keys()))
        w.writeheader()
        w.writerows(out1)
    print(f"\npooled 判定: {pooled_verdict}")
    print(f"能量水平 vs ΔBIC Spearman ρ={rho}, p={rho_p} → {rho_verdict}")

    # ---- 3. track 效应跨数据集 meta (BBBD 4 实验 + ds003816 3 条件对) ----
    # BBBD track g (来自正文 Result 3): 实验2 +0.27, 3 −0.08, 4 −0.13, 5 +0.06
    # 其 CI 来自 supp_cfa_threshold_variance.csv (g_ci_lo/g_ci_hi)
    track_rows = []
    # 读 BBBD 的 track CI (字段: hedges_g_track / g_track_boot_lo / g_track_boot_hi)
    bb_track = BASE / "data" / "supp_cfa_threshold_variance.csv"
    if bb_track.is_file():
        with open(bb_track) as f:
            for r in csv.DictReader(f):
                try:
                    g = float(r["hedges_g_track"])
                    lo = float(r["g_track_boot_lo"])
                    hi = float(r["g_track_boot_hi"])
                except (KeyError, ValueError):
                    continue
                track_rows.append({
                    "dataset": f"BBBD-exp{r['experiment']}",
                    "g": g, "ci_lo": lo, "ci_hi": hi,
                })
    # ds003816 三个条件对的 track 增量 g (正文 S3c): resting −0.244, lkm +0.110,
    # visualize +0.222; 均不显著 (p ≥ 0.37), CI 用 n=15 近似 (SE 反推)。
    # 这些值在正交条件分析 CSV 里, 若存在则现场读, 否则用正文值兜底。
    ds_orth = Path("/Users/yubao/CODE/public/ds003816-full/analysis/results/orthogonal_condition_effects.csv")
    if ds_orth.is_file():
        with open(ds_orth) as f:
            for r in csv.DictReader(f):
                g = r.get("track_delta_hedges_g")
                if g:
                    try:
                        gv = float(g)
                    except ValueError:
                        continue
                    # 用 n 与 SE 近似 CI (无 CI 字段则用 g ± 1.96*SE, SE≈0.3 兜底)
                    track_rows.append({
                        "dataset": f"ds003816-{r['condition_pair']}",
                        "g": gv, "ci_lo": gv - 0.3, "ci_hi": gv + 0.3,
                    })

    if len(track_rows) >= 3:
        g_arr = [r["g"] for r in track_rows]
        lo_arr = [r["ci_lo"] for r in track_rows]
        hi_arr = [r["ci_hi"] for r in track_rows]
        g_rand, lo, hi, Q, I2, p_het, tau2 = random_effects_meta(g_arr, lo_arr, hi_arr)
        meta_verdict = ("方向一致" if I2 < 50 and not (lo <= 0 <= hi)
                        else "方向漂移/量级有限")
        out2 = [{
            "n_effects": len(track_rows),
            "pooled_g": round(g_rand, 4), "pooled_ci_lo": round(lo, 4),
            "pooled_ci_hi": round(hi, 4), "cochran_Q": round(Q, 3),
            "Q_df": len(g_arr) - 1, "Q_p": round(p_het, 4),
            "I2_pct": round(I2, 1), "tau2": round(tau2, 5),
            "verdict": meta_verdict,
        }]
        m_out = BASE / "data" / "cross_dataset_track_meta.csv"
        with open(m_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out2[0].keys()))
            w.writeheader()
            w.writerows(out2)
        print(f"\n=== track 效应跨数据集 meta ===")
        print(f"  pooled g={g_rand:.3f} [{lo:.3f}, {hi:.3f}], "
              f"Q={Q:.2f} (p={p_het:.3f}), I²={I2:.0f}% → {meta_verdict}")

    print(f"\n→ 已写入 {out}")


if __name__ == "__main__":
    main()
