#!/usr/bin/env python3
"""BBBD epoch 振幅拒绝阈值的绝对效应量 (补 Result 4 的逻辑漏洞)。

背景:
  Result 4 报告 epoch_reject_uv 是实验 3–5 中效应量波动的最大方差来源 (η² 份额),
  但正文只给了"相对份额", 没有像 CFA 阈值 (图 3a) 那样报告"绝对效应量增量"。
  审稿人可合理质疑: 既然 epoch 阈值是比 CFA 阈值更大的决策自由度, 为什么只对
  CFA 阈值做了 TOST 等效性、却放过 epoch 阈值? 这是选择性报告的风险点。

本脚本补上这个缺口, 做法与正文图 3 (CFA 阈值) 对称:
  对每个实验, 在 3 个 epoch 拒绝阈值 {50, 75, 100} μV 下重算条件效应量
  (Hedges' dz), 计算 epoch 阈值因子在 clean 轨迹内解释的 generalized η²
  (被试为随机效应的重复测量方差分解), 并做以 η² = 0.01 为 SESOI 的 TOST
  等效性检验, 报告 bootstrap 95% CI 上界。

判定规则 (与 CFA 阈值一致):
  - 若 η² 点估计 < 1% 且 CI 上界 < SESOI → "等效于可忽略"
  - 若点估计 < 1% 但 CI 上界跨越 SESOI → "点估计小但不精确"
  - 若点估计 ≥ 1% → "不可忽略 (需在正文与 CFA 阈值并列报告)"

只读 multiverse_subject_effects.csv (每 pipeline × 被试的 diff_uv), 不重跑预处理。

产出:
  - data/experiment{X}/epoch_reject_abs_effect.csv   逐实验的 η² + CI + TOST
  - data/epoch_reject_abs_effect_summary.csv         四实验汇总
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE, _EXPERIMENTS

EPOCH_UVS = [50.0, 75.0, 100.0]
SESOI = 0.01          # 与 CFA 阈值 TOST 的 η² SESOI 对齐


def generalized_eta2(values_by_level):
    """单因素重复测量 (被试内因子 epoch_reject_uv) 的 generalized η²。

    values_by_level: list of np.ndarray, 每个 level 一个数组 (同一批被试在
    3 个阈值下的条件效应 diff_uv)。返回 generalized η²。
    公式: η²_G = SS_effect / (SS_effect + SS_error)
    """
    levels = [np.asarray(v, dtype=float) for v in values_by_level]
    n_sub = len(levels[0])
    k = len(levels)
    if n_sub < 2 or any(len(v) != n_sub for v in levels):
        return np.nan
    grand = np.mean(np.concatenate(levels))
    # 被试均值 (跨 k 水平)
    sub_mean = np.mean(np.stack(levels, axis=1), axis=1)
    # 效应平方和 (每个水平: n_sub × (水平均值 − 总均值)²)
    ss_effect = 0.0
    for v in levels:
        ss_effect += n_sub * (np.mean(v) - grand) ** 2
    # 被试平方和 (每个被试: k × (被试均值 − 总均值)²)
    ss_subject = k * np.sum((sub_mean - grand) ** 2)
    # 总平方和
    ss_total = np.sum((np.concatenate(levels) - grand) ** 2)
    ss_error = ss_total - ss_effect - ss_subject
    # 数值保护: 误差应为非负, 极小负值归零
    ss_error = max(ss_error, 0.0)
    denom = ss_effect + ss_error
    if denom <= 0:
        return np.nan
    return ss_effect / denom


def bootstrap_eta2(levels, B=2000, seed=42):
    """按被试聚类 bootstrap 重估 generalized η², 返回 (ci_lo, ci_hi) percentile。"""
    rng = np.random.default_rng(seed)
    levels = [np.asarray(v, dtype=float) for v in levels]
    n = len(levels[0])
    ests = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ests[b] = generalized_eta2([v[idx] for v in levels])
    ests = ests[~np.isnan(ests)]
    if len(ests) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5)))


def load_subject_effects(exp):
    """读某实验的 multiverse_subject_effects.csv, 返回 {(sub, pipeline_id): diff_uv}。
    仅保留 clean 轨迹 (与 CFA 阈值的嵌套设计一致)。"""
    p = BASE / "data" / f"experiment{exp}" / "multiverse_subject_effects.csv"
    if not p.is_file():
        return None
    agg = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["track"] != "clean":
                continue
            try:
                dv = float(r["diff_uv"])
            except (TypeError, ValueError):
                continue
            agg[(r["sub"], int(r["pipeline_id"]))] = dv
    return agg


def load_params(exp):
    """读 multiverse_params.csv, 返回 {pipeline_id: epoch_reject_uv}。"""
    p = BASE / "data" / f"experiment{exp}" / "multiverse_params.csv"
    if not p.is_file():
        return None
    m = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            m[int(r["pipeline_id"])] = float(r["epoch_reject_uv"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default=None,
                    choices=list(_EXPERIMENTS.keys()) + ["all"])
    args = ap.parse_args()

    exps = list(_EXPERIMENTS.keys()) if args.experiment in (None, "all") \
        else [args.experiment]

    summary = []
    for exp in exps:
        agg = load_subject_effects(exp)
        params = load_params(exp)
        if agg is None or params is None:
            print(f"[实验 {exp}] 缺少 multiverse 数据, 跳过")
            continue
        # 找出全部被试 (sub), 每个被试在 3 个 epoch 阈值下都有 clean 效应
        subs = sorted({s for (s, _) in agg})
        # 按 epoch 阈值分组 pipeline_id
        pid_by_uv = {uv: [] for uv in EPOCH_UVS}
        for pid, uv in params.items():
            if uv in pid_by_uv:
                pid_by_uv[uv].append(pid)

        # 每个被试 × 每个阈值, 取该阈值下所有 pipeline 的 diff_uv 均值
        # (固定其他 6 维度平均, 隔离 epoch 阈值的边际效应)
        levels = {}
        for uv in EPOCH_UVS:
            vals = []
            for sub in subs:
                dvs = [agg[(sub, pid)] for pid in pid_by_uv[uv]
                       if (sub, pid) in agg]
                if dvs:
                    vals.append(np.mean(dvs))
            levels[uv] = np.array(vals, dtype=float)

        # 对齐被试集: 仅保留在 3 个阈值下都有数据的被试
        sub_sets = [set() for _ in EPOCH_UVS]
        for i, uv in enumerate(EPOCH_UVS):
            for sub in subs:
                if any((sub, pid) in agg for pid in pid_by_uv[uv]):
                    sub_sets[i].add(sub)
        common = sorted(set.intersection(*sub_sets))
        aligned = []
        for uv in EPOCH_UVS:
            vals = [np.mean([agg[(sub, pid)] for pid in pid_by_uv[uv]
                            if (sub, pid) in agg]) for sub in common]
            aligned.append(np.array(vals, dtype=float))

        if any(len(v) < 3 for v in aligned):
            print(f"[实验 {exp}] 有效被试不足, 跳过")
            continue

        eta2 = generalized_eta2(aligned)
        ci_lo, ci_hi = bootstrap_eta2(aligned)

        # 各阈值的效应量 (供描述)
        dz_by_uv = {}
        for i, uv in enumerate(EPOCH_UVS):
            v = aligned[i]
            dz_by_uv[uv] = round(float(np.mean(v) / np.std(v, ddof=1))
                                 if np.std(v, ddof=1) > 0 else np.nan, 4)

        # TOST 判定
        if not np.isnan(eta2):
            if eta2 < SESOI and ci_hi < SESOI:
                verdict = "等效于可忽略"
            elif eta2 < SESOI:
                verdict = "点估计小但不精确"
            else:
                verdict = "不可忽略"
        else:
            verdict = "nan"

        out_row = {
            "experiment": exp,
            "n_subjects": len(aligned[0]),
            "eta2_gen": round(eta2, 5) if not np.isnan(eta2) else "",
            "eta2_ci_lo": round(ci_lo, 5) if not np.isnan(ci_lo) else "",
            "eta2_ci_hi": round(ci_hi, 5) if not np.isnan(ci_hi) else "",
            "sesoi": SESOI,
            "dz_50uv": dz_by_uv.get(50.0, ""),
            "dz_75uv": dz_by_uv.get(75.0, ""),
            "dz_100uv": dz_by_uv.get(100.0, ""),
            "verdict": verdict,
        }
        summary.append(out_row)

        out = BASE / "data" / f"experiment{exp}" / "epoch_reject_abs_effect.csv"
        fn = list(out_row.keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerow(out_row)
        print(f"[实验 {exp}] η²_G={out_row['eta2_gen']} "
              f"CI=[{out_row['eta2_ci_lo']}, {out_row['eta2_ci_hi']}] "
              f"→ {verdict}  (dz@50/75/100 = {dz_by_uv.get(50)}/{dz_by_uv.get(75)}/{dz_by_uv.get(100)})")

    if summary:
        sfn = list(summary[0].keys())
        s_out = BASE / "data" / "epoch_reject_abs_effect_summary.csv"
        with open(s_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sfn)
            w.writeheader()
            w.writerows(summary)
        print(f"\n→ 汇总已写入 {s_out}")


if __name__ == "__main__":
    main()
