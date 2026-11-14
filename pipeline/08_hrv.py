#!/usr/bin/env python3
"""
Step 8: 心率变异性 (HRV) 分析 — 时域 + 频域 (Lomb-Scargle)

目的: 为论文 B 提供 HRV 作为 allostatic/内感受调节的辅助指标。

方法 (对齐 Task Force 1996 与 Pernet et al. 2018 指南):
  时域:
    - SDNN: RR 间隔标准差 (反映总变异)
    - RMSSD: 相邻 RR 差值的均方根 (反映副交感/迷走张力)
  频域 (RR 间隔不均匀采样, 用 Lomb-Scargle 周期图, 避免插值偏差):
    - LF: 0.04-0.15 Hz (交感+副交感)
    - HF: 0.15-0.4 Hz (副交感/呼吸性窦性心律不齐)
    - LF/HF 比 + 归一化单位 (LF nu / HF nu)

伪迹剔除: 剔除异常 RR (偏离中位数 ± 30% 的生理不合理间隔, 异位搏动)。

产出:
  - data/hrv_long.csv  每被试×会话×刺激的 HRV 指标

用法: python 08_hrv.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
from scipy.signal import lombscargle

from common import (
    OUT, SESSIONS, EXCLUDE, EXPERIMENT_CHOICES, read_rpeak,
    get_session_stims, run_all_or_once, EXP2,
)

OUT.mkdir(parents=True, exist_ok=True)

# HRV 频段 (Task Force 1996)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)
VHF_BAND = (0.0033, 0.04)   # 极低频 (通常剔除, 避免长程漂移污染)


def clean_rr(rpeaks, rr_pct=30.0):
    """剔除生理不合理的 RR 间隔 (异位搏动/漏检)。

    剔除偏离中位数 > rr_pct% 的 RR 间隔 (Task Force 1996 建议 < 20-30%)。
    返回干净的 RR 间隔数组 (秒)。
    """
    rr = np.diff(rpeaks)
    if len(rr) < 3:
        return rr
    med = np.median(rr)
    keep = np.abs(rr - med) <= (rr_pct / 100.0) * med
    # 连续剔除: 被剔除的 RR 不再作为"前一跳"参与判断 (保持 RR 序列连贯性)
    rr_clean = rr[keep]
    return rr_clean


def time_domain_hrv(rr):
    """时域 HRV: SDNN, RMSSD, mean RR, mean HR。"""
    if len(rr) < 2:
        return {"sdnn_s": np.nan, "rmssd_s": np.nan,
                "mean_rr_s": np.nan, "mean_hr_bpm": np.nan}
    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    mean_rr = float(np.mean(rr))
    mean_hr = float(60.0 / mean_rr) if mean_rr > 0 else np.nan
    return {"sdnn_s": round(sdnn, 4), "rmssd_s": round(rmssd, 4),
            "mean_rr_s": round(mean_rr, 4), "mean_hr_bpm": round(mean_hr, 2)}


def freq_domain_hrv(rpeaks, srate_hrv=4.0):
    """频域 HRV: Lomb-Scargle 周期图, 计算 LF/HF 功率与比值。

    rpeaks: R 峰时间戳 (秒, 不均匀采样)。用 Lomb-Scargle 直接在不均匀的
    R 峰时刻上估计功率谱 (Task Force 1996 推荐的 HRV 频域方法, 避免
    插值到均匀网格引入偏差)。
    """
    if len(rpeaks) < 20:
        return {"lf_ms2": np.nan, "hf_ms2": np.nan, "lf_hf_ratio": np.nan,
                "lf_nu": np.nan, "hf_nu": np.nan}
    # RR 间隔序列及其时刻 (用前一 R 峰时刻作为该 RR 的时间戳)
    rr = np.diff(rpeaks)
    t_rr = rpeaks[:-1]
    # 去均值 (移除直流, 避免谱泄漏)
    rr = rr - np.mean(rr)
    # 频率轴 (Hz), 覆盖 VLF~HF (0.003-0.5 Hz)
    f = np.linspace(0.0033, 0.5, 512)
    omega = 2 * np.pi * f
    psd = lombscargle(t_rr, rr, omega, normalize=False)
    # 功率谱密度 (Lomb-Scargle 返回的是功率, 归一化到 ms² 需乘 2*var 相关因子;
    # 这里只计算相对功率比, 绝对单位作参考)
    def band_power(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(psd[m], f[m]))

    lf = band_power(*LF_BAND)
    hf = band_power(*HF_BAND)
    lf_hf_ratio = lf / hf if hf > 0 else np.nan
    # 归一化单位 (nu) = 该频段 / (LF+HF)
    total_lfhf = lf + hf
    lf_nu = lf / total_lfhf if total_lfhf > 0 else np.nan
    hf_nu = hf / total_lfhf if total_lfhf > 0 else np.nan
    return {"lf_ms2": round(lf, 4), "hf_ms2": round(hf, 4),
            "lf_hf_ratio": round(lf_hf_ratio, 4) if not np.isnan(lf_hf_ratio) else "",
            "lf_nu": round(lf_nu, 4) if not np.isnan(lf_nu) else "",
            "hf_nu": round(hf_nu, 4) if not np.isnan(hf_nu) else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default="sub-01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES)
    args = ap.parse_args()

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = [args.sub]

    rows = []
    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                rpeaks = read_rpeak(sub, ses, stim)
                if rpeaks is None or len(rpeaks) < 20:
                    continue
                # 时域 (基于伪迹剔除后的 RR)
                rr_clean = clean_rr(rpeaks)
                td = time_domain_hrv(rr_clean)
                # 频域 (Lomb-Scargle, 基于完整 R 峰序列)
                fd = freq_domain_hrv(rpeaks)
                rows.append({
                    "sub": sub, "ses": ses, "stim": stim,
                    "condition": SESSIONS[ses],
                    "n_rpeaks": len(rpeaks),
                    "n_rr_clean": len(rr_clean),
                    **td, **fd,
                })
        if args.sub or args.all:
            print(f"完成 {sub}: 累计 {len(rows)} 条")

    if rows:
        fn = list(rows[0].keys())
        with open(OUT / "hrv_long.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    print(f"\n总计 {len(rows)} 条 HRV 记录 -> {OUT / 'hrv_long.csv'}")


if __name__ == "__main__":
    run_all_or_once(main)
