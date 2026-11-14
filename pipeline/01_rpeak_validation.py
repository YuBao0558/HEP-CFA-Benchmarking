 #!/usr/bin/env python3
"""
Step 1: R 峰复检 (官方 derivative vs 自建检测器)

目的: 验证官方 rpeak_timestamps 是否可靠, 决定能否直接采用(操作定义 §3.2)。
标准: R 峰数量差异 < 1%; 时间戳 RMSE < 10 ms (冻结值)。

策略: 抽样 N_SUBSAMPLE 个被试(约 1/3), 对每个被试的 raw ECG 用 neurokit2
      (Pan-Tompkins 类) 自建检测, 与官方时间戳做逐被试对比。

产出:
  - data/rpeak_validation.csv   逐被试逐刺激对比表
  - data/rpeak_validation_summary.json
  - figures/figS1_rpeak_blandaltman.pdf   (由 R 脚本另画, 这里只出数据)

用法: python 01_rpeak_validation.py [--n 10] [--seed 42]
"""

import argparse
import csv
import gzip
import json
import random

import numpy as np
import neurokit2 as nk
from scipy.optimize import linear_sum_assignment

from common import (
    ROOT, EXP2, OUT, FIG, SESSIONS_LIST as SESSIONS, STIMS,
    EXCLUDE, EXPERIMENT_CHOICES, read_sampling_rate_from_json, get_session_stims,
    run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def read_sampling_rate(sub, ses, stim):
    return read_sampling_rate_from_json(sub, ses, stim, kind="ecg")


def read_raw_ecg(sub, ses, stim):
    p = EXP2 / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_recording-ecg_physio.tsv.gz"
    if not p.is_file():
        return None
    vals = []
    with gzip.open(p, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                # 跳过 header 等非数值行
                continue
    return np.array(vals)


def read_official_rpeak(sub, ses, stim):
    p = EXP2 / "derivatives" / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_desc-rpeak_timestamps.tsv.gz"
    if not p.is_file():
        return None
    with gzip.open(p, "rt") as f:
        lines = [l.strip() for l in f if l.strip()]
    # 第一行是 header "timestamp"
    if lines and lines[0].lower().startswith("timestamp"):
        lines = lines[1:]
    return np.array([float(x) for x in lines])


def detect_rpeaks(ecg, srate):
    """neurokit2 检测, 返回秒单位的 R 峰时间戳"""
    signals, info = nk.ecg_process(ecg, sampling_rate=srate)
    rpeaks = info["ECG_R_Peaks"]
    return rpeaks / srate


def estimate_global_offset(a, b, max_lag_s=0.3):
    """用 FFT 互相关估计两组 R 峰时间戳的全局时间偏移 (秒)。

    将两组峰转成等长脉冲串 (1 ms 分辨率, 覆盖共同时间范围), 用 FFT 做互相关
    (O(n log n)), 峰值位置即全局偏移。只在 |lag| <= max_lag_s 内搜索 (生理上
    合理偏移远小于 300 ms, 避免长记录下全范围搜索的数值噪声与无意义峰值)。
    返回偏移 (b - a, 秒), 或 None 若无法估计。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return None
    t0 = max(a.min(), b.min())
    t1 = min(a.max(), b.max())
    if t1 <= t0:
        return None
    # 只保留共同时间范围内的峰, 避免落在范围外的峰被 clip 压到端点形成伪峰
    a_in = a[(a >= t0) & (a <= t1)]
    b_in = b[(b >= t0) & (b <= t1)]
    if a_in.size == 0 or b_in.size == 0:
        return None
    step = 0.001  # 1 ms
    n = int(np.ceil((t1 - t0) / step)) + 1
    pulse_a = np.zeros(n)
    pulse_b = np.zeros(n)
    idx_a = np.clip(np.round((a_in - t0) / step).astype(int), 0, n - 1)
    idx_b = np.clip(np.round((b_in - t0) / step).astype(int), 0, n - 1)
    pulse_a[idx_a] = 1.0
    pulse_b[idx_b] = 1.0
    # FFT 互相关: corr[k] = sum_n pulse_a[n] * pulse_b[n+k] (b 相对 a 的滞后 k)
    fa = np.fft.rfft(pulse_a, n=2 * n)
    fb = np.fft.rfft(pulse_b, n=2 * n)
    corr_full = np.fft.irfft(fa * np.conj(fb), n=2 * n)  # 长度 2n
    # 前 n 个对应 lag 0..n-1, 后 n 个对应负 lag (-n+1..-1)
    max_lag = int(np.ceil(max_lag_s / step))
    max_lag = min(max_lag, n - 1)
    cand = []
    # 正滞后 k in [0, max_lag]
    cand.append((np.arange(0, max_lag + 1), corr_full[0:max_lag + 1]))
    # 负滞后 k in [-max_lag, -1] -> corr_full 尾部索引 2n-max_lag .. 2n-1
    neg_k = np.arange(-max_lag, 0)
    neg_idx = (2 * n + neg_k) % (2 * n)
    cand.append((neg_k, corr_full[neg_idx]))
    lags = np.concatenate([c[0] for c in cand])
    corr = np.concatenate([c[1] for c in cand])
    lag = lags[np.argmax(corr)]
    # 注意符号: FFT 互相关 corr[k] = sum_n pa[n] * pb[n+k] 在 b 晚于 a 时峰值落在
    # 负滞后 k (b 需前移才能对齐), 而返回的 offset 语义是 "b - a" (正值表示 b 晚于 a),
    # 供 match_peaks 的 b_corr = b - offset 使用。故取负号。
    return -lag * step  # b - a 偏移


def match_peaks(a, b, tol=0.05):
    """先互相关估全局偏移并校正, 再线性分配匹配两组时间戳。

    返回 (匹配对数, 时间差数组[自检-官方, 已扣全局偏移])。
    tol=0.05 (50ms): 官方时间戳用亚采样精度(抛物线插值), neurokit2 返回整数采样点,
    两者存在 ~24ms 系统偏移 + ~3ms 抖动。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0, np.array([])
    offset = estimate_global_offset(a, b)
    b_corr = b - offset if offset is not None else b
    cost = np.abs(a[:, None] - b_corr[None, :])
    row, col = linear_sum_assignment(cost)
    matched_diffs = []
    for i, j in zip(row, col):
        d = cost[i, j]
        if d <= tol:
            matched_diffs.append(b_corr[j] - a[i])  # 自检 - 官方 (b=detected - a=official)
    return len(matched_diffs), np.array(matched_diffs)


def selftest_estimate_global_offset():
    """验证 estimate_global_offset 的符号正确性 (无需数据)。

    关键: 若 b 晚于 a (b = a + δ), offset 必须返回 +δ, 使 b_corr = b - offset
    对齐到 a。若符号反了, match_peaks 会把匹配彻底带偏。
    """
    rng = np.random.default_rng(0)
    failures = []
    for shift in [-0.05, -0.012, 0.0, 0.012, 0.05]:
        a = np.sort(rng.uniform(10.0, 200.0, 300))
        b = a + shift
        off = estimate_global_offset(a, b)
        if off is None:
            failures.append(f"shift={shift:+.3f}: offset=None")
            continue
        residual = np.mean((b - off) - a)
        if abs(off - shift) > 1e-3 or abs(residual) > 1e-6:
            failures.append(
                f"shift={shift:+.3f}: offset={off:+.4f} residual={residual:+.5f}")
    if failures:
        raise SystemExit("estimate_global_offset 符号自检失败:\n" + "\n".join(failures))
    print("estimate_global_offset 符号自检通过 (offset == b - a, 对齐残差 ~0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="抽样被试数 (--all 时忽略)")
    ap.add_argument("--all", action="store_true", help="复检全部被试")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    ap.add_argument("--selftest", action="store_true",
                    help="仅运行互相关符号自检 (无需数据), 然后退出")
    args = ap.parse_args()

    if args.selftest:
        selftest_estimate_global_offset()
        return

    subs = sorted(
        p.name for p in EXP2.glob("sub-*")
        if p.is_dir() and p.name not in EXCLUDE
    )
    if args.all:
        sample = subs
    else:
        random.seed(args.seed)
        sample = sorted(random.sample(subs, min(args.n, len(subs))))

    rows = []
    skipped = []
    print(f"复检 {len(sample)} 个被试的 R 峰 (共 {len(subs)} 被试)\n")

    for sub in sample:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                key = f"{sub} {ses} {stim}"
                ecg = read_raw_ecg(sub, ses, stim)
                official = read_official_rpeak(sub, ses, stim)
                if ecg is None or official is None:
                    skipped.append({"record": key, "reason": "missing ecg/official"})
                    continue
                srate = read_sampling_rate(sub, ses, stim)
                if srate is None:
                    skipped.append({"record": key, "reason": "missing sampling rate"})
                    continue
                detected = detect_rpeaks(ecg, srate)

                n_off = len(official)
                n_self = len(detected)
                official_empty = n_off == 0
                n_diff_pct = abs(n_off - n_self) / max(n_off, 1) * 100

                n_match, diffs = match_peaks(official, detected, tol=0.05)
                rmse = float(np.sqrt(np.mean(diffs ** 2))) * 1000 if len(diffs) else np.nan  # ms
                mean_diff = float(np.mean(diffs)) * 1000 if len(diffs) else np.nan   # ms
                matched_rate = n_match / max(n_off, 1)

                rows.append({
                    "sub": sub, "ses": ses, "stim": stim,
                    "n_official": n_off, "n_self": n_self,
                    "official_empty": official_empty,
                    "n_diff_pct": round(n_diff_pct, 3),
                    "n_matched": n_match,
                    "matched_rate": round(matched_rate, 4),
                    "rmse_ms": round(rmse, 3) if not np.isnan(rmse) else None,
                    "mean_diff_ms": round(mean_diff, 3) if not np.isnan(mean_diff) else None,
                    "pass_count": n_diff_pct < 1.0,
                })

    # 写 CSV
    if not rows:
        print("无有效记录。")
        return
    fnames = ["sub", "ses", "stim", "n_official", "n_self", "official_empty",
              "n_diff_pct", "n_matched", "matched_rate", "rmse_ms", "mean_diff_ms",
              "rmse_centered_ms", "pass_count", "pass_rmse_centered"]
    with open(OUT / "rpeak_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 汇总
    n_rec = len(rows)
    n_pass_count = sum(r["pass_count"] for r in rows)

    # 固定偏移估计: 仅用高匹配率记录 (matched_rate >= 0.8) 的中位数 mean_diff。
    # 低匹配率记录是匹配失败 (n_matched 极少), 其 mean_diff 不反映真实偏移。
    high_match = [r for r in rows if r["matched_rate"] >= 0.8 and r["mean_diff_ms"] is not None]
    fixed_offset = float(np.median([r["mean_diff_ms"] for r in high_match])) if high_match else None

    # 扣偏后 RMSE: 残留随机误差 = sqrt(rmse^2 - mean_diff^2) (仅高匹配率记录)
    centered_vals = []
    n_pass_centered = 0
    n_usable = 0
    for r in rows:
        if r["matched_rate"] < 0.8 or r["rmse_ms"] is None or r["mean_diff_ms"] is None:
            continue
        n_usable += 1
        rm = r["rmse_ms"]
        md = r["mean_diff_ms"]
        var_resid = rm ** 2 - md ** 2
        c = float(np.sqrt(max(var_resid, 0.0)))
        r["rmse_centered_ms"] = round(c, 3)
        r["pass_rmse_centered"] = c < 10.0
        centered_vals.append(c)
        if c < 10.0:
            n_pass_centered += 1

    rmse_vals = [r["rmse_ms"] for r in rows if r["rmse_ms"] is not None]

    # ICC(1,1): 官方 vs 自检 每记录 R 峰数量的一致性 (随机效应单因素)
    def icc11(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        n = len(x)
        if n < 2:
            return None
        grand = np.mean(np.concatenate([x, y]))
        msb = 2 * np.sum((0.5 * (x + y) - grand) ** 2) / (n - 1)
        msw = np.sum((x - y) ** 2) / (2 * n)   # k=2 观测, 分母 2n (Shrout & Fleiss 1979)
        if msb == 0:
            return None
        return float((msb - msw) / (msb + msw))

    icc = icc11([r["n_official"] for r in rows], [r["n_self"] for r in rows])


    def wilson_ci(k, n, z=1.96):
        """Wilson score interval for a binomial proportion (95% CI by default)."""
        if n == 0:
            return (None, None)
        phat = k / n
        denom = 1 + z ** 2 / n
        center = (phat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    pass_count_ci = wilson_ci(n_pass_count, n_rec)
    pass_rmse_ci = wilson_ci(n_pass_centered, n_usable)

    summary = {
        "n_sampled_subjects": len(sample),
        "n_total_subjects": len(subs),
        "n_records": n_rec,
        "n_skipped": len(skipped),
        "skipped_records": skipped,
        "pass_count_rate": round(100 * n_pass_count / n_rec, 1) if n_rec else 0,
        "pass_count_rate_ci95": [round(100 * pass_count_ci[0], 1), round(100 * pass_count_ci[1], 1)] if pass_count_ci[0] is not None else None,
        "fixed_offset_ms": round(fixed_offset, 3) if fixed_offset is not None else None,
        "n_usable_records": n_usable,
        "pass_rmse_centered_rate": round(100 * n_pass_centered / n_usable, 1) if n_usable else 0,
        "pass_rmse_centered_ci95": [round(100 * pass_rmse_ci[0], 1), round(100 * pass_rmse_ci[1], 1)] if pass_rmse_ci[0] is not None else None,
        "median_rmse_ms": round(float(np.median(rmse_vals)), 3) if rmse_vals else None,
        "median_rmse_centered_ms": round(float(np.median(centered_vals)), 3) if centered_vals else None,
        "mean_abs_n_diff_pct": round(float(np.mean([r["n_diff_pct"] for r in rows])), 3) if rows else 0,
        "mean_n_matched_pct": round(float(np.mean([100 * r["n_matched"] / max(r["n_official"], 1) for r in rows])), 1) if rows else 0,
        "icc11_npeaks": round(icc, 4) if icc is not None else None,
    }
    with open(OUT / "rpeak_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"记录数: {n_rec} (跳过 {summary['n_skipped']} 条)")
    if summary["n_skipped"]:
        for s in skipped:
            print(f"  [跳过] {s['record']}: {s['reason']}")
    print(f"数量差异<1% 通过率: {summary['pass_count_rate']}% (95% CI: {summary['pass_count_rate_ci95']})")
    print(f"固定偏移 (高匹配率记录中位数): {summary['fixed_offset_ms']} ms")
    print(f"扣偏后 RMSE 中位数: {summary['median_rmse_centered_ms']} ms "
          f"(可用记录 {summary['n_usable_records']} 条)")
    print(f"扣偏后 RMSE<10ms 通过率: {summary['pass_rmse_centered_rate']}% (95% CI: {summary['pass_rmse_centered_ci95']})")
    print(f"原始 RMSE 中位数: {summary['median_rmse_ms']} ms (含固定偏移, 仅参考)")
    print(f"平均绝对数量差: {summary['mean_abs_n_diff_pct']}%")
    print(f"平均匹配率: {summary['mean_n_matched_pct']}%")
    print(f"ICC(1,1) 峰数一致性: {summary['icc11_npeaks']}")
    if not args.all:
        print(f"\n[提示] 仅抽样 {len(sample)}/{len(subs)} 被试, 上述 CI 为抽样估计; "
              f"建议用 --all 全量验证以获得稳定估计。")

    # 判定
    print("\n=== 冻结判定标准 ===")
    print(f"  R峰数量差异<1%: {'通过' if summary['pass_count_rate'] >= 95 else '需检查'}")
    print(f"  扣偏后RMSE<10ms: {'通过' if summary['pass_rmse_centered_rate'] >= 95 else '需检查'}")
    print("\n→ 结果已写入 data/rpeak_validation.csv 与 summary.json")


if __name__ == "__main__":
    run_all_or_once(main)
