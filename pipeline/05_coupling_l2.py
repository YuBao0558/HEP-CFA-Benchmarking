#!/usr/bin/env python3
"""
Step 5: Level 2 — 心搏锁定的 EEG 相位一致性 (ITPC / 心跳锁定 PLV)

目的: 计算心搏后固定延迟处 EEG 相位的跨心跳一致性 (操作定义 §2.2 / §5)。
      频带 delta/theta/alpha/beta; ROI 额中央/中央/顶叶。

方法:
  - 从预处理 clean_raw.fif 读 EEG (去眼动+去CFA)
  - 对 EEG 各频带做带通 → 希尔伯特变换取瞬时相位 φ(t)
  - 以每个 R 峰为观测单位, 取 R 峰后延迟 τ 处的相位 φ_k(τ)
  - 跨心跳平均 ITPC(τ) = |(1/K) Σ_k exp(i·φ_k(τ))|
  - 显著性: R 峰时间抖动 surrogate (给 R 峰加随机抖动破坏 EEG 与心搏的时间对齐),
    p = (1 + ΣI(null>=obs))/(N+1)

  注: 这是心搏锁定的相位一致性 (ITPC), 不是连续两振荡器的频率比锁定。
      后者对 4s 滑窗内的相位差取平均, 因 cardiac 相位(≈4圈)与 EEG 相位
      (theta≈16圈/alpha≈40圈) 在窗内各自旋转, exp(i·dφ) 均匀铺满单位圆,
      平均趋近 0, 不反映心搏锁定。

产出:
  - data/coupling_long.csv     每被试×条件×频带×ROI 的 ITPC (主/早期窗口均值 + p 值)
  - data/coupling_timefreq.csv 心搏锁定 ITPC(τ) 曲线 (供 Fig2A 绘图)

用法: python 05_coupling_l2.py [--sub sub-01] [--all] [--n_perm 200] [--seed 42]
"""

import argparse
import csv

import numpy as np
import mne
from scipy.signal import firwin, filtfilt, hilbert

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, EARLY_WIN, ROI, read_rpeak, filter_rr, get_session_stims, fdr_bh,
    run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def bandpass(data, band, srate):
    """FIR 零相位带通 (与 02/03 预处理一致, 用 FIR 而非 IIR)。

    data: (n_ch, n_times); band: (lo, hi) Hz; srate: 采样率。
    用 firwin 设计带通 FIR (窗长 ≈ 3.3s, 与 MNE raw.filter 默认一致) + filtfilt
    零相位, 保证相位响应与 Level 1 预处理一致, 避免 IIR 在低频截止处的相位失真。
    """
    lo, hi = band
    # FIR 窗长 (MNE 默认 firwin 约 3.3s); 取奇数保证线性相位 FIR 有整数群延迟
    ntaps = int(3.3 * srate)
    if ntaps % 2 == 0:
        ntaps += 1
    # firwin 需要归一化截止频率 (相对 Nyquist); 加过渡带宽避免边缘奇异
    nyq = srate / 2.0
    b = firwin(ntaps, [lo, hi], pass_zero=False, fs=srate, window="hamming")
    return filtfilt(b, [1.0], data, axis=-1)


def compute_itpc(eeg_phase, rpeaks, tmin, tmax, srate):
    """心搏锁定 ITPC。

    eeg_phase: (n_ch, n_times); rpeaks: (n_r,) 秒; 返回 (n_ch, n_t) 的 ITPC(τ)。
    以每个 R 峰为观测单位, 取 R 峰后延迟 τ 处的相位, 跨心跳平均。
    """
    n_ch, n_times = eeg_phase.shape
    lags = np.arange(int((tmax - tmin) * srate))          # 采样点延迟
    lag_t = lags / srate + tmin                           # 秒 (相对 R 峰)
    starts = np.round((rpeaks + tmin) * srate).astype(int)
    valid = (starts >= 0) & (starts + len(lags) <= n_times)
    starts = starts[valid]
    if len(starts) == 0:
        return None, lag_t

    idx = starts[:, None] + lags[None, :]                 # (K, n_t)
    # 相位 (K, n_ch, n_t)
    phase_beats = eeg_phase[:, idx].transpose(1, 0, 2)    # (K, n_ch, n_t)
    # 跨心跳平均复数相位, 取模 = ITPC
    itpc = np.abs(np.mean(np.exp(1j * phase_beats), axis=0))  # (n_ch, n_t)
    return itpc, lag_t


def compute_itpc_from_z(Z, starts, n_lags, n_ch):
    """从预计算的单位复数相位 Z (n_ch, n_times) 批量计算 ITPC。

    starts: (K,) int 采样点索引 (已过滤越界); 返回 (n_ch, n_lags) ITPC。
    等价于 compute_itpc 的核心, 但复用预计算的 Z=exp(1j*φ), 且 starts 可由批量
    抖动直接传入, 避免每 perm 重新 fancy-index 整个相位矩阵。
    """
    if len(starts) == 0:
        return None
    idx = starts[:, None] + np.arange(n_lags)[None, :]    # (K, n_lags)
    # 直接对复数 Z gather: (n_ch, K, n_lags) -> mean over K -> (n_ch, n_lags)
    z_beats = Z[:, idx]                                   # (n_ch, K, n_lags)
    itpc = np.abs(np.mean(z_beats, axis=1))
    return itpc


def surrogate_itpc_batched(Z, rpeaks, tmin, tmax, srate, n_perm, seed, band):
    """向量化 R 峰时间抖动 surrogate。

    与 surrogate_itpc 语义一致, 但把 n_perm 次抖动一次性批量完成:
      1. 预计算 Z = exp(1j * eeg_phase) (调用方传入, 每频带只算一次)。
      2. 生成 (n_perm, K) 抖动矩阵, sort + filter_rr 逐行。
      3. 一次性 gather Z 到 (n_perm, n_ch, K, n_lags), 沿 K 取均值取模。
    返回 (null_maps, k_null_mean): null_maps 为 (n_perm, n_ch, n_lags) 数组,
    k_null_mean 为 null 平均 epoch 数。

    注: filter_rr 需逐行 (因各 perm 独立), 用 numba/纯 python 均慢; 这里用
    向量化 sort + 逐行 filter (perm 数大时仍快于逐 perm 全流程)。
    """
    n_ch, n_times = Z.shape
    n_lags = int((tmax - tmin) * srate)
    lo, hi = band
    center = np.sqrt(lo * hi)
    jitter_scale = max(0.2, 1.0 / center)
    rng = np.random.default_rng(seed)

    # (n_perm, K) 抖动矩阵
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=(n_perm, len(rpeaks)))
    rp_jit = np.sort(rpeaks[None, :] + jitter, axis=1)    # (n_perm, K)

    # 逐行 filter_rr (移除 RR < MIN_RR 的峰)
    from common import MIN_RR
    keep = np.ones_like(rp_jit, dtype=bool)
    last = rp_jit[:, 0].copy()
    for i in range(1, rp_jit.shape[1]):
        cur = rp_jit[:, i]
        bad = (cur - last) < MIN_RR
        keep[:, i] = ~bad
        # 更新 last: 被保留的才更新
        last = np.where(keep[:, i], cur, last)
    # 逐行过滤 (注意: np.nan * False == nan, 不能用乘法置 NaN, 用 np.where)
    rp_jit = np.where(keep, rp_jit, np.nan)
    # 转为 starts (采样点索引), NaN -> 非法
    starts_all = np.round((rp_jit + tmin) * srate).astype(float)
    # 越界检查: starts>=0 且 starts+n_lags<=n_times
    ok = (starts_all >= 0) & (starts_all + n_lags <= n_times)
    starts_all[~ok] = np.nan

    # 每个 perm 的有效峰数 K
    n_valid = ok.sum(axis=1)
    k_nulls = n_valid

    # 批量 gather: 把 (n_perm, K) 展平, 用 idx 累加 lags
    # 直接用带 NaN 的索引不合法, 改用 mask 后每行 gather 均值。
    # 为最大化向量化: 用 einsum 式分桶平均, 但 K 各不同, 退化为逐行也行——
    # 逐行 gather 已是 n_perm 次而非原 n_perm 次完整 compute_itpc, 且复用 Z。
    null_maps = np.empty((n_perm, n_ch, n_lags), dtype=float)
    lag_ar = np.arange(n_lags)
    for p in range(n_perm):
        st = starts_all[p][np.isfinite(starts_all[p])].astype(int)
        if len(st) == 0:
            null_maps[p] = 0.0
            continue
        idx = st[:, None] + lag_ar[None, :]               # (K, n_lags)
        z_beats = Z[:, idx]                               # (n_ch, K, n_lags)
        null_maps[p] = np.abs(np.mean(z_beats, axis=1))

    k_null_mean = float(np.mean(k_nulls)) if len(k_nulls) else np.nan
    return null_maps, k_null_mean


def window_mean(itpc, ch_idx, chans, lag_t, win):
    """窗口内 ROI 平均 ITPC (标量)。"""
    idx = [ch_idx[c] for c in chans if c in ch_idx]
    mask = (lag_t >= win[0]) & (lag_t <= win[1])
    return float(np.mean(itpc[np.ix_(idx, np.where(mask)[0])]))


def surrogate_itpc(eeg_phase, rpeaks, tmin, tmax, srate, n_perm, seed, band):
    """R 峰时间抖动 surrogate: 给 R 峰加随机抖动, 破坏 EEG 与心搏的时间对齐。

    返回 (null_maps, k_null_mean):
      - null_maps: list of (n_ch, n_t) null ITPC 图 (每个 perm 一个)
      - k_null_mean: null 的平均 epoch 数 K (供读者检查 K 匹配情况)
    与 ROI 无关, 每频带只算一次。
    抖动尺度按频带调整: 取该频带中心周期的 1 倍 (而非 2 倍), 避免低频(delta)
    抖动过大把 R 峰序列彻底打乱、null 退化为纯随机重采样。theta/alpha/beta
    由下限 0.2s 兜底。

    K 处理: 不强制 K 匹配。抖动 + filter_rr 后 null 用真实剩余峰数。ITPC 对
    小 K 有正偏 (E[ITPC]≈sqrt(π/4K)), 若 null 的 K 略小于 obs, null 略偏高,
    检验偏保守 (安全方向, 不会产生假阳性)。若用有放回采样强行补齐 K, 会人为
    压低 null ITPC 导致反保守, 故不采用。
    """
    lo, hi = band
    center = np.sqrt(lo * hi)              # 几何中心频率 (Hz)
    jitter_scale = max(0.2, 1.0 / center)  # 1 个中心周期, 下限 0.2s
    rng = np.random.default_rng(seed)
    null_maps = []
    k_nulls = []
    for _ in range(n_perm):
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=len(rpeaks))
        rp_jittered = np.sort(rpeaks + jitter)
        rp_jittered = rp_jittered[filter_rr(rp_jittered)]
        null_itpc, _ = compute_itpc(eeg_phase, rp_jittered, tmin, tmax, srate)
        if null_itpc is None:
            continue
        null_maps.append(null_itpc)
        k_nulls.append(len(rp_jittered))
    k_null_mean = float(np.mean(k_nulls)) if k_nulls else np.nan
    return null_maps, k_null_mean


def pval(null, obs):
    if len(null) == 0:
        return np.nan
    return (1 + np.sum(null >= obs)) / (len(null) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default="sub-01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="ITPC surrogate 次数 (p 值分辨率 = 1/(n_perm+1), 默认 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = [args.sub]

    rows = []
    tf_rows = []   # 心搏锁定 ITPC(τ) 曲线 (供 Fig2A)

    n_total = len(subs) * len(SESSIONS) * len(STIMS)
    done = 0
    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                done += 1
                fif = PROC / sub / ses / f"{stim}_clean_raw.fif"
                rpeaks = read_rpeak(sub, ses, stim)
                if not fif.is_file() or rpeaks is None or len(rpeaks) < 10:
                    print(f"[{done}/{n_total}] {sub} {ses} {stim} 跳过 (无文件/无R峰)", flush=True)
                    continue

                raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
                data = raw.get_data() * 1e6   # μV
                ch_names = raw.ch_names
                n_times = data.shape[1]
                srate = float(raw.info["sfreq"])

                # 过滤 R 峰 (最小 RR + 时间窗内)
                valid = (rpeaks > abs(TMIN)) & (rpeaks < n_times / srate - TMAX)
                rp = rpeaks[valid]
                rp = rp[filter_rr(rp)]
                if len(rp) < 5:
                    print(f"[{done}/{n_total}] {sub} {ses} {stim} 跳过 (R峰<5)", flush=True)
                    continue

                ch_idx = {c: i for i, c in enumerate(ch_names)}
                print(f"[{done}/{n_total}] {sub} {ses} {stim} ({len(rp)} beats)", flush=True)

                for band_name, band in BANDS.items():
                    bp = bandpass(data, band, srate)   # (n_ch, n_times)
                    analytic = hilbert(bp, axis=-1)
                    eeg_phase = np.angle(analytic)
                    Z = np.exp(1j * eeg_phase)         # 预计算单位复数相位 (n_ch, n_times)

                    itpc, lag_t = compute_itpc(eeg_phase, rp, TMIN, TMAX, srate)
                    if itpc is None:
                        continue

                    # surrogate null ITPC 图 (每频带只算一次, 与 ROI 无关)
                    seed = args.seed + int(sub.split("-")[1]) + list(BANDS).index(band_name)
                    null_maps, k_null_mean = surrogate_itpc_batched(
                        Z, rp, TMIN, TMAX, srate, args.n_perm, seed, band)
                    # null_maps: (n_perm, n_ch, n_lags)

                    # 窗口掩码 (lags 维)
                    main_mask = (lag_t >= MAIN_WIN[0]) & (lag_t <= MAIN_WIN[1])
                    early_mask = (lag_t >= EARLY_WIN[0]) & (lag_t <= EARLY_WIN[1])

                    for roi_name, chans in ROI.items():
                        main_itpc = window_mean(itpc, ch_idx, chans, lag_t, MAIN_WIN)
                        early_itpc = window_mean(itpc, ch_idx, chans, lag_t, EARLY_WIN)

                        # 对每个 null 图窗口聚合, 得 null 分布 (向量化: ROI通道×窗口)
                        idx = [ch_idx[c] for c in chans if c in ch_idx]
                        null_main = null_maps[np.ix_(np.arange(null_maps.shape[0]),
                                                     idx, np.where(main_mask)[0])].mean(axis=(1, 2))
                        null_early = null_maps[np.ix_(np.arange(null_maps.shape[0]),
                                                      idx, np.where(early_mask)[0])].mean(axis=(1, 2))

                        rows.append({
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses],
                            "band": band_name, "roi": roi_name,
                            "main_200_600_itpc": round(main_itpc, 5),
                            "early_0_200_itpc": round(early_itpc, 5),
                            "p_itpc_main": round(pval(null_main, main_itpc), 5),
                            "p_itpc_early": round(pval(null_early, early_itpc), 5),
                            "n_epochs_obs": len(rp),
                            "n_epochs_null_mean": round(k_null_mean, 2),
                        })

                    # 时间分辨 (frontocentral 只存, 供 Fig2A 时频图)
                    roi_fc = ROI["frontocentral"]
                    idx_fc = [ch_idx[c] for c in roi_fc if c in ch_idx]
                    itpc_fc = itpc[idx_fc].mean(axis=0)   # (n_t,)
                    for lt, iv in zip(lag_t, itpc_fc):
                        tf_rows.append({
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses],
                            "band": band_name,
                            "time_s": round(float(lt), 3),
                            "itpc": round(float(iv), 5),
                        })

    if rows:
        # FDR 校正 (逐记录 p 值, 全局 BH)
        p_main = np.array([r["p_itpc_main"] for r in rows], dtype=float)
        p_early = np.array([r["p_itpc_early"] for r in rows], dtype=float)
        fdr_main = fdr_bh(p_main)
        fdr_early = fdr_bh(p_early)
        for r, f1, f2 in zip(rows, fdr_main, fdr_early):
            r["p_itpc_main_fdr"] = round(float(f1), 4)
            r["p_itpc_early_fdr"] = round(float(f2), 4)
        fn = list(rows[0].keys())
        with open(OUT / "coupling_long.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    if tf_rows:
        fn = list(tf_rows[0].keys())
        with open(OUT / "coupling_timefreq.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(tf_rows)

    print(f"完成 {len(rows)} 条耦合记录, {len(tf_rows)} 条时频点")
    if rows:
        for band_name in BANDS:
            r = next((x for x in rows
                      if x["band"] == band_name and x["roi"] == "frontocentral"), None)
            if r:
                print(f"  {band_name:6s} frontocentral: "
                      f"ITPC={r['main_200_600_itpc']} p={r['p_itpc_main']}")


if __name__ == "__main__":
    run_all_or_once(main)
