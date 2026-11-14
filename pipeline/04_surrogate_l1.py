#!/usr/bin/env python3
"""
Step 4: Surrogate / Null 分析 (Level 1 HEP) — 纯 numpy 高效版

目的: 验证"心搏锁定的 HEP"真实, 而非伪迹/共同刺激/缓慢漂移。

三类 surrogate (操作定义 §6):
  1. 时间平移:    R 峰序列平移 Δ(5-60s), 用 numpy 索引直接切片
  2. 相位随机化:  对每个 epoch 向量化 FFT 随机相位
  3. 跨被试替代:  (需跨被试, 见 04b)

统计: 真实 HEP RMS vs null 分布, p = (1 + ΣI(null>=obs))/(N+1)

完全绕开 MNE Epochs, 用纯 numpy fancy-indexing 提取 epoch, 速度快 100x。

用法: python 04_surrogate_l1.py [--sub sub-01] [--all] [--n_perm 1000] [--n_perm_pr 200]
"""

import argparse
import csv

import numpy as np
import mne

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI, read_rpeak, filter_rr, roi_idx, get_end_time,
    get_session_stims, fdr_bh, reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)


def read_raw_data(sub, ses, stim):
    """返回 (data μV 2D, ch_names, end_time, srate)。读预处理后的 clean_raw.fif (与主分析一致)"""
    p = PROC / sub / ses / f"{stim}_clean_raw.fif"
    if not p.is_file():
        return None, None, None, None
    raw = mne.io.read_raw_fif(p, preload=True, verbose=False)
    srate = float(raw.info["sfreq"])

    end_time = get_end_time(sub, ses, stim)
    if end_time is None:
        end_time = raw.n_times / srate

    raw = raw.copy().crop(0.0, end_time)
    data = raw.get_data() * 1e6   # μV
    return data, raw.ch_names, end_time, srate


def main_window_mask(times):
    return (times >= MAIN_WIN[0]) & (times <= MAIN_WIN[1])


def baseline_mask(times):
    return (times >= BASELINE_WIN[0]) & (times < BASELINE_WIN[1])


def extract_epochs(data, rpeaks, srate, tmin, tmax):
    """纯 numpy 提取 epochs。返回 (n_ep, n_ch, n_t)"""
    n_ch, n_total = data.shape
    t = np.arange(int((tmax - tmin) * srate)) / srate + tmin
    starts = np.round((rpeaks + tmin) * srate).astype(int)
    n_t = len(t)
    # 过滤越界
    valid = (starts >= 0) & (starts + n_t <= n_total)
    starts = starts[valid]
    # fancy indexing
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx]  # (n_ch, n_ep, n_t)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_ep, n_ch, n_t)
    return epochs, t


def roi_rms(epochs, roi_idx_arr, win_mask, base_mask):
    """基线校正后 ROI×窗口 RMS。先 epoch 振幅拒绝, 再每通道减基线, 算窗口 RMS。"""
    if len(roi_idx_arr) == 0:
        return None
    keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
    epochs = epochs[keep]
    if len(epochs) < 5:
        return None
    hep = epochs.mean(axis=0)  # (n_ch, n_t)
    base = hep[:, base_mask].mean(axis=1, keepdims=True)
    hep_bl = hep - base
    vals = hep_bl[np.ix_(roi_idx_arr, np.where(win_mask)[0])]
    return float(np.sqrt(np.mean(vals ** 2)))


def process_subject(sub, n_perm, n_perm_pr, seed, roi):
    """处理单个被试的全部会话×刺激, 返回 (rows, null_rows)。

    每个被试使用独立的 RNG (seed + 被试编号), 保证 --sub 与 --all 中同一被试可复现。
    """
    sub_seed = seed + int(sub.split("-")[1])
    rng = np.random.default_rng(sub_seed)
    rows = []
    null_rows = []

    for ses in SESSIONS:
        for stim in get_session_stims(ses):
            data, ch_names, end_time, srate = read_raw_data(sub, ses, stim)
            rpeaks = read_rpeak(sub, ses, stim)
            if data is None or rpeaks is None or len(rpeaks) == 0:
                continue

            ridx = roi_idx(ch_names, roi)
            if len(ridx) == 0:
                continue
            valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
            rp = rpeaks[valid]
            rp = rp[filter_rr(rp)]
            if len(rp) < 5:
                continue

            # 真实 HEP
            epochs, t = extract_epochs(data, rp, srate, TMIN, TMAX)
            wmask = main_window_mask(t)
            bmask = baseline_mask(t)
            obs_rms = roi_rms(epochs, ridx, wmask, bmask)
            if obs_rms is None:
                continue
            n_epochs = epochs.shape[0]

            # --- Surrogate 1: 时间平移 (丢弃越界, 不取模折回) ---
            null_ts = []
            for _ in range(n_perm):
                delta = rng.uniform(5.0, 60.0)
                shifted = rp + delta
                sv = shifted[(shifted > abs(TMIN)) & (shifted < end_time - TMAX)]
                sv = sv[filter_rr(sv)]
                if len(sv) < 5:
                    continue
                se, _ = extract_epochs(data, sv, srate, TMIN, TMAX)
                v = roi_rms(se, ridx, wmask, bmask)
                if v is not None:
                    null_ts.append(v)
            null_ts = np.array(null_ts, dtype=float)

            # --- Surrogate 2: 相位随机化 (DC/Nyquist 保持实数) ---
            # 与观测值 obs_rms 走同一套 epoch 振幅拒绝逻辑 (reject_epochs_by_amplitude),
            # 否则 obs 与 null 的预处理不一致会使 p 值有偏。
            null_pr = []
            roi_data = data[ridx]
            n_total = roi_data.shape[-1]
            # 补零到下一个 2 的幂, 使 rfft/irfft 走快速 Cooley-Tukey 路径而非
            # 非 2 次幂的 fftblue 慢速回退 (33413 点信号会慢 5-20 倍)。
            # 补零只在高频 bins 加零, 相位随机化的统计性质不变, 截回原长后等价。
            n_fft = 1 << (n_total - 1).bit_length()
            rfft = np.fft.rfft(roi_data, n=n_fft, axis=-1)
            mag = np.abs(rfft)
            n_freq = rfft.shape[-1]
            n_t = len(t)
            starts0 = np.round((rp + TMIN) * srate).astype(int)
            starts0 = starts0[(starts0 >= 0) & (starts0 + n_t <= n_total)]
            idx0 = starts0[:, None] + np.arange(n_t)[None, :]
            wm = np.where(wmask)[0]
            bm = np.where(bmask)[0]
            batch = 50
            for i in range(0, n_perm_pr, batch):
                k = min(batch, n_perm_pr - i)
                # 只随机化中间频率的相位, DC/Nyquist 保持 0 相位 (实部)
                phase = np.zeros((k, len(ridx), n_freq), dtype=complex)
                phase[..., 1:-1] = np.exp(1j * rng.uniform(
                    0, 2 * np.pi, size=(k, len(ridx), n_freq - 2)))
                rand = np.fft.irfft(mag[None] * phase, n=n_fft, axis=-1)
                rand = rand[..., :n_total]   # 截回原长 (补零部分丢弃)
                for j in range(k):
                    rr = rand[j]
                    # epoch 提取 + 振幅拒绝 (与 roi_rms 一致)
                    se = np.transpose(rr[:, idx0], (1, 0, 2))  # (n_ep, n_ch, n_t)
                    keep = reject_epochs_by_amplitude(se, EPOCH_REJECT_UV)
                    se = se[keep]
                    if len(se) < 5:
                        continue
                    hep = se.mean(axis=0)
                    base = hep[:, bm].mean(axis=1, keepdims=True)
                    hep_bl = hep - base
                    vals = hep_bl[:, wm]
                    null_pr.append(float(np.sqrt(np.mean(vals ** 2))))
            null_pr = np.array(null_pr, dtype=float)

            def pval(null, obs):
                if len(null) == 0:
                    return np.nan
                return (1 + np.sum(null >= obs)) / (len(null) + 1)

            rows.append({
                "sub": sub, "ses": ses, "stim": stim,
                "condition": SESSIONS[ses], "n_epochs": n_epochs,
                "obs_rms_uv": round(obs_rms, 4),
                "p_time_shift": round(pval(null_ts, obs_rms), 4),
                "p_phase_rand": round(pval(null_pr, obs_rms), 4),
            })
            for v in null_ts:
                null_rows.append({"sub": sub, "ses": ses, "stim": stim, "type": "time_shift", "rms_uv": v})
            for v in null_pr:
                null_rows.append({"sub": sub, "ses": ses, "stim": stim, "type": "phase_rand", "rms_uv": v})

    return rows, null_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default=None, help="单被试(如 sub-01)")
    ap.add_argument("--all", action="store_true", help="跑全部被试")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="时间平移 surrogate 次数 (快)")
    ap.add_argument("--n_perm_pr", type=int, default=1000,
                    help="相位随机化 surrogate 次数 (慢, p 值分辨率 1/(n+1), 默认1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    n_perm = args.n_perm
    n_perm_pr = args.n_perm_pr
    roi = ROI["frontocentral"]

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    elif args.sub:
        subs = [args.sub]
    else:
        subs = [args.sub] if args.sub else ["sub-01"]

    all_rows = []
    all_null_rows = []

    for sub in subs:
        rows, null_rows = process_subject(sub, n_perm, n_perm_pr, args.seed, roi)
        all_rows.extend(rows)
        all_null_rows.extend(null_rows)
        print(f"完成 {sub}: {len(rows)} 条记录")

    if all_rows:
        # FDR 校正 (逐记录 p 值, 全局 BH)
        p_ts = np.array([r["p_time_shift"] for r in all_rows])
        p_pr = np.array([r["p_phase_rand"] for r in all_rows])
        fdr_ts = fdr_bh(p_ts)
        fdr_pr = fdr_bh(p_pr)
        for r, f1, f2 in zip(all_rows, fdr_ts, fdr_pr):
            r["p_time_shift_fdr"] = round(float(f1), 4)
            r["p_phase_rand_fdr"] = round(float(f2), 4)
        fn = list(all_rows[0].keys())
        with open(OUT / "surrogate_l1.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(all_rows)
    if all_null_rows:
        fn = list(all_null_rows[0].keys())
        with open(OUT / "surrogate_null_dist.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(all_null_rows)

    print(f"\n总计 {len(all_rows)} 条记录")
    for r in all_rows[:20]:
        print(f"  {r['sub']} {r['ses']} {r['stim']}: obs={r['obs_rms_uv']:.3f}μV, "
              f"p_ts={r['p_time_shift']}, p_pr={r['p_phase_rand']}")


if __name__ == "__main__":
    run_all_or_once(main)
