#!/usr/bin/env python3
"""
Step 3 (双轨版): HEP 提取 — 读取预处理后的 clean/eyeonly 双轨数据

与裸版(03_hep_extract.py)的区别: 本脚本读取 Step 2 预处理产物,
对每个记录提取两套 HEP:
  - clean_raw.fif    (去眼动 + 去CFA)
  - eyeonly_raw.fif  (只去眼动, 保留CFA 作对照)

从而直接产出 CFA 双轨对比 (Fig 1 的核心)。

产出:
  - data/hep_dual.csv      每被试×条件×ROI×窗口×轨迹(clean/eyeonly) 的 HEP RMS
  - data/hep_waveform.csv  平均 HEP 波形 (供 Fig1A 双轨绘图)

用法: python 03b_hep_dual.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
import mne

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, EARLY_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI,
    read_rpeak, filter_rr, roi_idx, get_end_time, get_session_stims,
    reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)

# 绘图用关键通道 (Fig1A 波形)
PLOT_CHANNELS = ["Fz", "Cz", "Pz"]


def load_hep(fif_path, rpeaks, end_time):
    """读一次 fif, 提取基线校正后的 HEP 波形。

    返回 (hep_bl, hep_raw, ch_names, t, n_epochs, n_epochs_before_reject)。
    n_epochs 为振幅拒绝后保留数, n_epochs_before_reject 为拒绝前数 (供报告拒绝比例)。
    与 ROI 无关; 供后续对所有 ROI 在内存中算 RMS, 及取关键通道波形 (避免重复读 fif)。
    """
    if not fif_path.is_file():
        return None
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw = raw.copy().crop(0.0, end_time)
    ch_names = raw.ch_names
    srate = float(raw.info["sfreq"])

    valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
    rp = rpeaks[valid]
    rp = rp[filter_rr(rp)]
    if len(rp) < 5:
        return None

    data = raw.get_data() * 1e6   # μV

    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    starts = np.round((rp + TMIN) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)
    n_before = epochs.shape[0]

    # epoch 振幅拒绝: 剔除峰峰值 > 阈值的伪迹 epoch (Virjee 第二大影响参数)
    keep_ep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
    epochs = epochs[keep_ep]
    if len(epochs) < 5:
        return None

    hep = epochs.mean(axis=0)  # (n_ch, n_t)

    base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    # 基线校正: 每通道减其基线均值
    baseline = hep[:, base_mask].mean(axis=1, keepdims=True)  # (n_ch, 1)
    hep_bl = hep - baseline

    return hep_bl, hep, ch_names, t, len(epochs), n_before


def rms_from_hep(hep_bl, hep_raw, ch_names, roi_chans, t):
    """从已提取的 HEP 波形计算 ROI×窗口幅值 (双报告: 有符号均值 + RMS)。

    同时输出两个指标:
      - mean (有符号均值电压): 保留 HEP 极性 (Kandasamy 2026 发现早期 HEP 负向),
        对背景噪声无正偏; 但极性在被试间翻转时可能互相抵消。
      - rms (均方根): 对极性翻转鲁棒 (Steinfath 2025 的 HEP amplitude 惯例),
        但对噪声有正偏 (E[RMS]>0)。
    主分析用 rms (对齐 HEP 惯例), 有符号均值作为极性信息/敏感性对照。
    """
    ridx = roi_idx(ch_names, roi_chans)
    if len(ridx) == 0:
        return None
    main_mask = (t >= MAIN_WIN[0]) & (t <= MAIN_WIN[1])
    early_mask = (t >= EARLY_WIN[0]) & (t <= EARLY_WIN[1])
    out = {}
    for win_name, mask in [("main_200_600", main_mask), ("early_0_200", early_mask)]:
        vals_raw = hep_raw[np.ix_(ridx, np.where(mask)[0])]
        vals_bl = hep_bl[np.ix_(ridx, np.where(mask)[0])]
        # 有符号均值电压 (mean amplitude, 保留极性)
        out[f"{win_name}_mean_uv"] = float(np.mean(vals_bl))
        out[f"{win_name}_mean_uv_raw"] = float(np.mean(vals_raw))
        # RMS (幅值大小, 对极性鲁棒)
        out[f"{win_name}_rms_uv"] = float(np.sqrt(np.mean(vals_bl ** 2)))
        out[f"{win_name}_rms_uv_raw"] = float(np.sqrt(np.mean(vals_raw ** 2)))
    return out


def plot_from_hep(hep_bl, ch_names):
    """从 HEP 波形取关键通道 (Fz/Cz/Pz) 波形 (基线校正后, 供绘图)。"""
    plot_wave = {}
    for c in PLOT_CHANNELS:
        if c in ch_names:
            plot_wave[c] = hep_bl[ch_names.index(c)]
    return plot_wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default="sub-01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = [args.sub]

    rows = []
    wave_rows = []

    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                rpeaks = read_rpeak(sub, ses, stim)
                if rpeaks is None or len(rpeaks) == 0:
                    continue
                end_time = get_end_time(sub, ses, stim)
                if end_time is None:
                    continue

                for track in ["clean", "eyeonly"]:
                    fif = PROC / sub / ses / f"{stim}_{track}_raw.fif"
                    loaded = load_hep(fif, rpeaks, end_time)
                    if loaded is None:
                        continue
                    hep_bl, hep_raw, ch_names, t, n_epochs, n_epochs_before = loaded

                    # 关键通道波形 (Fz/Cz/Pz, 绘图用): 每 track 独立存一次
                    plot_wave = plot_from_hep(hep_bl, ch_names)
                    for c, wv in plot_wave.items():
                        for ti, amp in zip(t, wv):
                            wave_rows.append({
                                "sub": sub, "ses": ses, "stim": stim,
                                "condition": SESSIONS[ses], "track": track,
                                "channel": c, "time_ms": round(ti * 1000, 3),
                                "amp_uv": round(float(amp), 5),
                            })

                    for roi_name, roi_chans in ROI.items():
                        rms_dict = rms_from_hep(hep_bl, hep_raw, ch_names, roi_chans, t)
                        if rms_dict is None:
                            continue
                        row = {
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses], "track": track,
                            "roi": roi_name, "n_epochs": n_epochs,
                            "n_epochs_before_reject": n_epochs_before,
                        }
                        row.update(rms_dict)
                        rows.append(row)

    if rows:
        fn = list(rows[0].keys())
        with open(OUT / "hep_dual.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    if wave_rows:
        fn = list(wave_rows[0].keys())
        with open(OUT / "hep_waveform.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(wave_rows)

    print(f"完成 {len(rows)} 条双轨记录, {len(wave_rows)} 条波形点")
    if rows:
        # 打印 CFA 双轨对比样例
        print("\nCFA 双轨对比 (sub-01 ses-01, 主窗口 RMS):")
        for r in rows[:6]:
            print(f"  {r['stim']} {r['track']} {r['roi']}: "
                  f"main={r['main_200_600_rms_uv']:.3f}μV, "
                  f"early={r['early_0_200_rms_uv']:.3f}μV")


if __name__ == "__main__":
    run_all_or_once(main)
