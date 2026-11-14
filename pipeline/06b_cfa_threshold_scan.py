#!/usr/bin/env python3
"""
Step 6b: CFA 阈值敏感性扫描 (最小多宇宙)

目的: 验证两个论点 (供论文角度 B/C):
  1. CFA 分离是否"连续谱" — 阈值 vs CFA 检出率曲线是否有自然断点
  2. 状态调制零结果是否在阈值变化下稳健 — 阈值 vs ΔBHCI 是否恒 ≈ 0

方法 (与 Step 2 一致, 固定 ICA=20 成分, 只扫 CFA 能量比阈值):
  对每个记录跑一次 ICA, 计算每个成分的锁时能量比, 然后对阈值
  [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50] 重新分类 CFA 成分,
  重算 frontocentral 主窗口的 clean/eyeonly HEP RMS。

产出:
  - data/cfa_threshold_scan.csv  每记录 × 阈值 × track 的 HEP RMS + n_cfa

用法: python 06b_cfa_threshold_scan.py [--sub sub-01] [--all] [--n_subs N]
"""

import argparse
import csv

import numpy as np

from common import (
    ROOT, EXP2, OUT, SESSIONS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI, read_rpeak, filter_rr, get_end_time, get_session_stims,
    reject_epochs_by_amplitude, run_all_or_once,
)
from preprocess_utils import (
    read_eeg_bdf, apply_montage_and_interpolate, bandpass_filter,
    set_average_reference, fit_ica,
    compute_energy_ratios, detect_eye_idx,
)

OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
# ICA 成分数扫描维度 (主分析 40 成分向下扫描 15-40, 验证 CFA 分类稳健性;
# 上限 40 = 数据有效秩 ~41, 超过会导致 ICA 病态不收敛)
N_COMPONENTS = [15, 20, 25, 30, 40]


def extract_roi_rms(raw_eeg_clean, rpeaks, end_time, srate, roi_chans):
    """从清洗后的 EEG 提取 frontocentral 主窗口 HEP RMS (基线校正, 与 03b/06 一致)。

    含 epoch 振幅拒绝 (reject_epochs_by_amplitude), 返回 (rms, n_epochs_kept)。
    """
    data = raw_eeg_clean.get_data() * 1e6  # μV
    ch_names = raw_eeg_clean.ch_names
    ch_idx = {c: i for i, c in enumerate(ch_names)}
    idx = [ch_idx[c] for c in roi_chans if c in ch_idx]
    if not idx:
        return None, 0

    valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
    rp = rpeaks[valid]
    rp = rp[filter_rr(rp)]
    if len(rp) < 5:
        return None, 0

    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    starts = np.round((rp + TMIN) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
    idx_t = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx_t].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)

    keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
    epochs = epochs[keep]
    if len(epochs) < 5:
        return None, 0

    hep = epochs.mean(axis=0)

    # 基线校正
    base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    hep_bl = hep - hep[:, base_mask].mean(axis=1, keepdims=True)

    mask = (t >= MAIN_WIN[0]) & (t <= MAIN_WIN[1])
    vals = hep_bl[np.ix_(idx, np.where(mask)[0])]
    return float(np.sqrt(np.mean(vals ** 2))), len(epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n_subs", type=int, default=None, help="限制被试数(测试)")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    elif args.sub:
        subs = [args.sub]
    else:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)[:5]
    if args.n_subs:
        subs = subs[:args.n_subs]

    roi_chans = ROI["frontocentral"]
    rows = []
    ratio_rows = []
    n_rec = 0

    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                rpeaks = read_rpeak(sub, ses, stim)
                end_time = get_end_time(sub, ses, stim)
                if rpeaks is None or end_time is None:
                    continue
                try:
                    raw = read_eeg_bdf(sub, ses, stim)
                    if raw is None:
                        continue
                    srate = float(raw.info["sfreq"])
                    raw = raw.copy().crop(0, end_time)
                    apply_montage_and_interpolate(raw)
                    set_average_reference(raw)
                    raw = bandpass_filter(raw, phase='zero')
                    raw_eeg = raw.copy().pick('eeg')
                    # 与 02_preprocess 一致: ICA 在 0.5-30 Hz 带通信号上拟合
                    # (bandpass 已含 0.5 Hz 高通), 不再额外高通避免双重滤波。
                    raw_eeg_hp = raw_eeg.copy()

                    # 对每个 ICA 成分数拟合并扫描 (验证 CFA 分类对成分数的稳健性)
                    for n_comp in N_COMPONENTS:
                        ica = fit_ica(raw_eeg_hp, n_components=n_comp, random_state=42)

                        ratios = compute_energy_ratios(ica, raw_eeg_hp, rpeaks, end_time, srate)
                        if ratios is None:
                            continue
                        eye_idx = detect_eye_idx(ica, raw_eeg)

                        n_rec += 1
                        # 导出每个成分的锁时能量比 (连续谱证据)
                        ratio_rows.append({
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses],
                            "n_components": n_comp,
                            **{f"comp_{i:02d}": round(float(ratios[i]), 6) for i in range(ica.n_components_)},
                            "n_eye": len(eye_idx),
                        })
                        # eyeonly = 只去眼动, 与 CFA 阈值无关 (不依赖 thr), 只算一次。
                        # (原实现放在阈值循环内重复 apply 7 次, 结果逐位相同, 纯冗余)
                        ica_eye = ica.copy()
                        ica_eye.exclude = list(eye_idx)
                        eyeonly = ica_eye.apply(raw_eeg.copy(), verbose=False)
                        rms_eye, n_ep_eye = extract_roi_rms(eyeonly, rpeaks, end_time, srate, roi_chans)

                        # 对每个阈值重分类 CFA, 重算 clean RMS
                        for thr in THRESHOLDS:
                            cfa_idx = set(i for i, r in enumerate(ratios)
                                          if i not in eye_idx and r > thr)
                            # clean = 去眼动+去CFA
                            ica_clean = ica.copy()
                            ica_clean.exclude = list(eye_idx) + list(cfa_idx)
                            clean = ica_clean.apply(raw_eeg.copy(), verbose=False)

                            rms_clean, n_ep_clean = extract_roi_rms(clean, rpeaks, end_time, srate, roi_chans)
                            if rms_clean is None or rms_eye is None:
                                continue
                            rows.append({
                                "sub": sub, "ses": ses, "stim": stim,
                                "condition": SESSIONS[ses],
                                "n_components": n_comp,
                                "threshold": thr,
                                "n_cfa": len(cfa_idx),
                                "n_epochs_kept": n_ep_clean,
                                "rms_clean": round(rms_clean, 5),
                                "rms_eyeonly": round(rms_eye, 5),
                                "delta_uv": round(rms_eye - rms_clean, 5),
                            })
                except Exception as e:
                    print(f"[ERROR] {sub} {ses} {stim}: {e}")
                    continue

        print(f"完成 {sub}: 累计 {len(rows)} 条扫描记录")

    if rows:
        fn = ["sub", "ses", "stim", "condition", "n_components", "threshold", "n_cfa",
              "n_epochs_kept", "rms_clean", "rms_eyeonly", "delta_uv"]
        with open(OUT / "cfa_threshold_scan.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    print(f"\n总计 {n_rec} 条记录 × {len(N_COMPONENTS)} 成分数 × {len(THRESHOLDS)} 阈值 = {len(rows)} 行")
    print("→ 已写入 data/cfa_threshold_scan.csv")
    # 注: 不再写 cfa_energy_ratio_distribution.csv (该文件由 06c_energy_ratio_only.py
    # 以单成分数 40 逐记录增量写出, 供 06d 分布检验)。06b 只负责阈值扫描产物。


if __name__ == "__main__":
    run_all_or_once(main)
