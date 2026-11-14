#!/usr/bin/env python3
"""
补充分析：共同 epoch 版本的双轨 HEP 比较（回应审稿意见 4）。

背景：主分析 03b_hep_dual.py 对 clean 与 eyeonly 两条轨迹【分别】做振幅拒绝
（峰峰值 > 50 μV 剔除），导致两条轨迹可能保留不同的心搏 epoch——eyeonly 因保留
CFA 幅度更大，部分 epoch 在 eyeonly 超阈值被剔除、却在 clean 被保留，反之亦然。
这样 clean−eyeonly 的 RMS 差就不再仅仅是 ICA 去留的信号差异，还混入了 epoch
样本构成的差异。

本脚本改用【共同 epoch】方案：以 eyeonly 轨迹（仅去眼动、保留 CFA，最接近原始
信号）作为统一参考，确定共同有效 epoch（峰峰值 ≤ 50 μV），再把这同一 keep-mask
同时施加到 clean 与 eyeonly，使两条轨迹使用完全相同的 epoch 集合。

产出：
  - data/experiment{X}/hep_dual_common_epoch.csv   共同 epoch 的双轨 RMS
  - data/experiment{X}/epoch_overlap_summary.csv   逐记录的两轨保留 epoch 数与重叠比例

用法：python supp_common_epoch.py --all （或 BBBD_EXPERIMENT=2 ...）
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, EARLY_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI,
    read_rpeak, filter_rr, roi_idx, get_end_time, get_session_stims,
    reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)


def build_epochs(fif_path, rpeaks, end_time):
    """读一次 fif，构造 (n_ep, n_ch, n_t) 的 epoch 数组（μV），不做拒绝。"""
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw = raw.copy().crop(0.0, end_time)
    ch_names = raw.ch_names
    srate = float(raw.info["sfreq"])
    valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
    rp = rpeaks[valid]
    rp = rp[filter_rr(rp)]
    if len(rp) < 5:
        return None
    data = raw.get_data() * 1e6  # μV
    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    starts = np.round((rp + TMIN) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)
    return epochs, ch_names, t


def extract_common_hep(clean_fif, eye_fif, rpeaks, end_time):
    """以 eyeonly 为统一参考确定共同 epoch，并对两轨施加同一 keep-mask。

    返回 (clean_hep_bl, eye_hep_bl, n_common, n_total, n_clean_only, n_eye_only)。
    """
    c = build_epochs(clean_fif, rpeaks, end_time)
    e = build_epochs(eye_fif, rpeaks, end_time)
    if c is None or e is None:
        return None
    c_ep, c_ch, t = c
    e_ep, e_ch, _ = e
    # 两轨应来自同一记录，通道与 epoch 数应一致（都以同一 rpeaks 构造）
    if c_ep.shape[0] != e_ep.shape[0]:
        # 理论上不会发生；若发生则以较少的对齐
        n = min(c_ep.shape[0], e_ep.shape[0])
        c_ep, e_ep = c_ep[:n], e_ep[:n]
    n_total = c_ep.shape[0]

    # 共同参考：eyeonly 轨迹的峰峰值拒绝（返回保留 epoch 的索引数组）
    keep_common = reject_epochs_by_amplitude(e_ep, EPOCH_REJECT_UV)

    # 两轨各自独立拒绝（用于报告重叠）
    keep_clean = reject_epochs_by_amplitude(c_ep, EPOCH_REJECT_UV)
    keep_eye = keep_common

    n_common = int(len(keep_common))
    # 独立拒绝下各自保留集合，与共同集合的重叠统计
    clean_set = set(int(i) for i in keep_clean)
    eye_set = set(int(i) for i in keep_eye)
    common_set = set(int(i) for i in keep_common)
    n_clean_only = len(clean_set - common_set)
    n_eye_only = len(eye_set - common_set)
    # 两轨各自独立保留集合的交集与并集（用于报告真正的"重叠比例"）
    n_clean_kept = len(clean_set)
    n_eye_kept = len(eye_set)
    inter = clean_set & eye_set
    union = clean_set | eye_set
    jaccard = len(inter) / len(union) if union else 0.0

    def hep_from(epochs, keep):
        ep = epochs[keep]
        if len(ep) < 5:
            return None
        hep = ep.mean(axis=0)  # (n_ch, n_t)
        base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
        baseline = hep[:, base_mask].mean(axis=1, keepdims=True)
        return hep - baseline, hep

    cbl = hep_from(c_ep, keep_common)
    ebl = hep_from(e_ep, keep_common)
    if cbl is None or ebl is None:
        return None
    return (cbl[0], cbl[1], ebl[0], ebl[1], c_ch, t,
            n_common, n_total, n_clean_kept, n_eye_kept, jaccard)


def rms_from_hep(hep_bl, hep_raw, ch_names, roi_chans, t):
    ridx = roi_idx(ch_names, roi_chans)
    if len(ridx) == 0:
        return None
    main_mask = (t >= MAIN_WIN[0]) & (t <= MAIN_WIN[1])
    early_mask = (t >= EARLY_WIN[0]) & (t <= EARLY_WIN[1])
    out = {}
    for win_name, mask in [("main_200_600", main_mask), ("early_0_200", early_mask)]:
        vals_raw = hep_raw[np.ix_(ridx, np.where(mask)[0])]
        vals_bl = hep_bl[np.ix_(ridx, np.where(mask)[0])]
        out[f"{win_name}_mean_uv"] = float(np.mean(vals_bl))
        out[f"{win_name}_mean_uv_raw"] = float(np.mean(vals_raw))
        out[f"{win_name}_rms_uv"] = float(np.sqrt(np.mean(vals_bl ** 2)))
        out[f"{win_name}_rms_uv_raw"] = float(np.sqrt(np.mean(vals_raw ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sub", type=str, default=None)
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES)
    args = ap.parse_args()

    exp_id = args.experiment or "2"
    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    elif args.sub:
        subs = [args.sub]
    else:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)[:5]

    rows = []
    overlap_rows = []
    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                rpeaks = read_rpeak(sub, ses, stim)
                if rpeaks is None or len(rpeaks) == 0:
                    continue
                end_time = get_end_time(sub, ses, stim)
                if end_time is None:
                    continue
                clean_fif = PROC / sub / ses / f"{stim}_clean_raw.fif"
                eye_fif = PROC / sub / ses / f"{stim}_eyeonly_raw.fif"
                if not (clean_fif.is_file() and eye_fif.is_file()):
                    continue
                r = extract_common_hep(clean_fif, eye_fif, rpeaks, end_time)
                if r is None:
                    continue
                (c_bl, c_raw, e_bl, e_raw, ch_names, t,
                 n_common, n_total, n_clean_kept, n_eye_kept, jaccard) = r
                overlap_rows.append({
                    "sub": sub, "ses": ses, "stim": stim,
                    "condition": SESSIONS[ses],
                    "n_total": n_total, "n_common": n_common,
                    "n_clean_kept": n_clean_kept, "n_eye_kept": n_eye_kept,
                    "jaccard_overlap": round(jaccard, 4),
                })
                for roi_name, roi_chans in ROI.items():
                    cr = rms_from_hep(c_bl, c_raw, ch_names, roi_chans, t)
                    er = rms_from_hep(e_bl, e_raw, ch_names, roi_chans, t)
                    if cr is None or er is None:
                        continue
                    for track, rdict in [("clean", cr), ("eyeonly", er)]:
                        row = {
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses], "track": track,
                            "roi": roi_name, "n_epochs": n_common,
                            "n_epochs_before_reject": n_total,
                        }
                        row.update(rdict)
                        rows.append(row)

    data_dir = OUT  # pipeline_data/data/experimentX/
    data_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fn = list(rows[0].keys())
        with open(data_dir / "hep_dual_common_epoch.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    if overlap_rows:
        fn = list(overlap_rows[0].keys())
        with open(data_dir / "epoch_overlap_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(overlap_rows)
    print(f"完成实验 {exp_id}: {len(rows)} 条共同 epoch 双轨记录, {len(overlap_rows)} 条重叠统计")


if __name__ == "__main__":
    main()
