#!/usr/bin/env python3
"""
Step 3e: 导出 HEP 头皮地形数据 (Fig1C)

目的: 导出 200-600ms 窗口内各电极的平均 HEP 电压 + 标准 10-20 通道坐标,
      供 R ggplot 绘制头皮地形图 (Fig1C)。

产出:
  - data/hep_topography.csv  每被试×条件×通道的窗口平均电压 (μV)
  - data/channel_positions.csv  标准 10-20 通道坐标 (x, y)

用法: python 03e_topography.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
import mne

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, BASELINE_WIN, EPOCH_REJECT_UV, read_rpeak, filter_rr, get_end_time,
    get_session_stims, reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)


def channel_positions():
    """返回标准 10-20 montage 的通道坐标 (x, y), 用于地形图。"""
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']  # dict: ch -> (x, y, z) 米
    rows = []
    for ch, xyz in pos.items():
        x, y, z = xyz
        rows.append({"channel": ch, "x": round(float(x), 5),
                     "y": round(float(y), 5), "z": round(float(z), 5)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default="sub-01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    # 写通道坐标 (只写一次)
    pos_rows = channel_positions()
    if pos_rows:
        with open(OUT / "channel_positions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["channel", "x", "y", "z"])
            w.writeheader()
            w.writerows(pos_rows)

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = [args.sub]

    rows = []
    for sub in subs:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                fif = PROC / sub / ses / f"{stim}_clean_raw.fif"
                rpeaks = read_rpeak(sub, ses, stim)
                end_time = get_end_time(sub, ses, stim)
                if not fif.is_file() or rpeaks is None or end_time is None:
                    continue

                raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
                raw = raw.copy().crop(0.0, end_time)
                ch_names = raw.ch_names
                srate = float(raw.info["sfreq"])
                data = raw.get_data() * 1e6  # μV

                valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
                rp = rpeaks[valid]
                rp = rp[filter_rr(rp)]
                if len(rp) < 5:
                    continue

                n_t = int((TMAX - TMIN) * srate)
                t = np.arange(n_t) / srate + TMIN
                starts = np.round((rp + TMIN) * srate).astype(int)
                starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
                if len(starts) < 5:
                    continue
                idx_t = starts[:, None] + np.arange(n_t)[None, :]
                epochs = data[:, idx_t].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)

                keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
                epochs = epochs[keep]
                if len(epochs) < 5:
                    continue

                hep = epochs.mean(axis=0)  # (n_ch, n_t)

                # 基线校正 + 主窗口平均
                base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
                hep_bl = hep - hep[:, base_mask].mean(axis=1, keepdims=True)
                main_mask = (t >= MAIN_WIN[0]) & (t <= MAIN_WIN[1])
                main_amp = hep_bl[:, main_mask].mean(axis=1)  # (n_ch,)

                for ch, amp in zip(ch_names, main_amp):
                    rows.append({
                        "sub": sub, "ses": ses, "stim": stim,
                        "condition": SESSIONS[ses],
                        "channel": ch, "amp_uv": round(float(amp), 5),
                    })

        print(f"完成 {sub}: 累计 {len(rows)} 条地形记录")

    if rows:
        fn = list(rows[0].keys())
        with open(OUT / "hep_topography.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    print(f"\n总计 {len(rows)} 条 -> {OUT / 'hep_topography.csv'}")


if __name__ == "__main__":
    run_all_or_once(main)
