#!/usr/bin/env python3
"""
Step 3c: Pseudotrial 校正 HEP (Steinfath et al. 2025, Imaging Neuroscience)

目的: 心跳周期性使 R 峰前基线窗与前一跳重叠, 且 HEP 中混有与心跳无关的
      脑活动 (如 CNV、慢漂移)。用 pseudotrial 校正分离"心跳锁定"活动:

  原理: 在真实 R 峰附近放置伪触发点 (与真实心跳无相位锁定关系),
        提取"伪 HEP", 从真实 HEP 中减去, 得到心跳锁定的纯净 HEP。

方法:
  - 伪 R 峰 = 真实 R 峰 + 随机抖动 (±0.25s), 再按最小 RR 过滤
    (不用"相邻中点"法: 中点序列会继承呼吸性窦性心律不齐的周期波动,
     伪 HEP 可能仍残留心跳锁定活动, 故改用抖动破坏相位锁定)。
  - 真 HEP 与伪 HEP 均做同样的基线校正 + epoch 振幅拒绝 + ROI/窗口 RMS
  - 输出三列: hep_true / hep_pseudo / hep_corrected (真 - 伪)

产出:
  - data/hep_pseudotrial.csv  每被试×条件×track×ROI 的 真/伪/校正后 RMS

用法: python 03c_pseudotrial.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
import mne

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, EARLY_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI, read_rpeak, filter_rr, roi_idx, get_end_time,
    get_session_stims, reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)


def extract_hep_amps(data, ch_names, srate, rp, end_time):
    """提取 R 峰锁定的 HEP 波形 (基线校正), 返回 (hep_bl (n_ch, n_t), t, n_kept)。

    与主分析 03b 一致: 先做 epoch 振幅拒绝 (reject_epochs_by_amplitude),
    再平均 + 基线校正。n_kept 为拒绝后保留的 epoch 数 (供输出报告)。
    """
    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    starts = np.round((rp + TMIN) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
    if len(starts) < 5:
        return None, t, 0
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)
    keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
    epochs = epochs[keep]
    if len(epochs) < 5:
        return None, t, 0
    hep = epochs.mean(axis=0)  # (n_ch, n_t)
    base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    hep_bl = hep - hep[:, base_mask].mean(axis=1, keepdims=True)
    return hep_bl, t, len(epochs)


def roi_window_rms(hep_bl, ridx, t, win):
    mask = (t >= win[0]) & (t <= win[1])
    vals = hep_bl[np.ix_(ridx, np.where(mask)[0])]
    return float(np.sqrt(np.mean(vals ** 2)))


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

                # 真实 R 峰 (最小 RR 过滤 + 时间窗内)
                valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
                rp = rpeaks[valid]
                rp = rp[filter_rr(rp)]
                if len(rp) < 6:
                    continue

                # 伪 R 峰 = 真实 R 峰序列整体平移 (全局时间平移 + 循环折回)。
                # 用固定平移量 (10s) 而非逐峰抖动 (±0.25s), 原因:
                #   1. 逐峰抖动会让伪触发点落在真实心跳的 T 波区 (QT 间期 250-400ms),
                #      伪 HEP 将提取到 T 波 CFA, 相减后注入倒置伪迹;
                #   2. 全局平移彻底解耦心跳相位, 同时保留 RR 间期的统计结构
                #      (Steinfath 2025 的 surrogate 时间平移原理)。
                rng = np.random.default_rng(42 + int(sub.split("-")[1]))
                shift = rng.uniform(5.0, 15.0)   # 每个记录一个 5-15s 随机平移量
                pseudo = np.sort((rp + shift) % end_time)
                pseudo = pseudo[filter_rr(pseudo)]

                # 提取真/伪 HEP (两者均做同样的 epoch 振幅拒绝)
                hep_true, t, n_kept_true = extract_hep_amps(data, ch_names, srate, rp, end_time)
                hep_pseudo, t, n_kept_pseudo = extract_hep_amps(data, ch_names, srate, pseudo, end_time)
                if hep_true is None or hep_pseudo is None:
                    continue

                # pseudotrial 校正: 真 - 伪
                hep_corr = hep_true - hep_pseudo

                for roi_name, roi_chans in ROI.items():
                    ridx = roi_idx(ch_names, roi_chans)
                    if len(ridx) == 0:
                        continue
                    row = {
                        "sub": sub, "ses": ses, "stim": stim,
                        "condition": SESSIONS[ses], "roi": roi_name,
                        "n_epochs_true": len(rp),
                        "n_epochs_true_kept": n_kept_true,
                        "n_epochs_pseudo": len(pseudo),
                        "n_epochs_pseudo_kept": n_kept_pseudo,
                    }
                    for win_name, win in [("main_200_600", MAIN_WIN), ("early_0_200", EARLY_WIN)]:
                        row[f"{win_name}_true"] = round(roi_window_rms(hep_true, ridx, t, win), 5)
                        row[f"{win_name}_pseudo"] = round(roi_window_rms(hep_pseudo, ridx, t, win), 5)
                        row[f"{win_name}_corrected"] = round(roi_window_rms(hep_corr, ridx, t, win), 5)
                    rows.append(row)

        print(f"完成 {sub}: 累计 {len(rows)} 条记录")

    if rows:
        fn = list(rows[0].keys())
        with open(OUT / "hep_pseudotrial.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    print(f"\n总计 {len(rows)} 条 pseudotrial 记录 -> {OUT / 'hep_pseudotrial.csv'}")


if __name__ == "__main__":
    run_all_or_once(main)
