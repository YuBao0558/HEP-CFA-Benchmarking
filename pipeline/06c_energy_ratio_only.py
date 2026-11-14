#!/usr/bin/env python3
"""
Step 6c: 只重算 CFA 能量比分布 (修复 06b 写 energy_ratio 崩溃的补丁)

与 06b 的预处理/ICA 逻辑完全一致, 但只拟合单一成分数 (默认 50, 主分析),
且逐记录增量写盘, 避免内存崩溃/写盘字段不匹配。

产出:
  - data/cfa_energy_ratio_distribution.csv  每记录 × N 成分的锁时能量比

用法: BBBD_EXPERIMENT=3 python 06c_energy_ratio_only.py --all
"""

import argparse
import csv

from common import (
    ROOT, EXP2, OUT, SESSIONS, EXCLUDE, EXPERIMENT_CHOICES, read_rpeak,
    get_end_time, get_session_stims, run_all_or_once,
)
from preprocess_utils import (
    read_eeg_bdf, apply_montage_and_interpolate, bandpass_filter,
    set_average_reference, fit_ica,
    compute_energy_ratios, detect_eye_idx,
)

OUT.mkdir(parents=True, exist_ok=True)

# ICA 成分数: 默认 40 (≈ 数据有效秩 41, 避免降维 ICA 损害成分质量;
# Artoni et al. 2018; Pokorny et al. 2022)。可用环境变量 ICA_N_COMPONENTS 覆盖。
import os as _os
N_COMP = int(_os.environ.get("ICA_N_COMPONENTS", "40"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sub", type=str, default=None)
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

    comp_cols = [f"comp_{i:02d}" for i in range(N_COMP)]
    fn = ["sub", "ses", "stim", "condition", "n_components"] + comp_cols + ["n_eye"]
    out_path = OUT / "cfa_energy_ratio_distribution.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        n_rows = 0
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
                        # 与 02/06b 一致: ICA 在 0.5-30 Hz 带通信号上拟合, 不再额外高通。
                        raw_eeg_hp = raw_eeg.copy()

                        ica = fit_ica(raw_eeg_hp, n_components=N_COMP, random_state=42)

                        ratios = compute_energy_ratios(ica, raw_eeg_hp, rpeaks, end_time, srate)
                        if ratios is None:
                            continue
                        eye_idx = detect_eye_idx(ica, raw_eeg)
                        row = {
                            "sub": sub, "ses": ses, "stim": stim,
                            "condition": SESSIONS[ses],
                            "n_components": N_COMP,
                            **{f"comp_{i:02d}": round(float(ratios[i]), 6) for i in range(N_COMP)},
                            "n_eye": len(eye_idx),
                        }
                        w.writerow({k: row.get(k, "") for k in fn})
                        n_rows += 1
                    except Exception as e:
                        print(f"[ERROR] {sub} {ses} {stim}: {e}")
                        continue
            print(f"完成 {sub}: 累计 {n_rows} 条")
        print(f"→ 已写入 {out_path} ({n_rows} 条记录 × {N_COMP} 成分)")


if __name__ == "__main__":
    run_all_or_once(main)
