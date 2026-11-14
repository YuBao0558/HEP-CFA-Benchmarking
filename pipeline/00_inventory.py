#!/usr/bin/env python3
"""
Step 0: 数据盘点与质量审计 (BBBD experiment2)

产出:
  - data/subjects.csv          被试 × 会话 × 刺激 的完整覆盖表
  - data/availability_matrix.csv  被试 × 模态 可用性矩阵 (每格为存在文件数, 满分 = 会话数 × 刺激数)
  - data/missing_inventory.csv    缺失文件清单
  - data/summary.json             汇总统计

不预设假设, 只做盘点。先跑通单被试(sub-01)再全量。
用法: python 00_inventory.py [--experiment 2|3|4|5]
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np

from common import (
    ROOT, EXP2, OUT, SESSIONS, STIMS, EXPERIMENT_CHOICES, get_session_stims, read_rpeak,
    MIN_RR, BASELINE_WIN, MAIN_WIN, EARLY_WIN, EPOCH_REJECT_UV, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)

# 模态 -> 原始数据文件名匹配规则 (per subject/session/stim)
MODALITIES = {
    "eeg_bdf":       "eeg/{sub}_{ses}_task-{stim}_eeg.bdf",
    "eeg_events":    "eeg/{sub}_{ses}_task-{stim}_events.tsv",
    "ecg_raw":       "beh/{sub}_{ses}_task-{stim}_recording-ecg_physio.tsv.gz",
    "eog_raw":       "beh/{sub}_{ses}_task-{stim}_recording-eog_physio.tsv.gz",
    "gaze":          "eyetrack/{sub}_{ses}_task-{stim}_gaze_visualangle_eyetrack.tsv.gz",
    "pupil":         "eyetrack/{sub}_{ses}_task-{stim}_pupil_eyetrack.tsv.gz",
    "head":          "eyetrack/{sub}_{ses}_task-{stim}_head_eyetrack.tsv.gz",
}
# derivatives 模态
DERIV_MODALITIES = {
    "rpeak":         "beh/{sub}_{ses}_task-{stim}_desc-rpeak_timestamps.tsv.gz",
    "heartrate":     "beh/{sub}_{ses}_task-{stim}_desc-heartrate.tsv.gz",
    "filteredECG":   "beh/{sub}_{ses}_task-{stim}_desc-filteredECG.tsv.gz",
    "blinks":        "eyetrack/{sub}_{ses}_task-{stim}_desc-blinks.tsv.gz",
    "fixations":     "eyetrack/{sub}_{ses}_task-{stim}_desc-fixations.tsv.gz",
    "saccades":      "eyetrack/{sub}_{ses}_task-{stim}_desc-saccades.tsv.gz",
}


def list_subjects():
    # 盘点需要包含缺失被试 (不应用 EXCLUDE), 以便暴露 sub-08/sub-28
    subs = sorted(
        p.name for p in EXP2.glob("sub-*")
        if p.is_dir() and p.name not in (".DS_Store",)
    )
    return subs


def inventory():
    subs = list_subjects()
    rows = []
    missing = []

    for sub in subs:
        for ses in SESSIONS:
            # 检查会话是否存在
            ses_dir = EXP2 / sub / ses
            if not ses_dir.is_dir():
                missing.append({"sub": sub, "ses": ses, "type": "session_missing"})
                continue

            # 每个刺激逐个检查
            for stim in get_session_stims(ses):
                rec = {
                    "sub": sub, "ses": ses, "stim": stim,
                    "condition": SESSIONS[ses],
                }
                # 原始模态
                for mod, pattern in MODALITIES.items():
                    p = EXP2 / sub / ses / pattern.format(sub=sub, ses=ses, stim=stim)
                    rec[mod] = int(p.is_file())

                # derivatives 模态
                for mod, pattern in DERIV_MODALITIES.items():
                    p = EXP2 / "derivatives" / sub / ses / pattern.format(sub=sub, ses=ses, stim=stim)
                    rec[mod] = int(p.is_file())

                rows.append(rec)

                # 记录缺失
                for mod in list(MODALITIES) + list(DERIV_MODALITIES):
                    if rec[mod] == 0:
                        missing.append({"sub": sub, "ses": ses, "stim": stim, "modality": mod})

    # 写 subjects.csv
    fieldnames = ["sub", "ses", "stim", "condition"] + list(MODALITIES) + list(DERIV_MODALITIES)
    with open(OUT / "subjects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 写 availability_matrix.csv (被试 × 模态 可用性矩阵)
    matrix_fieldnames = ["sub"] + list(MODALITIES) + list(DERIV_MODALITIES)
    matrix_rows = []
    for sub in subs:
        sub_rows = [r for r in rows if r["sub"] == sub]
        matrix_row = {"sub": sub}
        for mod in list(MODALITIES) + list(DERIV_MODALITIES):
            matrix_row[mod] = sum(r[mod] for r in sub_rows)
        matrix_rows.append(matrix_row)
    with open(OUT / "availability_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=matrix_fieldnames)
        w.writeheader()
        w.writerows(matrix_rows)

    # 写 missing_inventory.csv (始终创建, 含 type 列)
    with open(OUT / "missing_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sub", "ses", "stim", "modality", "type"])
        w.writeheader()
        for m in missing:
            w.writerow({k: m.get(k, "") for k in ["sub", "ses", "stim", "modality", "type"]})

    # 采样率核对 (分别找第一个成功读取的 EEG / ECG 样例)
    sample_eeg_json = None
    for sample_sub in subs:
        for sample_ses in SESSIONS:
            for sample_stim in get_session_stims(sample_ses):
                j = EXP2 / sample_sub / sample_ses / "eeg" / f"{sample_sub}_{sample_ses}_task-{sample_stim}_eeg.json"
                if j.is_file():
                    with open(j, encoding="utf-8") as f:
                        sample_eeg_json = json.load(f)
                    break
            if sample_eeg_json:
                break
        if sample_eeg_json:
            break
    sample_ecg_json = None
    for sample_sub in subs:
        for sample_ses in SESSIONS:
            for sample_stim in get_session_stims(sample_ses):
                ecg_j = EXP2 / sample_sub / sample_ses / "beh" / f"{sample_sub}_{sample_ses}_task-{sample_stim}_recording-ecg_physio.json"
                if ecg_j.is_file():
                    with open(ecg_j, encoding="utf-8") as f:
                        sample_ecg_json = json.load(f)
                    break
            if sample_ecg_json:
                break
        if sample_ecg_json:
            break

    # RR 间期分布 (抽样前 N 个被试的 rpeak, 供阈值选择合理性评估)
    rr_all = []
    rr_sample_n = min(len(subs), 5)
    for sub in subs[:rr_sample_n]:
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                rp = read_rpeak(sub, ses, stim)
                if rp is not None and len(rp) > 2:
                    rr = np.diff(rp)
                    rr_all.extend(rr.tolist())
    rr_arr = np.array(rr_all) if rr_all else np.array([])
    rr_stats = None
    if len(rr_arr) > 0:
        rr_stats = {
            "n_rr_intervals": int(len(rr_arr)),
            "mean_s": round(float(np.mean(rr_arr)), 3),
            "median_s": round(float(np.median(rr_arr)), 3),
            "pct_lt_0p5_s": round(100 * float(np.mean(rr_arr < 0.5)), 1),
            "pct_lt_0p88_s": round(100 * float(np.mean(rr_arr < 0.88)), 1),
        }

    # 汇总
    summary = {
        "n_subjects": len(subs),
        "n_sessions_expected": len(subs) * 2,
        "n_stims_total": len(STIMS),
        "n_stims_per_session": {s: len(get_session_stims(s)) for s in SESSIONS},
        "total_records": len(rows),
        "total_expected": len(subs) * sum(len(get_session_stims(s)) for s in SESSIONS),
        "modality_completeness": {},
        "eeg_sampling_rate": sample_eeg_json.get("SamplingFrequency") if sample_eeg_json else None,
        "ecg_sampling_rate": sample_ecg_json.get("SamplingFrequency") if sample_ecg_json else None,
        "n_missing_files": sum(1 for m in missing if m.get("type") != "session_missing"),
        "n_missing_sessions": sum(1 for m in missing if m.get("type") == "session_missing"),
        "rr_distribution": rr_stats,
        "methodological_params": {
            "min_rr_s": MIN_RR,
            "baseline_win_s": list(BASELINE_WIN),
            "main_win_s": list(MAIN_WIN),
            "early_win_s": list(EARLY_WIN),
            "epoch_reject_uv": EPOCH_REJECT_UV,
        },
    }
    total_expected = len(subs) * sum(len(get_session_stims(s)) for s in SESSIONS)
    for mod in list(MODALITIES) + list(DERIV_MODALITIES):
        n_have = sum(r[mod] for r in rows)
        summary["modality_completeness"][mod] = {
            "present": n_have,
            "expected": total_expected,
            "pct": round(100 * n_have / total_expected, 1) if total_expected else 0,
        }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)    # 终端打印
    print(f"被试数: {summary['n_subjects']}")
    print(f"总记录(被试×会话×刺激): {summary['total_records']} / 期望 {summary['total_expected']}")
    print(f"缺失文件数: {summary['n_missing_files']}")
    print(f"EEG 采样率: {summary['eeg_sampling_rate']} Hz")
    print(f"ECG 采样率: {summary['ecg_sampling_rate']} Hz")
    if summary["rr_distribution"]:
        rr = summary["rr_distribution"]
        print(f"RR 间期分布 (抽样): 均值 {rr['mean_s']}s, 中位 {rr['median_s']}s, "
              f"<0.5s 占 {rr['pct_lt_0p5_s']}%, <0.88s 占 {rr['pct_lt_0p88_s']}%")
    print("\n各模态完整度:")
    for mod, info in summary["modality_completeness"].items():
        flag = "OK " if info["pct"] == 100 else "GAP"
        print(f"  [{flag}] {mod:15s} {info['present']:3d}/{info['expected']:3d} ({info['pct']}%)")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    ap.parse_args()
    run_all_or_once(inventory)
