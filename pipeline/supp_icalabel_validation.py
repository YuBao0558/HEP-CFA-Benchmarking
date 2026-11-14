#!/usr/bin/env python3
"""补强分析: ICLabel 作为反差对照, 检验通用自动分类器在 BBBD 上能否识别 CFA。

定位 (与主判据的关系, 详见论文 §2.3 / §4.1):
  主判据"R 峰锁时能量富集比"是自定义指标, 次要判据是 ECG 时域相关。ICLabel
  (Pion-Tonachini et al. 2019; MNE-ICALabel Li et al. 2022) 是 EEG 领域最广泛
  使用的自动 IC 分类器。本脚本把它作为**反差对照**(而非第三方验证): 检验 ICLabel
  在 BBBD 这一具体数据上对"heart beat"类的识别能力, 以论证"通用自动分类器无法
  胜任微弱容积传导 CFA 的识别, 因此需要心跳锁时的专门判据"。

  ICLabel 训练域为高采样率(≥256 Hz)、1-100 Hz 频谱的 EEG, 其 "heart beat" 类
  主要针对大幅度 ECG 伪迹; BBBD 128 Hz 采样下的 CFA 是微弱、容积传导的心跳锁时
  信号, 超出 ICLabel 训练域。预注册预期: ICLabel 在 BBBD 上对 "heart beat" 类
  近乎零识别。

方法 (与主管线 02_preprocess 一致, 复用同一 ICA 成分数):
  对每个实验的全部有效被试 (排除 EXCLUDE), 复用预处理(读 BDF → 坏导插值 → CAR →
  1 Hz 高通 → ICA), 然后:
    - 能量富集比 > 0.3 标记为 CFA (自定义主判据)
    - ICLabel 7 类分类 (brain/muscle/eye/heart/line/channel/other)
  记录每个非眼动成分的 "ICLabel 类 × 富集比二分类(>0.3 vs ≤0.3)" 完整混淆矩阵。

  ICA 成分数与主管线一致, 由环境变量 ICA_N_COMPONENTS 控制(默认 40, 见
  02_preprocess.py)。**成分数变更后需与本脚本同步重跑。**

产出:
  - pipeline/pipeline_data/data/experimentX/icalabel_validation.csv   (逐成分)
  - pipeline/pipeline_data/data/icalabel_validation.csv               (合并, 含 experiment 列)
  - pipeline/pipeline_data/data/icalabel_validation_summary.csv       (每实验汇总)

用法:
  python supp_icalabel_validation.py --all --experiment all     # 4 实验全跑
  python supp_icalabel_validation.py --all --experiment 2       # 单实验
  ICA_N_COMPONENTS=40 python supp_icalabel_validation.py --all --experiment all
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from common import (
    EXP2, OUT, SESSIONS, EXCLUDE, EXPERIMENT_ID, EXPERIMENT_CHOICES,
    read_rpeak, get_end_time, get_session_stims, run_all_or_once,
)
from preprocess_utils import (
    read_eeg_bdf, apply_montage_and_interpolate, bandpass_filter,
    set_average_reference, fit_ica, compute_energy_ratios, detect_eye_idx,
    CFA_ENERGY_RATIO_THRESH, ICA_HIGHPASS,
)

BASE = Path(__file__).resolve().parent.parent / "pipeline" / "pipeline_data"
DATA_ROOT = BASE / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ICLabel 的 "heart beat" 类标签 (与 mne_icalabel 0.4 ICLABEL_NUMERICAL_TO_STRING 一致)。
HEART_LABEL = "heart beat"

# ICA 成分数: 与主管线 02_preprocess 一致 (默认 40, 可环境变量覆盖)。
N_COMPONENTS = int(os.environ.get("ICA_N_COMPONENTS", "40"))


def collect_rows(subs, label_components):
    """对给定被试跑 ICLabel, 返回逐成分 rows 列表。"""
    rows = []
    for sub in subs:
        n_rec = 0
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
                    raw_ica = bandpass_filter(raw.copy(), phase="zero", l_freq=ICA_HIGHPASS)
                    raw_eeg_ica = raw_ica.copy().pick("eeg")
                    ica = fit_ica(raw_eeg_ica, n_components=N_COMPONENTS, random_state=42)

                    ratios = compute_energy_ratios(ica, raw_eeg_ica, rpeaks, end_time, srate)
                    if ratios is None:
                        continue
                    eye_idx = detect_eye_idx(ica, raw_eeg_ica)

                    # ICLabel 分类 (需 montage; 已在 apply_montage_and_interpolate 设置)。
                    try:
                        result = label_components(raw_eeg_ica, ica, method="iclabel")
                        ic_labels = result["labels"]
                        ic_proba = result["y_pred_proba"]
                        heart_score = np.where(
                            np.array(ic_labels) == HEART_LABEL, ic_proba, 0.0)
                    except Exception as e:
                        print(f"[ICLabel失败] {sub} {ses} {stim}: {e}")
                        continue

                    n_rec += 1
                    for i in range(ica.n_components_):
                        rows.append({
                            "experiment": EXPERIMENT_ID,
                            "sub": sub, "ses": ses, "stim": stim,
                            "component": i,
                            "energy_ratio": round(float(ratios[i]), 5),
                            "iclabel_label": ic_labels[i],
                            "iclabel_heart_score": round(float(heart_score[i]), 5),
                            "is_eye": i in eye_idx,
                            "cfa_by_ratio": (i not in eye_idx
                                             and ratios[i] > CFA_ENERGY_RATIO_THRESH),
                            "cfa_by_icalabel": (i not in eye_idx
                                                and ic_labels[i] == HEART_LABEL),
                        })
                except Exception as e:
                    print(f"[ERROR] {sub} {ses} {stim}: {e}")
                    continue
        print(f"[{EXPERIMENT_ID}] 完成 {sub} ({n_rec} 记录)", flush=True)
    return rows


def summarize(rows):
    """对逐成分 rows 做每实验汇总: heart beat 识别数、other 但富集比>0.3 等。"""
    n_comp = len(rows)
    n_non_eye = sum(1 for r in rows if not r["is_eye"])
    n_heart = sum(1 for r in rows if not r["is_eye"] and r["iclabel_label"] == HEART_LABEL)
    n_ratio = sum(1 for r in rows if r["cfa_by_ratio"])
    n_other_ratio = sum(1 for r in rows if not r["is_eye"]
                        and r["iclabel_label"] == "other" and r["cfa_by_ratio"])

    # 每被试的 CFA 候选(富集比>0.3)分布。
    sub_cfa = {}
    for r in rows:
        if r["cfa_by_ratio"]:
            sub_cfa[r["sub"]] = sub_cfa.get(r["sub"], 0) + 1

    return {
        "experiment": EXPERIMENT_ID,
        "n_subjects": len(set(r["sub"] for r in rows)),
        "n_records": len(set((r["sub"], r["ses"], r["stim"]) for r in rows)),
        "n_components": n_comp,
        "n_non_eye": n_non_eye,
        "n_heart_beat": n_heart,
        "heart_beat_rate": round(n_heart / n_non_eye, 5) if n_non_eye else 0.0,
        "n_cfa_by_ratio": n_ratio,
        "n_other_ratio": n_other_ratio,
        "other_ratio_rate": round(n_other_ratio / n_ratio, 5) if n_ratio else 0.0,
        "n_subs_with_cfa": len(sub_cfa),
        "cfa_candidates_per_sub": sub_cfa,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="跑全部有效被试(否则仅前 5 名)")
    ap.add_argument("--n_subs", type=int, default=None, help="限制被试数(测试用)")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部)")
    args = ap.parse_args()

    from mne_icalabel import label_components

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)[:5]
    if args.n_subs:
        subs = subs[:args.n_subs]

    rows = collect_rows(subs, label_components)

    fn = ["experiment", "sub", "ses", "stim", "component", "energy_ratio",
          "iclabel_label", "iclabel_heart_score", "is_eye",
          "cfa_by_ratio", "cfa_by_icalabel"]

    # 逐实验写回 (与主管线其他产出隔离)。
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "icalabel_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)

    # 合并文件 (追加模式, 便于跨实验汇总)。
    combined = DATA_ROOT / "icalabel_validation.csv"
    write_header = not combined.is_file() or combined.stat().st_size == 0
    with open(combined, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        if write_header:
            w.writeheader()
        w.writerows(rows)

    # 每实验汇总。
    summ = summarize(rows)
    print(f"\n=== [{EXPERIMENT_ID}] ICLabel 反差对照汇总 ===")
    print(f"  被试 {summ['n_subjects']} 名 / {summ['n_records']} 记录 / "
          f"{summ['n_components']} 成分 (非眼动 {summ['n_non_eye']})")
    print(f"  ICLabel 'heart beat' 识别数 = {summ['n_heart_beat']} / {summ['n_non_eye']}"
          f" ({summ['heart_beat_rate']:.1%})")
    print(f"  富集比>0.3 的 CFA 候选 = {summ['n_cfa_by_ratio']} 个")
    print(f"  其中被 ICLabel 标为 'other' 的 = {summ['n_other_ratio']}"
          f" ({summ['other_ratio_rate']:.1%})")
    print(f"  有 CFA 候选的被试 = {summ['n_subs_with_cfa']} / {summ['n_subjects']}")
    print(f"  CFA 候选按被试分布 = {summ['cfa_candidates_per_sub']}")

    # 追加汇总 (带 experiment 列, 汇总 CSV 用单独字段)。
    sf = ["experiment", "n_subjects", "n_records", "n_components", "n_non_eye",
          "n_heart_beat", "heart_beat_rate", "n_cfa_by_ratio", "n_other_ratio",
          "other_ratio_rate", "n_subs_with_cfa"]
    summ_flat = {k: summ[k] for k in sf}
    summary_csv = DATA_ROOT / "icalabel_validation_summary.csv"
    write_header = not summary_csv.is_file() or summary_csv.stat().st_size == 0
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sf)
        if write_header:
            w.writeheader()
        w.writerow(summ_flat)

    print(f"→ 逐成分写入 {OUT / 'icalabel_validation.csv'}")
    print(f"→ 合并文件 {combined}")
    print(f"→ 汇总写入 {summary_csv}")


if __name__ == "__main__":
    run_all_or_once(main)
