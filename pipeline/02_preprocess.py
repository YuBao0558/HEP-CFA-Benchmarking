#!/usr/bin/env python3
"""
Step 2: EEG 预处理 (双轨) — 滤波/参考/ICA 去眼动+CFA/坏导

目的: 产出"干净"与"未清洗"两套信号, 供后续 HEP 分析双轨使用。

流程 (操作定义 §3.1):
  1. 读 BDF, 设置 CAR 参考
  2. 带通滤波 0.5-30 Hz (主分析零相位 filtfilt; 因果 lfilter 仅作敏感性对照)
  3. 坏导检测 (峰峰值) + 样条插值
   4. ICA 去伪迹:
      - 眼动成分: 额极地形图权重判据 (EYE_TOPO_THRESH)
      - 心电成分(CFA): R 峰锁时能量比判据 (CFA_ENERGY_RATIO_THRESH)
   5. 输出双轨: 去CFA(clean) / 不去CFA(raw_clean_eye_only)

注意: ICLabel 在此环境与 mne 1.6 不兼容, 改用透明的手动判据
      (与 Steinfath 综述 54% 主流做法一致, 更可复现)。

产出:
  - derivatives_bhci/processed/{sub}/{ses}/{stim}_clean_raw.fif   (去眼动+去CFA)
  - derivatives_bhci/processed/{sub}/{ses}/{stim}_eyeonly_raw.fif (只去眼动, 保留CFA)
  - data/preprocess_report.csv   (每记录: 坏导数、剔除成分、解释方差)

用法: python 02_preprocess.py [--sub sub-01] [--all]
"""

import argparse
import csv
import os

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, EXCLUDE, EXPERIMENT_CHOICES, read_rpeak, get_end_time,
    get_session_stims, read_eog, read_ecg, read_sampling_rate_from_json, run_all_or_once,
)
from preprocess_utils import (
    read_eeg_bdf, apply_montage_and_interpolate,
    bandpass_filter, set_average_reference, fit_ica,
    compute_energy_ratios, detect_eye_idx, verify_rpeak_alignment,
    validate_cfa_with_ecg,
    CFA_ENERGY_RATIO_THRESH, ICA_HIGHPASS, BANDPASS_LOW,
)

PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# 注: EOG/ECG 相关仅作为验证列输出 (eye_eog_corr_max / cfa_ecg_corr_max),
#     不参与成分分类, 故无相关阈值常量。


def preprocess(sub, ses, stim):
    raw = read_eeg_bdf(sub, ses, stim)
    rpeaks = read_rpeak(sub, ses, stim)
    end_time = get_end_time(sub, ses, stim)
    if raw is None or rpeaks is None or end_time is None:
        return None
    srate = float(raw.info["sfreq"])

    # 裁剪到有效时长
    raw = raw.copy().crop(0.0, end_time)

    # 设置 montage + 坏导检测 + 插值 (先于平均参考, 避免坏导污染参考)
    bad = apply_montage_and_interpolate(raw)

    # 平均参考
    set_average_reference(raw)

    # 带通滤波, 双轨 (Tanner et al. 2015 解耦策略):
    #   - 主分析信号: 0.1-30 Hz (保留 HEP 慢波, 避免 >=0.3Hz 高通伪迹)
    #   - 零相位 FIR: 主分析 (R 峰对齐 0 ms)。
    #   - 最小相位 FIR (因果): 多宇宙敏感性。
    raw_zero = bandpass_filter(raw.copy(), phase='zero', l_freq=BANDPASS_LOW)
    raw_causal = bandpass_filter(raw.copy(), phase='causal', l_freq=BANDPASS_LOW)

    # ICA 拟合轨 (与主分析滤波解耦): 用 1 Hz 高通去除低频漂移 (Winkler et al.
    # 2015), 拟合 ICA 解混矩阵; 最终施加到 0.1 Hz 主分析信号 (Tanner et al. 2015)。
    raw_eeg = raw_zero.copy().pick('eeg')          # 主分析信号 (0.1-30 Hz)
    raw_ica = bandpass_filter(raw.copy(), phase='zero', l_freq=ICA_HIGHPASS)
    raw_eeg_ica = raw_ica.copy().pick('eeg')        # ICA 拟合信号 (1-30 Hz)
    # ICA 成分数: 默认 50 (≈ 有效通道数 63 的 79%, 避免降维 ICA 损害成分质量;
    # Artoni et al. 2018; Pokorny et al. 2022)。可用环境变量 ICA_N_COMPONENTS 覆盖。
    ica = fit_ica(raw_eeg_ica, n_components=int(os.environ.get("ICA_N_COMPONENTS", "40")),
                  random_state=42)

    # 识别眼动成分: 地形图判据 (前额主导) —— 主判据, 与 06b/06c 一致。
    # 独立 EOG 相关作为验证值 (不参与成分选择), 写入报告供审稿人检查。
    ica_src_hp = ica.get_sources(raw_eeg_ica).get_data()  # (n_comp, n_times) 供相关验证
    eye_idx = detect_eye_idx(ica, raw_eeg)

    # 注意: 1 Hz 轨 (raw_ica / raw_eeg_ica) 不能在此释放——下游 CFA 能量比计算
    # (compute_energy_ratios) 仍需要 raw_eeg_ica。旧版在此处 del 会在 CFA 段抛
    # UnboundLocalError (从未被完整重跑发现)。改为在 preprocess() 末尾统一由
    # Python GC 回收 (单记录 ~12MB, 可接受)。

    # 独立 EOG/ECG 验证: 对 physio 信号做与 ICA 相同的高通(0.5Hz), 再与 ICA 源相关。
    # 目的: 报告每个成分的 EOG/ECG 相关系数 (验证列), 不改变 eye_idx/cfa_idx。
    # 假设: physio 与 EEG 起始时间对齐 (BioSemi 同步采样, 通常成立)。若存在起始
    #       偏移, 相关系数会被削弱 (不会虚高), 属保守方向。
    eog = read_eog(sub, ses, stim)
    ecg = read_ecg(sub, ses, stim)
    eye_eog_corr = [None] * ica.n_components_   # 每成分与 EOG 的最大 |相关|
    cfa_ecg_corr = [None] * ica.n_components_   # 每成分与 ECG 的 |相关|
    eog_skip_reason = ""   # "" / "missing" / "srate_mismatch"
    ecg_skip_reason = ""

    def _highpass(x, fs):
        # 约定 physio 数组为 (n_times, n_chan) 或 (n_times,), axis=0 沿时间轴滤波
        # 高通 0.5 Hz, 与 ICA 拟合的高通一致 (对齐 Steinfath 2026 建议)。
        if x is None or x.ndim == 0 or len(x) < 10:
            return None
        b_hp, a_hp = butter(4, ICA_HIGHPASS / (fs / 2), btype='highpass')
        return filtfilt(b_hp, a_hp, x, axis=0)

    # 采样率对齐: physio 与 EEG 采样率不一致时跳过相关验证 (时间轴不对齐则无意义)。
    # json 缺失时 (None) 假设 physio 与 EEG 同采样率 (BioSemi 通常同步采样)。
    srate_eog_json = read_sampling_rate_from_json(sub, ses, stim, kind="eog")
    srate_ecg_json = read_sampling_rate_from_json(sub, ses, stim, kind="ecg")
    srate_eog_ok = (srate_eog_json is None) or (abs(srate_eog_json - srate) < 1e-6)
    srate_ecg_ok = (srate_ecg_json is None) or (abs(srate_ecg_json - srate) < 1e-6)
    eog_f = None
    ecg_f = None
    if eog is None:
        eog_skip_reason = "missing"
    elif not srate_eog_ok:
        eog_skip_reason = "srate_mismatch"
    else:
        eog_f = _highpass(eog, srate)
    if ecg is None:
        ecg_skip_reason = "missing"
    elif not srate_ecg_ok:
        ecg_skip_reason = "srate_mismatch"
    else:
        ecg_f = _highpass(ecg, srate)
    n_src = ica_src_hp.shape[1]
    if eog_f is not None:
        n = min(eog_f.shape[0], n_src)
        for i in range(ica.n_components_):
            comp = ica_src_hp[i, :n]
            if eog_f.shape[1] > 0:
                eye_eog_corr[i] = max(
                    abs(np.corrcoef(comp, eog_f[:n, c])[0, 1])
                    for c in range(eog_f.shape[1])
                    if np.std(eog_f[:n, c]) > 0
                )
    if ecg_f is not None:
        n = min(ecg_f.shape[0], n_src)
        if np.std(ecg_f[:n]) > 0:
            for i in range(ica.n_components_):
                cfa_ecg_corr[i] = abs(
                    np.corrcoef(ica_src_hp[i, :n], ecg_f[:n])[0, 1])

    # 识别 CFA 成分: 与 R 峰锁时能量比 —— 主判据 (与 06b/06c 一致)。
    cfa_idx = set()
    rp = rpeaks[(rpeaks > 0.1) & (rpeaks < end_time - 0.6)]
    if len(rp) > 10:
        # CFA 识别在 ICA 拟合信号 (1 Hz 高通) 上算锁时能量比 (与分解矩阵同频带),
        # 最终 apply 作用于 0.1-30 Hz 主分析信号 (保留 HEP 慢波)。
        ratios = compute_energy_ratios(ica, raw_eeg_ica, rpeaks, end_time, srate)
        if ratios is not None:
            for i, r in enumerate(ratios):
                if i in eye_idx:
                    continue
                if r > CFA_ENERGY_RATIO_THRESH:
                    cfa_idx.add(i)

    # CFA 成分合并解释方差占比 (跨研究可比, 比"删了几个成分"更有信息量)
    cfa_explained_var = None
    if ica_src_hp is not None and ica_src_hp.shape[1] > 0:
        comp_var = np.var(ica_src_hp, axis=1)  # 每成分方差
        total_var = comp_var.sum()
        if total_var > 0:
            cfa_explained_var = float(comp_var[list(cfa_idx)].sum() / total_var)

    # ECG 正交验证 CFA 判据 (不改变 cfa_idx, 仅写报告供审稿人检查)。
    # overlap_ratio < 0.5 提示锁时能量比判据与 ECG 相关判据不一致, 需人工审查。
    cfa_ecg_overlap = validate_cfa_with_ecg(ica, raw_eeg, cfa_idx, ecg, rpeaks, srate)
    cfa_ecg_overlap_ratio = cfa_ecg_overlap["overlap_ratio"] if cfa_ecg_overlap else None

    # 输出双轨
    eye_exclude = list(eye_idx)
    cfa_exclude = list(cfa_idx)

    # 干净版(去眼动+去CFA) 与 只去眼动版 (均作用于 ICA 拟合的 EEG 通道集)
    ica_clean = ica.copy()
    ica_clean.exclude = eye_exclude + cfa_exclude
    clean = ica_clean.apply(raw_eeg.copy(), verbose=False)

    ica_eyeonly = ica.copy()
    ica_eyeonly.exclude = eye_exclude
    eyeonly = ica_eyeonly.apply(raw_eeg.copy(), verbose=False)

    # 因果版敏感性: 同一套 ICA 成分应用于因果滤波信号 (隔离滤波相位差异)
    raw_eeg_causal = raw_causal.copy().pick('eeg')
    clean_causal = ica_clean.apply(raw_eeg_causal.copy(), verbose=False)
    eyeonly_causal = ica_eyeonly.apply(raw_eeg_causal.copy(), verbose=False)

    # 保存
    sp = PROC / sub / ses
    sp.mkdir(parents=True, exist_ok=True)
    clean.save(sp / f"{stim}_clean_raw.fif", overwrite=True)
    eyeonly.save(sp / f"{stim}_eyeonly_raw.fif", overwrite=True)
    clean_causal.save(sp / f"{stim}_clean_causal_raw.fif", overwrite=True)
    eyeonly_causal.save(sp / f"{stim}_eyeonly_causal_raw.fif", overwrite=True)

    # R 峰对齐验证 (Steinfath 2026 强制校验): 用原始 ECG 信号 epoch 到官方
    # R 峰, 确认 ECG 的 R 波峰值精确落在 0 ms (验证 R 峰时间戳与生理信号对齐,
    # 排除滤波/下采样引入的系统偏移)。lag_ms 越接近 0 越好。
    align = verify_rpeak_alignment(ecg, rpeaks, srate) if ecg is not None else None
    rpeak_lag_ms = align["rpeak_lag_ms"] if align else None

    # 报告
    # 验证列: 眼动成分的最大 EOG 相关, CFA 成分的 ECG 相关 (供审稿人检查判据)
    eye_eog_max = max([eye_eog_corr[i] for i in sorted(eye_idx)
                       if eye_eog_corr[i] is not None], default=None)
    cfa_ecg_max = max([cfa_ecg_corr[i] for i in sorted(cfa_idx)
                       if cfa_ecg_corr[i] is not None], default=None)
    # 地形图判据与 EOG 相关判据的一致性: 眼动成分中 EOG 相关 > 0.3 的比例
    eye_eog_agree = None
    eye_corrs = [eye_eog_corr[i] for i in sorted(eye_idx)
                 if eye_eog_corr[i] is not None]
    if eye_corrs:
        eye_eog_agree = float(np.mean([c > 0.3 for c in eye_corrs]))
    return {
        "sub": sub, "ses": ses, "stim": stim,
        "condition": SESSIONS[ses],
        "n_bad_channels": len(bad),
        "bad_channels": ";".join(bad),
        "n_eye_components": len(eye_idx),
        "eye_components": ",".join(str(i) for i in sorted(eye_idx)),
        "n_cfa_components": len(cfa_idx),
        "cfa_components": ",".join(str(i) for i in sorted(cfa_idx)),
        "eye_eog_corr_max": round(eye_eog_max, 4) if eye_eog_max is not None else "",
        "cfa_ecg_corr_max": round(cfa_ecg_max, 4) if cfa_ecg_max is not None else "",
        "eog_corr_skipped_reason": eog_skip_reason,
        "ecg_corr_skipped_reason": ecg_skip_reason,
        "cfa_explained_var": round(cfa_explained_var, 4) if cfa_explained_var is not None else "",
        "ica_n_iter": int(ica.n_iter_) if hasattr(ica, "n_iter_") and ica.n_iter_ is not None else "",
        "eye_topo_eog_agree_ratio": round(eye_eog_agree, 4) if eye_eog_agree is not None else "",
        "rpeak_lag_ms": rpeak_lag_ms if rpeak_lag_ms is not None else "",
        "cfa_ecg_overlap_ratio": cfa_ecg_overlap_ratio if cfa_ecg_overlap_ratio is not None else "",
    }


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
                try:
                    r = preprocess(sub, ses, stim)
                except Exception as e:
                    print(f"[ERROR] {sub} {ses} {stim}: {e}")
                    r = {
                        "sub": sub, "ses": ses, "stim": stim,
                        "condition": SESSIONS[ses],
                        "n_bad_channels": None,
                        "bad_channels": "",
                        "n_eye_components": None,
                        "eye_components": "",
                        "n_cfa_components": None,
                        "cfa_components": "",
                        "eye_eog_corr_max": "",
                        "cfa_ecg_corr_max": "",
                        "eog_corr_skipped_reason": "",
                        "ecg_corr_skipped_reason": "",
                        "cfa_explained_var": "",
                        "ica_n_iter": "",
                        "eye_topo_eog_agree_ratio": "",
                        "rpeak_lag_ms": "",
                        "cfa_ecg_overlap_ratio": "",
                        "error": str(e),
                    }
                if r:
                    rows.append(r)

    # 全量报告: 覆盖式写出本次所有记录
    report_path = OUT / "preprocess_report.csv"
    fn = ["sub", "ses", "stim", "condition", "n_bad_channels", "bad_channels",
          "n_eye_components", "eye_components", "n_cfa_components", "cfa_components",
          "eye_eog_corr_max", "cfa_ecg_corr_max",
          "eog_corr_skipped_reason", "ecg_corr_skipped_reason", "cfa_explained_var",
          "ica_n_iter", "eye_topo_eog_agree_ratio", "rpeak_lag_ms",
          "cfa_ecg_overlap_ratio", "error"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fn})

    print(f"本次处理 {len(rows)} 条记录")

    # CFA 判据一致性检查 (锁时能量比 vs ECG 相关, 正交验证)。
    # overlap_ratio < 0.5 提示两判据不一致, 需审查 (Virjee 2026 强调 CFA 判据收敛性)。
    overlaps = [float(r["cfa_ecg_overlap_ratio"]) for r in rows
                if r.get("cfa_ecg_overlap_ratio") not in ("", None)]
    if overlaps:
        overlaps = np.array(overlaps)
        n_low = int(np.sum(overlaps < 0.5))
        print(f"\nCFA 判据一致性 (锁时能量比 vs ECG 相关):")
        print(f"  中位 overlap_ratio = {np.median(overlaps):.3f}")
        print(f"  < 0.5 的记录: {n_low}/{len(overlaps)} ({100*n_low/len(overlaps):.1f}%)")
        if n_low > 0.2 * len(overlaps):
            print("  [警告] >20% 记录 CFA 判据与 ECG 验证不一致, "
                  "需审查锁时能量比阈值或考虑替代策略")

    for r in rows[:20]:
        print(f"  {r['sub']} {r['ses']} {r['stim']}: 坏导{r['n_bad_channels']} "
              f"眼动{r['n_eye_components']} CFA{r['n_cfa_components']}")


if __name__ == "__main__":
    run_all_or_once(main)
