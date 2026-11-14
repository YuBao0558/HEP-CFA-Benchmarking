#!/usr/bin/env python3
"""
Step 6: 多宇宙分析 (Level 1 + Level 2)

目的: 系统变化分析参数, 量化 ΔBHCI 状态调制对流程的稳健性 (操作定义 §8)。

冻结策略 (分块降维版, 已定):
  Level 1 (HEP振幅): 参考×ICA×CFA×基线×高通×低通 = 遍历参考/CFA/基线/滤波
  Level 2 (耦合):   指标×时间窗×ROI×参考×ICA×CFA

简化说明: 完整遍历需 Step2 输出多套预处理变体。当前实现基于已有的
  clean/eyeonly 双轨 + 不同 ROI + 不同时间窗 + 不同 min_rr + 不同 epoch 拒绝阈值
  (50/75/100 μV) + 不同基线策略, 作为多宇宙的骨架验证。
  完整参数空间遍历见 multiverse_parameter_space.md 附录 A。

产出:
  - data/multiverse_effects.csv   每 pipeline 的 ΔBHCI (配对 Hedges' dz) + 显著性
  - data/multiverse_params.csv    每 pipeline 的参数指纹

用法: python 06_multiverse.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
import mne
from scipy import stats

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    BASELINE_WIN, ROI as ROIS, read_rpeak, filter_rr, get_end_time,
    BASE_SESSION, CONTRAST_SESSION, get_session_stims, fdr_bh,
    reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)

# 多宇宙维度
# track: CFA 去/不去 (clean=去CFA, eyeonly=保留CFA)
# filt: 滤波方式 (zerophase=零相位 FIR 主分析, causal=最小相位 FIR 因果敏感性)
#   依据 Steinfath 2026: 因果滤波引入群延迟, 零相位保持 R 峰对齐; 两者作为多宇宙
#   维度量化"未来信息泄漏"对 HEP 的实际影响 (Box 2 待解问题)。
TRACKS = [("clean", "zerophase"), ("clean", "causal"),
          ("eyeonly", "zerophase"), ("eyeonly", "causal")]
WINDOWS = {"main_200_600": (0.2, 0.6), "early_0_200": (0.0, 0.2)}
# 最小 RR 间期 (Virjee 综述: 敏感性分析中贡献 97% 效应; 0.5 最宽松, 0.88 为模态)
# 注: 0.88s 会剔除心率>68bpm 的 epoch, 故加 0.7/0.75 中间值形成平滑敏感性曲线。
MIN_RRS = [0.5, 0.65, 0.7, 0.75, 0.8, 0.88]
# 基线策略 (Steinfath 2026: 基线窗可能与前一心跳晚期成分重叠, 建议多基线对照)
#   corrected:  R 峰前 -100~-25ms 基线 (Coll 2021, 避开 R 峰起始)
#   raw:        无基线校正 (仅靠 0.5Hz 带通去趋势)
#   wholecycle: 全 epoch 均值基线 (全心动周期基线, 避免 P 波污染)
BASELINE_MODES = ["corrected", "raw", "wholecycle"]
# epoch 振幅拒绝阈值敏感性 (μV): 主分析冻结 100 μV, 50/75 μV 作敏感性
# (Virjee 综述: 振幅拒绝阈值是 HEP 第二大影响参数, 需报告阈值敏感性)。
EPOCH_REJECT_UVS = [50.0, 75.0, 100.0]


def load_hep_data(fif, end_time):
    """读一次 fif, 缓存原始数据, 返回 (data μV, ch_names, srate, t) 或 None。

    数据与 min_rr / 拒绝阈值无关, 故只读一次; 不同参数仅影响后续过滤/提取。
    """
    if not fif.is_file():
        return None
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    raw = raw.copy().crop(0.0, end_time)
    ch_names = raw.ch_names
    srate = float(raw.info["sfreq"])
    data = raw.get_data() * 1e6
    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    return data, ch_names, srate, t


def extract_hep_waveform(data, ch_names, srate, t, rpeaks, end_time, min_rr, reject_uv):
    """从已读入的 data 提取 HEP 波形 (按 min_rr 过滤 rpeaks, 按 reject_uv 拒绝 epoch)。

    返回 (hep_raw, hep_bl, hep_wc) 或 None。
    """
    valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
    rp = rpeaks[valid]
    rp = rp[filter_rr(rp, min_rr)]
    if len(rp) < 5:
        return None

    n_t = len(t)
    starts = np.round((rp + TMIN) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= data.shape[1])]
    idx_t = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx_t].transpose(1, 0, 2)

    # epoch 振幅拒绝: 剔除峰峰值 > 阈值的伪迹 epoch (Virjee 第二大影响参数)
    keep = reject_epochs_by_amplitude(epochs, reject_uv)
    epochs = epochs[keep]
    if len(epochs) < 5:
        return None

    hep = epochs.mean(axis=0)

    # R 峰前基线校正
    base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    hep_bl = hep - hep[:, base_mask].mean(axis=1, keepdims=True)
    # 全 epoch 均值基线校正
    hep_wc = hep - hep.mean(axis=1, keepdims=True)
    return hep, hep_bl, hep_wc


def rms_from_waveform(hep_bl, ch_names, roi_chans, win, t):
    """从已提取的 HEP 波形计算 ROI×窗口 RMS。"""
    ch_idx = {c: i for i, c in enumerate(ch_names)}
    idx = [ch_idx[c] for c in roi_chans if c in ch_idx]
    if not idx:
        return None
    mask = (t >= win[0]) & (t <= win[1])
    vals = hep_bl[np.ix_(idx, np.where(mask)[0])]
    return float(np.sqrt(np.mean(vals ** 2)))


def hedges_dz(a, b):
    """配对样本 Hedges' dz (corrected Cohen's dz)。

    a, b 为同一批被试的两种条件 (配对设计), 故用差值均值的效应量,
    而非独立样本的 Hedges' g。
    """
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2:
        return np.nan
    d_sd = np.std(d, ddof=1)
    if d_sd == 0:
        return np.nan
    dz = np.mean(d) / d_sd
    # Hedges 小样本无偏修正 (配对设计, df = n-1)
    correction = 1 - 3 / (4 * (n - 1) - 1)
    return dz * correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES,
                    help="实验标识 (2/3/4/5, 或 all 依次跑全部; 默认读环境变量 BBBD_EXPERIMENT, 再默认 2)")
    args = ap.parse_args()

    if args.all:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in EXCLUDE)
    else:
        subs = [args.sub] if args.sub else sorted(p.name for p in EXP2.glob("sub-*")
                                                   if p.is_dir() and p.name not in EXCLUDE)[:3]

    # 收集每个 pipeline 的每被试两条件 BHCI 值
    effects = []
    params = []
    sub_effects = []   # 缺口一: pipeline × subject 级配对差值 (供层次模型/聚类稳健)
    pipeline_id = 0

    for track, filt in TRACKS:
        # fif 文件名: 零相位 = {stim}_{track}_raw.fif, 因果 = {stim}_{track}_causal_raw.fif
        suffix = "causal_raw" if filt == "causal" else "raw"

        # 先按 (sub, ses, stim) 读一次 fif, 缓存原始 data (与 min_rr/阈值无关),
        # 再按 min_rr + 拒绝阈值过滤/提取波形 (避免 6×3 组合重复读 fif)。
        data_cache = {}   # (sub, ses, stim) -> (data, ch_names, srate, t) 或 None
        wave_cache = {}   # (sub, ses, stim, min_rr, reject_uv) -> (hep_raw, hep_bl, hep_wc) 或 None
        for sub in subs:
            for ses in SESSIONS:
                for stim in get_session_stims(ses):
                    fif = PROC / sub / ses / f"{stim}_{track}_{suffix}.fif"
                    rpeaks = read_rpeak(sub, ses, stim)
                    end = get_end_time(sub, ses, stim)
                    loaded = load_hep_data(fif, end) if (rpeaks is not None and end is not None) else None
                    if loaded is None:
                        data_cache[(sub, ses, stim)] = None
                        for min_rr in MIN_RRS:
                            for reject_uv in EPOCH_REJECT_UVS:
                                wave_cache[(sub, ses, stim, min_rr, reject_uv)] = None
                        continue
                    data, ch_names, srate, t = loaded
                    data_cache[(sub, ses, stim)] = loaded
                    for min_rr in MIN_RRS:
                        for reject_uv in EPOCH_REJECT_UVS:
                            wave_cache[(sub, ses, stim, min_rr, reject_uv)] = \
                                extract_hep_waveform(data, ch_names, srate, t, rpeaks, end, min_rr, reject_uv)

        for min_rr in MIN_RRS:
            for reject_uv in EPOCH_REJECT_UVS:
                for baseline in BASELINE_MODES:
                    for roi_name, roi_chans in ROIS.items():
                        for win_name, win in WINDOWS.items():
                            pipeline_id += 1
                            # 用 dict 按被试 ID 存储, 保证配对对齐
                            sub_base = {}    # sub -> 基准会话均值
                            sub_cont = {}    # sub -> 对照会话均值
                            for sub in subs:
                                for ses in SESSIONS:
                                    vals = []
                                    for stim in get_session_stims(ses):
                                        res = wave_cache.get((sub, ses, stim, min_rr, reject_uv))
                                        if res is None:
                                            continue
                                        hep_raw, hep_bl, hep_wc = res
                                        ch_names = data_cache[(sub, ses, stim)][1]
                                        t = data_cache[(sub, ses, stim)][3]
                                        hep_use = {
                                            "corrected": hep_bl,
                                            "raw": hep_raw,
                                            "wholecycle": hep_wc,
                                        }[baseline]
                                        rms = rms_from_waveform(hep_use, ch_names, roi_chans, win, t)
                                        if rms is not None:
                                            vals.append(rms)
                                    if vals:
                                        if ses == BASE_SESSION:
                                            sub_base[sub] = np.mean(vals)
                                        elif ses == CONTRAST_SESSION:
                                            sub_cont[sub] = np.mean(vals)

                            # 只保留两个条件都有的被试, 按 ID 配对
                            common_subs = sorted(set(sub_base) & set(sub_cont))
                            a = [sub_base[s] for s in common_subs]
                            b = [sub_cont[s] for s in common_subs]
                            # 被试级配对差值 (缺口一: pipeline 非独立性建模所需的原始数据)
                            # diff_i = b_i - a_i, 每个 common_sub 一个值。
                            for si, s in enumerate(common_subs):
                                sub_effects.append({
                                    "pipeline_id": pipeline_id,
                                    "sub": s,
                                    "track": track, "filt": filt, "min_rr": min_rr,
                                    "epoch_reject_uv": reject_uv,
                                    "baseline": baseline, "roi": roi_name, "window": win_name,
                                    "base_uv": float(a[si]),
                                    "contrast_uv": float(b[si]),
                                    "diff_uv": float(b[si] - a[si]),
                                })
                            # hedges_dz(b, a) 使效应量方向 = contrast − base,
                            # 与 mean_diff_uv (= b - a)、ttest_rel(b, a)、LMM condition 系数一致。
                            g = hedges_dz(b, a)
                            # 配对 t 检验 (CONTRAST vs BASE)
                            p_val = np.nan
                            if len(a) >= 3:
                                try:
                                    _, p_val = stats.ttest_rel(b, a)
                                except Exception:
                                    p_val = np.nan
                            effects.append({
                                "pipeline_id": pipeline_id,
                                "track": track, "filt": filt, "min_rr": min_rr,
                                "epoch_reject_uv": reject_uv,
                                "baseline": baseline, "roi": roi_name, "window": win_name,
                                "hedges_dz": round(g, 4) if not np.isnan(g) else None,
                                "n_subjects": len(common_subs),
                                "mean_base_uv": round(float(np.mean(a)), 4) if a else None,
                                "mean_contrast_uv": round(float(np.mean(b)), 4) if b else None,
                                "mean_diff_uv": round(float(np.mean(b) - np.mean(a)), 4) if a else None,
                                "sd_diff_uv": round(float(np.std(np.array(b) - np.array(a), ddof=1)), 4)
                                               if len(a) >= 2 else None,
                                "p_val": round(float(p_val), 5) if not np.isnan(p_val) else None,
                            })
                            params.append({
                                "pipeline_id": pipeline_id,
                                "track": track, "filt": filt, "min_rr": min_rr,
                                "epoch_reject_uv": reject_uv,
                                "baseline": baseline, "roi": roi_name, "window": win_name,
                            })

    if effects:
        # FDR 校正 (对全部 pipeline 的 p 值, 全局 BH)
        p_vals = np.array([r["p_val"] if r["p_val"] is not None else np.nan
                           for r in effects], dtype=float)
        fdr = fdr_bh(p_vals)
        for r, fp in zip(effects, fdr):
            r["p_fdr"] = round(float(fp), 5) if not np.isnan(fp) else None
        fn = list(effects[0].keys())
        with open(OUT / "multiverse_effects.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(effects)
    if params:
        fn = list(params[0].keys())
        with open(OUT / "multiverse_params.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(params)

    if sub_effects:
        fn = list(sub_effects[0].keys())
        with open(OUT / "multiverse_subject_effects.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(sub_effects)

    print(f"完成 {len(effects)} 个 pipeline")
    for e in effects:
        print(f"  P{e['pipeline_id']}: {e['track']:8s} {e['filt']:9s} rr={e['min_rr']:.2f} "
              f"rej={e['epoch_reject_uv']:.0f} {e['baseline']:10s} "
              f"{e['roi']:14s} {e['window']:14s} -> "
              f"dz={e['hedges_dz']} p={e['p_val']} (FDR {e['p_fdr']})")


if __name__ == "__main__":
    run_all_or_once(main)
