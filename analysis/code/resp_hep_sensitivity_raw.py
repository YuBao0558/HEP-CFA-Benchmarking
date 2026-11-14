#!/usr/bin/env python3
"""呼吸相位调制 HEP 的原始数据级稳健性分析（第 1/2 项）。

严格复用 pipeline 的读取约定 (common.py) 与 03d_singletrial.py 的提取逻辑，
但对 HEP 窗口、基线校正、额外伪迹协变量做参数化扫描：

  第 1 项：窗口/基线敏感性 (specification curve)
    - 扫描 HEP 主窗口 ∈ {(0.2,0.6), (0.25,0.4), (0.2,0.4), (0.25,0.5), (0.3,0.6)}
    - 扫描基线 ∈ {无基线(当前), (-0.1,-0.025), (-0.3,-0.1)}
    - 每组合重算 per-subject sin 系数, 报告符号一致性与 one-sample t p 值。

  第 2 项：呼吸相关非神经伪迹控制
    - 从原始呼吸信号提取两个运动代理:
        (a) 瞬时幅度包络 (|analytic signal|), 呼吸胸带张力随吸/呼的机械运动
        (b) 幅度一阶导数 (变化率), 捕捉胸壁快速运动
    - 将这些代理作为额外协变量加入 circular-linear 模型, 看 sin 项是否仍显著。
    - 非脑参考通道对照: 用颞叶/外围通道 (非 frontocentral ROI) 重算 HEP,
      若呼吸调制只出现在额中央而非外围, 则排除全局运动伪迹。

产出: analysis/data/respiration_coupling/robustness/
  - window_baseline_sensitivity.csv
  - movement_artifact_control.csv
  - peripheral_channel_control.csv

用法: python resp_hep_sensitivity_raw.py --experiment 4 [--sub sub-01]
      默认跑全部被试 (需已生成 *_clean_raw.fif)。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mne
from scipy.signal import hilbert

sys.path.insert(0, str(Path("/Users/yubao/CODE/public/BBBD/pipeline")))
from common import (  # noqa: E402
    ROOT, EXP2, PROC, OUT, SESSIONS, EXCLUDE,
    TMIN, TMAX, EPOCH_REJECT_UV, ROI, read_rpeak, filter_rr, get_end_time,
    get_session_stims, read_respiration, reject_epochs_by_amplitude,
)
from scipy import stats  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, filtfilt, find_peaks  # noqa: E402


def respiration_phase(sub, ses, stim, srate, n_times):
    """读呼吸信号, 用双阶段峰谷插值提取呼吸相位 (与 03d_singletrial.py 一致)。"""
    resp = read_respiration(sub, ses, stim)
    if resp is None or len(resp) < 10:
        return None
    resp = np.asarray(resp, dtype=float)
    resp = resp - np.mean(resp)
    b, a = butter(3, [0.05 / (srate / 2), 0.5 / (srate / 2)], btype='bandpass')
    resp_f = filtfilt(b, a, resp)
    min_dist = int(1.5 * srate)
    troughs, _ = find_peaks(-resp_f, distance=min_dist)
    peaks, _ = find_peaks(resp_f, distance=min_dist)
    if len(troughs) < 2 or len(peaks) < 1:
        return None
    if len(troughs) > 3:
        trough_int = np.diff(troughs)
        med_int = np.median(trough_int)
        mad_int = np.median(np.abs(trough_int - med_int))
        if mad_int > 0:
            ok_edge = np.abs(trough_int - med_int) < 3 * mad_int * 1.4826
            ok_keep = np.ones(len(troughs), dtype=bool)
            for i in range(1, len(troughs) - 1):
                if not (ok_edge[i - 1] and ok_edge[i]):
                    ok_keep[i] = False
            troughs = troughs[ok_keep]
            if len(troughs) < 2:
                return None
    phase = np.full(len(resp), np.nan)
    for i in range(len(troughs) - 1):
        t0 = troughs[i]
        t1 = troughs[i + 1]
        p_mid = peaks[(peaks > t0) & (peaks < t1)]
        if len(p_mid) != 1:
            seg = np.arange(t0, t1)
            phase[seg] = 2 * np.pi * (seg - t0) / (t1 - t0)
            continue
        p = p_mid[0]
        seg_in = np.arange(t0, p)
        phase[seg_in] = np.pi * (seg_in - t0) / (p - t0)
        seg_ex = np.arange(p, t1)
        phase[seg_ex] = np.pi + np.pi * (seg_ex - p) / (t1 - p)
    first_t0, first_t1 = troughs[0], troughs[1]
    pre = np.arange(0, first_t0)
    if len(pre) > 0:
        phase[pre] = 2 * np.pi * (pre - first_t0) / (first_t1 - first_t0)
    last_t0, last_t1 = troughs[-2], troughs[-1]
    post = np.arange(last_t1, len(resp))
    if len(post) > 0:
        phase[post] = 2 * np.pi * (post - last_t0) / (last_t1 - last_t0)
    phase = np.mod(phase, 2 * np.pi)
    n = min(len(phase), n_times)
    return phase[:n]

ROBUST_OUT = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling/robustness")
ROBUST_OUT.mkdir(parents=True, exist_ok=True)

# 敏感性扫描网格
WIN_GRID = [(0.2, 0.6), (0.25, 0.4), (0.2, 0.4), (0.25, 0.5), (0.3, 0.6)]
BASELINE_GRID = [None, (-0.1, -0.025), (-0.3, -0.1)]

# 颞叶/外围通道 (非 frontocentral), 用作全局运动伪迹对照
PERIPHERAL = ['T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'P9', 'P10', 'PO7', 'PO8']


def extract_hepatic_epochs(sub, ses, stim, win, baseline):
    """按指定窗口/基线提取 (n_ep, n_ch, n_t) epoch 与对应 R 峰/呼吸相位/代理。

    返回 (epochs, rpeak_idx_list, resp_phase_at_r, resp_env_at_r, resp_deriv_at_r)
    或 None (该 stim 不可用)。
    """
    fif = PROC / sub / ses / f"{stim}_clean_raw.fif"
    rpeaks = read_rpeak(sub, ses, stim)
    end_time = get_end_time(sub, ses, stim)
    if not fif.is_file() or rpeaks is None or end_time is None:
        return None

    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    raw = raw.copy().crop(0.0, end_time)
    ch_names = raw.ch_names
    srate = float(raw.info["sfreq"])
    data = raw.get_data() * 1e6

    valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
    rp_all = rpeaks[valid]
    keep_mask = filter_rr(rp_all)
    rp = rp_all[keep_mask]
    if len(rp) < 6:
        return None

    # 前一个 RR 间期 (与 03d_singletrial.py 一致, 基于未 filter_rr 的完整序列)
    rr_all = np.diff(rp_all, prepend=np.nan)
    kept_indices = np.where(keep_mask)[0]
    rr_prev_kept = rr_all[kept_indices]
    if len(kept_indices) > 1:
        consecutive = np.diff(kept_indices) == 1
        prev_valid = np.concatenate([[True], consecutive])
        rr_prev_kept[~prev_valid] = np.nan

    resp = read_respiration(sub, ses, stim)
    resp_phase = respiration_phase(sub, ses, stim, srate, data.shape[1])

    # 呼吸运动代理 (原始信号, 采样率对齐 EEG)
    resp_env = None
    resp_deriv = None
    if resp is not None:
        resp = np.asarray(resp, dtype=float)
        resp = resp - np.mean(resp)
        resp = resp[: data.shape[1]]
        # 瞬时幅度包络
        env = np.abs(hilbert(resp))
        # 一阶导数
        deriv = np.gradient(resp) * srate
        resp_env = env
        resp_deriv = deriv

    n_t = int((TMAX - TMIN) * srate)
    t = np.arange(n_t) / srate + TMIN
    starts = np.round((rp + TMIN) * srate).astype(int)
    ok = (starts >= 0) & (starts + n_t <= data.shape[1])
    starts = starts[ok]
    rp = rp[ok]
    if len(starts) < 5:
        return None
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = data[:, idx].transpose(1, 0, 2)

    keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
    epochs = epochs[keep]
    starts = starts[keep]
    rp_kept = rp[keep]
    rr_prev = rr_prev_kept[keep]
    if len(epochs) < 5:
        return None

    # 基线校正 (None = 不校正)
    if baseline is not None:
        base_mask = (t >= baseline[0]) & (t < baseline[1])
        epochs = epochs - epochs[:, :, base_mask].mean(axis=2, keepdims=True)

    rpeak_idx = np.round(rp_kept * srate).astype(int)
    ph_at_r = (resp_phase[rpeak_idx] if resp_phase is not None
               and np.all(rpeak_idx < len(resp_phase)) else None)
    env_at_r = (resp_env[rpeak_idx] if resp_env is not None
                and np.all(rpeak_idx < len(resp_env)) else None)
    deriv_at_r = (resp_deriv[rpeak_idx] if resp_deriv is not None
                  and np.all(rpeak_idx < len(resp_deriv)) else None)

    return epochs, ch_names, t, rpeak_idx, ph_at_r, env_at_r, deriv_at_r, rr_prev


def hep_from_epochs(epochs, ch_names, t, win, roi="frontocentral"):
    """从 epoch 计算指定 ROI 的单试次 HEP (有符号均值 + RMS)。"""
    if roi == "peripheral":
        fc = [c for c in PERIPHERAL if c in ch_names]
    else:
        fc = [c for c in ROI["frontocentral"] if c in ch_names]
    if len(fc) == 0:
        return None, None
    fc_idx = [ch_names.index(c) for c in fc]
    mask = (t >= win[0]) & (t <= win[1])
    vals = epochs[:, fc_idx, :][:, :, mask].mean(axis=1)
    mean_amp = vals.mean(axis=1)
    rms_amp = np.sqrt(np.mean(vals ** 2, axis=1))
    return mean_amp, rms_amp


def per_subject_sin(amp, ph, rr=None, extra=None):
    """z(amp) ~ sin + cos + rr + [extra] 的 sin 系数 (最小二乘)。"""
    amp = np.asarray(amp, float)
    sd = amp.std()
    if sd == 0 or np.isnan(sd):
        return None, None
    z = (amp - amp.mean()) / sd
    cols = [np.sin(ph), np.cos(ph)]
    if rr is not None:
        cols.append(rr.astype(float))
    if extra is not None:
        for e in extra:
            if e is not None:
                cols.append(e.astype(float) if hasattr(e, "astype") else e)
    X = np.column_stack([np.ones(len(z))] + cols)
    # 剔除含 NaN 的行
    ok = np.all(np.isfinite(X), axis=1)
    if ok.sum() < 5:
        return None, None
    X = X[ok]
    z = z[ok]
    try:
        b, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    return b[1], b[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="4", choices=["4", "5"])
    ap.add_argument("--sub", type=str, default=None)
    args = ap.parse_args()

    # 切换实验需要重设模块常量: 通过环境变量 + 重新加载 common 的方式不可行,
    # 故本脚本直接操作 EXP2 路径映射。
    global EXP2, PROC, OUT
    from common import ROOT as _ROOT
    exp_name = f"experiment{args.experiment}"
    EXP2 = _ROOT / exp_name
    PROC = _ROOT / "pipeline" / "pipeline_data" / "derivatives_bhci" / f"processed_{exp_name}"

    if args.sub:
        subs = [args.sub]
    else:
        subs = sorted(p.name for p in EXP2.glob("sub-*")
                      if p.is_dir() and p.name not in (".DS_Store",) and p.name not in EXCLUDE)

    # ---- 第 1 项: 窗口/基线敏感性 ----
    wb_rows = []
    for win in WIN_GRID:
        for baseline in BASELINE_GRID:
            # 按被试聚合: 合并该被试所有 session/stim 的试次
            sub_amp = {}   # sub -> list of amp
            sub_ph = {}    # sub -> list of phase
            sub_rr = {}    # sub -> list of rr
            for sub in subs:
                amps, phs, rrs = [], [], []
                for ses in SESSIONS:
                    for stim in get_session_stims(ses):
                        r = extract_hepatic_epochs(sub, ses, stim, win, baseline)
                        if r is None:
                            continue
                        (epochs, ch_names, t, rpeak_idx,
                         ph_at_r, env_at_r, deriv_at_r, rr_prev) = r
                        if ph_at_r is None:
                            continue
                        mean_amp, _ = hep_from_epochs(epochs, ch_names, t, win)
                        if mean_amp is None:
                            continue
                        valid = ~np.isnan(ph_at_r) & ~np.isnan(mean_amp)
                        amps.append(mean_amp[valid])
                        phs.append(ph_at_r[valid])
                        rrs.append(rr_prev[valid])
                if amps:
                    sub_amp[sub] = np.concatenate(amps)
                    sub_ph[sub] = np.concatenate(phs)
                    sub_rr[sub] = np.concatenate(rrs)

            sin_coefs = []
            for sub in sub_amp:
                b_sin, _ = per_subject_sin(sub_amp[sub], sub_ph[sub], rr=sub_rr[sub])
                if b_sin is not None:
                    sin_coefs.append(b_sin)
            sin_coefs = np.array(sin_coefs)
            if len(sin_coefs) < 3:
                continue
            t_stat, p = stats.ttest_1samp(sin_coefs, 0)
            frac_neg = (sin_coefs < 0).mean()
            wb_rows.append({
                "experiment": exp_name, "window": f"{win[0]:.2f}-{win[1]:.2f}",
                "baseline": "none" if baseline is None else f"{baseline[0]:.3f}-{baseline[1]:.3f}",
                "n_sub": len(sin_coefs), "sin_mean": sin_coefs.mean(),
                "sin_median": np.median(sin_coefs), "t": t_stat, "p": p,
                "frac_negative": frac_neg,
            })
    wb_df = pd.DataFrame(wb_rows) if wb_rows else pd.DataFrame()
    wb_df.to_csv(ROBUST_OUT / "window_baseline_sensitivity.csv", index=False)
    print("\n=== 第 1 项: 窗口/基线敏感性 ===")
    if len(wb_df):
        print(wb_df.round(4).to_string(index=False))
    else:
        print("无结果 (检查预处理 fif 是否齐全)")

    # ---- 第 2 项: 运动代理伪迹控制 (仅主窗口, 无基线) ----
    mc_rows = []
    for sub in subs:
        sub_amp, sub_ph, sub_env, sub_deriv, sub_rr = [], [], [], [], []
        for ses in SESSIONS:
            for stim in get_session_stims(ses):
                r = extract_hepatic_epochs(sub, ses, stim, (0.25, 0.4), None)
                if r is None:
                    continue
                (epochs, ch_names, t, rpeak_idx,
                 ph_at_r, env_at_r, deriv_at_r, rr_prev) = r
                if ph_at_r is None or env_at_r is None or deriv_at_r is None:
                    continue
                mean_amp, _ = hep_from_epochs(epochs, ch_names, t, (0.25, 0.4))
                if mean_amp is None:
                    continue
                valid = ~np.isnan(ph_at_r) & ~np.isnan(mean_amp)
                sub_amp.append(mean_amp[valid])
                sub_ph.append(ph_at_r[valid])
                sub_env.append(env_at_r[valid])
                sub_deriv.append(deriv_at_r[valid])
                sub_rr.append(rr_prev[valid])
        if not sub_amp:
            continue
        amp = np.concatenate(sub_amp)
        ph = np.concatenate(sub_ph)
        env = np.concatenate(sub_env)
        deriv = np.concatenate(sub_deriv)
        rr = np.concatenate(sub_rr)
        b0, _ = per_subject_sin(amp, ph, rr=rr)
        b_env, _ = per_subject_sin(amp, ph, rr=rr, extra=[env])
        b_deriv, _ = per_subject_sin(amp, ph, rr=rr, extra=[deriv])
        b_both, _ = per_subject_sin(amp, ph, rr=rr, extra=[env, deriv])
        for label, v in [("base", b0), ("env", b_env),
                         ("deriv", b_deriv), ("both", b_both)]:
            if v is not None:
                mc_rows.append({"experiment": exp_name, "sub": sub, "model": label,
                                "sin_coef": v})
    mc_df = pd.DataFrame(mc_rows) if mc_rows else pd.DataFrame()
    mc_df.to_csv(ROBUST_OUT / "movement_artifact_control.csv", index=False)
    print("\n=== 第 2 项: 运动代理伪迹控制 (各模型 sin 系数跨被试) ===")
    if len(mc_df):
        for model, g in mc_df.groupby("model"):
            t, p = stats.ttest_1samp(g["sin_coef"].values, 0)
            print(f"  {model:6s}: sin 均值={g['sin_coef'].mean():+.4f} "
                  f"n={len(g)} t={t:.2f} p={p:.4g}")

    print("\n产出目录:", ROBUST_OUT)


if __name__ == "__main__":
    main()
