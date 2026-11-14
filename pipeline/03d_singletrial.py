#!/usr/bin/env python3
"""
Step 3d: 单试次 HEP 长格式导出 (供 LMM 建模)

目的: 导出单试次 (single-trial) 水平的 HEP 振幅 + 协变量, 供 07_lmm.R 做
      线性混合效应模型 (hep_amp ~ condition + rr_interval + resp_phase + (1|sub))。
      相比被试水平平均, 单试次 LMM 能纳入 RR 间期 (心跳重叠) 与呼吸相位
      (呼吸窦性心律不齐 RSA) 作为协变量, 控制试次间不平衡。

产出:
  - data/hep_singletrial.csv  每试次一行: sub/ses/stim/condition/rpeak_idx/
                               hep_amp_uv / rr_prev_s / resp_phase

用法: python 03d_singletrial.py [--sub sub-01] [--all]
"""

import argparse
import csv

import numpy as np
import mne
from scipy.signal import butter, filtfilt, find_peaks

from common import (
    ROOT, EXP2, PROC, OUT, SESSIONS, STIMS, EXCLUDE, EXPERIMENT_CHOICES, TMIN, TMAX,
    MAIN_WIN, BASELINE_WIN, EPOCH_REJECT_UV, ROI, read_rpeak, filter_rr, get_end_time,
    get_session_stims, read_respiration, read_sampling_rate_from_json,
    reject_epochs_by_amplitude, run_all_or_once,
)

OUT.mkdir(parents=True, exist_ok=True)

# 单试次 HEP 窗口 (晚期成分, Steinfath 2025: 250-400ms 为 T 波/晚期 HEP 核心)
ST_WIN = (0.25, 0.40)


def respiration_phase(sub, ses, stim, srate, n_times):
    """读呼吸信号, 用双阶段峰谷插值提取呼吸相位。返回 (n_times,) 相位数组 (0~2π), 或 None。

    方法: 检测呼气谷 (trough) 与吸气峰 (peak), 用双阶段分段线性插值:
      - 谷 → 峰 映射为 0 → π (吸气相, inspiration)
      - 峰 → 下一谷 映射为 π → 2π (呼气相, expiration)
    双阶段插值保留吸/呼时间的不对称性 (I:E 通常 1:1.5~1:2), 而非单阶段谷-谷
    插值强制吸气峰落在 π (Zaccaro et al. 2018; Kluger et al. 2021)。

    相位定义: 呼气谷=0, 吸气峰=π (峰的位置由实际检测决定, 而非假设在中点)。
    """
    resp = read_respiration(sub, ses, stim)
    if resp is None or len(resp) < 10:
        return None
    # 采样率对齐 (json 缺失时假设与 EEG 同采样率)
    srate_resp = read_sampling_rate_from_json(sub, ses, stim, kind="respiration")
    srate_ok = (srate_resp is None) or (abs(srate_resp - srate) < 1e-6)
    if not srate_ok:
        return None
    resp = np.asarray(resp, dtype=float)
    resp = resp - np.mean(resp)
    # 去趋势 + 0.05-0.5 Hz 带通 (呼吸频率), 仅用于峰谷定位 (不用于相位本身)
    b, a = butter(3, [0.05 / (srate / 2), 0.5 / (srate / 2)], btype='bandpass')
    resp_f = filtfilt(b, a, resp)
    # 检测呼气谷 (局部极小) 与吸气峰 (局部极大)。
    # distance 用呼吸周期下界 1.5s 防误检 (7.5-40 bpm 的合理呼吸频率范围)。
    min_dist = int(1.5 * srate)
    troughs, _ = find_peaks(-resp_f, distance=min_dist)
    peaks, _ = find_peaks(resp_f, distance=min_dist)
    if len(troughs) < 2 or len(peaks) < 1:
        return None
    # 峰谷间隔合理性检查: 剔除异常周期 (间隔 > 中位数 ± 3 MAD), 防止呼吸带脱落/
    # 叹息导致的伪峰污染双阶段插值。
    if len(troughs) > 3:
        trough_int = np.diff(troughs)
        med_int = np.median(trough_int)
        mad_int = np.median(np.abs(trough_int - med_int))
        if mad_int > 0:
            ok_edge = np.abs(trough_int - med_int) < 3 * mad_int * 1.4826
            # 保留"两端都正常"的谷 (除首末谷)
            ok_keep = np.ones(len(troughs), dtype=bool)
            for i in range(1, len(troughs) - 1):
                if not (ok_edge[i - 1] and ok_edge[i]):
                    ok_keep[i] = False
            troughs = troughs[ok_keep]
            if len(troughs) < 2:
                return None
    # 双阶段分段线性插值: 谷→峰 = 吸气 0→π, 峰→谷 = 呼气 π→2π
    phase = np.full(len(resp), np.nan)
    for i in range(len(troughs) - 1):
        t_trough0 = troughs[i]
        t_trough1 = troughs[i + 1]
        # 找当前谷-谷之间的吸气峰 (有且仅一个)
        p_mid = peaks[(peaks > t_trough0) & (peaks < t_trough1)]
        if len(p_mid) != 1:
            # 该周期缺峰, 退化为单阶段谷-谷插值
            seg = np.arange(t_trough0, t_trough1)
            phase[seg] = 2 * np.pi * (seg - t_trough0) / (t_trough1 - t_trough0)
            continue
        p = p_mid[0]
        # 阶段 1: 吸气 (谷→峰) -> [0, π)
        seg_in = np.arange(t_trough0, p)
        phase[seg_in] = np.pi * (seg_in - t_trough0) / (p - t_trough0)
        # 阶段 2: 呼气 (峰→下一谷) -> [π, 2π)
        seg_ex = np.arange(p, t_trough1)
        phase[seg_ex] = np.pi + np.pi * (seg_ex - p) / (t_trough1 - p)
    # 首谷前 / 末谷后: 用最近完整周期的斜率线性延续 (外推)
    first_t0, first_t1 = troughs[0], troughs[1]
    pre = np.arange(0, first_t0)
    if len(pre) > 0:
        phase[pre] = 2 * np.pi * (pre - first_t0) / (first_t1 - first_t0)
    last_t0, last_t1 = troughs[-2], troughs[-1]
    post = np.arange(last_t1, len(resp))
    if len(post) > 0:
        phase[post] = 2 * np.pi * (post - last_t0) / (last_t1 - last_t0)
    # 相位归一化到 [0, 2π) (外推段可能超出)
    phase = np.mod(phase, 2 * np.pi)
    # 对齐到 EEG 长度
    n = min(len(phase), n_times)
    return phase[:n]


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

                valid = (rpeaks > abs(TMIN)) & (rpeaks < end_time - TMAX)
                # 时间窗过滤后的完整序列 (未 filter_rr), 用于计算真实的生理前 RR。
                # filter_rr 会剔除 RR<MIN_RR 的峰, 若基于过滤后序列算 rr_prev 会
                # 跨过被剔除的峰、高估真实前 RR (心跳间隔是生理事实, 不因剔除而改变)。
                rp_all = rpeaks[valid]
                keep_mask = filter_rr(rp_all)
                rp = rp_all[keep_mask]
                if len(rp) < 6:
                    continue

                # 前一个 RR 间期: 基于未 filter_rr 的完整序列 (rp_all) 计算,
                # 映射到 filter_rr 保留的峰; 若"前一个原始峰被剔除", 置 NaN。
                rr_all = np.diff(rp_all, prepend=np.nan)
                kept_indices = np.where(keep_mask)[0]
                rr_prev_kept = rr_all[kept_indices]
                if len(kept_indices) > 1:
                    consecutive = np.diff(kept_indices) == 1
                    prev_valid = np.concatenate([[True], consecutive])
                    rr_prev_kept[~prev_valid] = np.nan

                # 呼吸相位 (若该实验有呼吸信号)
                resp_phase = respiration_phase(sub, ses, stim, srate, data.shape[1])

                # 提取单试次 epoch
                n_t = int((TMAX - TMIN) * srate)
                t = np.arange(n_t) / srate + TMIN
                starts = np.round((rp + TMIN) * srate).astype(int)
                # 越界过滤 (同步裁剪 rp / starts / rr_prev_kept, 保证三者长度一致)
                ok = (starts >= 0) & (starts + n_t <= data.shape[1])
                starts = starts[ok]
                rp = rp[ok]
                rr_prev_kept = rr_prev_kept[ok]
                if len(starts) < 5:
                    continue
                idx = starts[:, None] + np.arange(n_t)[None, :]
                epochs = data[:, idx].transpose(1, 0, 2)  # (n_ep, n_ch, n_t)

                # epoch 振幅拒绝
                keep = reject_epochs_by_amplitude(epochs, EPOCH_REJECT_UV)
                epochs = epochs[keep]
                starts = starts[keep]
                rp_kept = rp[keep]
                rr_prev = rr_prev_kept[keep]
                if len(epochs) < 5:
                    continue

                # frontocentral ROI 平均
                fc = ROI["frontocentral"]
                fc_idx = [ch_names.index(c) for c in fc if c in ch_names]
                if len(fc_idx) == 0:
                    continue

                # 基线校正 (每 epoch 减其 -300~-100ms 基线均值, 避开 P 波)
                base_mask = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
                epochs = epochs - epochs[:, :, base_mask].mean(axis=2, keepdims=True)

                # 单试次 HEP (250-400ms 窗口, frontocentral 平均)。
                # 双报告: 有符号均值电压 (mean, 主指标, 保留极性/无噪声正偏) 与 RMS (对照)。
                # 有符号均值避免"信噪比波动被误判为神经增强" (Luck 2014)。
                st_mask = (t >= ST_WIN[0]) & (t <= ST_WIN[1])
                st_vals = epochs[:, fc_idx, :][:, :, st_mask].mean(axis=1)  # (n_ep, n_t_win)
                hep_amp_mean = st_vals.mean(axis=1)                          # (n_ep,) 有符号均值
                hep_amp_rms = np.sqrt(np.mean(st_vals ** 2, axis=1))        # (n_ep,) RMS

                # R 峰时刻采样点 (用于取 R 峰处呼吸相位, 非 epoch 起始点)。
                # epoch 起始点在 R 峰前 300ms (TMIN=-0.3), 若用 starts[i] 会取错相位时刻。
                rpeak_idx = np.round(rp_kept * srate).astype(int)

                for i in range(len(epochs)):
                    row = {
                        "sub": sub, "ses": ses, "stim": stim,
                        "condition": SESSIONS[ses],
                        "hep_amp_uv": round(float(hep_amp_mean[i]), 5),
                        "hep_amp_rms_uv": round(float(hep_amp_rms[i]), 5),
                        "rr_prev_s": round(float(rr_prev[i]), 4) if not np.isnan(rr_prev[i]) else "",
                        "resp_phase": round(float(resp_phase[rpeak_idx[i]]), 4)
                                       if resp_phase is not None and rpeak_idx[i] < len(resp_phase) else "",
                    }
                    rows.append(row)

        print(f"完成 {sub}: 累计 {len(rows)} 试次")

    if rows:
        fn = list(rows[0].keys())
        with open(OUT / "hep_singletrial.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(rows)
    print(f"\n总计 {len(rows)} 试次 -> {OUT / 'hep_singletrial.csv'}")


if __name__ == "__main__":
    run_all_or_once(main)
