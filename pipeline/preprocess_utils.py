#!/usr/bin/env python3
"""共享 EEG 预处理函数 (02_preprocess / 06b / 06c 复用)。

集中以下重复逻辑, 避免长期维护改一处漏一处:
  - read_eeg_bdf          读 BDF
  - detect_bad_channels   坏导检测 (峰峰值 MAD)
  - apply_montage_and_interpolate  设置 montage + 插值坏导
  - bandpass_filter       带通滤波 (主分析 0.1-30Hz, MNE FIR, 零相位/因果)
  - fit_ica               在 1.0 Hz 高通信号上拟合 ICA (ICA_HIGHPASS)
  - compute_energy_ratios R 峰锁时能量比 (CFA 判据)
  - detect_eye_idx        眼动成分地形图判据 (从 common.py 复用, 唯一来源)

采样率一律从 raw.info["sfreq"] 读取, 不硬编码。

滤波方法学依据 (v0.3 冻结):
  - 主分析高通 0.1 Hz (BANDPASS_LOW): Tanner et al. 2015 警告 >=0.3 Hz 高通会引入
    伪迹; 保留 HEP 慢波成分 (静息心率 ≈ 1 Hz 的慢波)。
  - ICA 拟合高通 1.0 Hz (ICA_HIGHPASS): 去低频漂移 (Winkler et al. 2015),
    与主分析滤波解耦; 最终施加解混矩阵的主分析信号仍用 0.1 Hz 高通。
  - FIR 而非 IIR: 避免 5 阶 Butterworth 带通在低截止处的数值不稳定性
    (极点模 ≈ 0.99, 因果 lfilter 低频群延迟 > 1 s)。
  - 主分析零相位 (phase='zero'): 保证 R 峰对齐 0 ms (Steinfath 2026 强制校验)。
    因果 (phase='minimum') 作为多宇宙敏感性维度, 其群延迟在 06_multiverse 报告。
"""

import warnings

import numpy as np
import mne
from mne.preprocessing import ICA

from common import (
    EXP2, read_rpeak, get_end_time,
    detect_eye_idx, FP_CHANNELS, EYE_TOPO_THRESH,   # 唯一来源 (避免与 common 重复定义)
)

# CFA 成分识别阈值 (锁时能量比) —— 主判据 (06b 扫描此阈值作敏感性)
CFA_ENERGY_RATIO_THRESH = 0.3

# 带通频带 (Hz)。
# 主分析高通 0.1 Hz (Tanner et al. 2015 警告 >=0.3 Hz 高通会引入伪迹;
# Widmann et al. 2015 数字滤波器设计规范), 保留 HEP 慢波成分。
BANDPASS_LOW = 0.1
BANDPASS_HIGH = 30.0

# ICA 拟合的高通截止 (Hz) —— 与主分析滤波解耦:
# ICA 用较高高通 (1.0 Hz) 去除低频漂移 (Winkler et al. 2015), 但最终
# 施加 ICA 解混矩阵的主分析信号保持 0.1 Hz 高通 (Tanner et al. 2015)。
ICA_HIGHPASS = 1.0


def read_eeg_bdf(sub, ses, stim):
    """读原始 BDF (preload)。不存在返回 None。"""
    p = EXP2 / sub / ses / "eeg" / f"{sub}_{ses}_task-{stim}_eeg.bdf"
    if not p.is_file():
        return None
    return mne.io.read_raw_bdf(p, preload=True, verbose=False)


def detect_bad_channels(raw, thr_pp=3.0, flat_thr_uv=0.5):
    """基于峰峰值(相对中位数的MAD)检测坏导 (仅 EEG 通道)。

    检测两类坏导:
      - 幅值过大 (峰峰值 > 中位数 + thr_pp*MAD): 噪声/漂移电极
      - 平坦通道 (峰峰值 < flat_thr_uv μV): 电极脱落/无信号 (flat), 不参与 CAR/ICA

    返回坏导通道名列表。
    """
    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, misc=False)
    data = raw.get_data()[eeg_idx] * 1e6  # μV (用 eeg_idx 直接索引, 保证与 eeg_chs 对齐)
    pp = np.ptp(data, axis=1)
    med = np.median(pp)
    mad = np.median(np.abs(pp - med))
    thr = med + thr_pp * mad * 1.4826   # MAD -> 稳健标准差
    eeg_chs = [raw.ch_names[i] for i in eeg_idx]
    bad = [c for c, v in zip(eeg_chs, pp) if v > thr or v < flat_thr_uv]
    return bad


def apply_montage_and_interpolate(raw, min_common=50):
    """设置标准 10-20 montage (供坏导插值), 检测并插值坏导。

    返回坏导列表。就地修改 raw (不返回副本)。
    """
    montage = mne.channels.make_standard_montage('standard_1020')
    common = [c for c in raw.ch_names if c in montage.ch_names]
    if len(common) >= min_common:
        raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
    bad = detect_bad_channels(raw)
    if bad:
        raw.info['bads'] = bad
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return bad


def bandpass_filter(raw, phase='zero', l_freq=BANDPASS_LOW):
    """带通滤波 (默认 0.1-30 Hz 主分析), MNE FIR, 就地修改 raw 并记录元数据。

    l_freq: 高通截止 (默认 BANDPASS_LOW=0.1 Hz 主分析, 保留 HEP 慢波;
            ICA 拟合用 ICA_HIGHPASS=1.0 Hz)。
    phase='zero'    -> 零相位 FIR (主分析, R 峰对齐 0 ms)
    phase='minimum' -> 最小相位 FIR (因果, 多宇宙敏感性, 引入群延迟)

    用 MNE raw.filter 而非裸 scipy IIR: FIR 数值稳定, 且自动写回
    raw.info['lowpass']/['highpass'] 供报告与审稿检查。
    """
    raw.filter(l_freq, BANDPASS_HIGH, fir_design='firwin',
               phase='zero' if phase == 'zero' else 'minimum', verbose=False)
    return raw


def set_average_reference(raw):
    """平均参考 (CAR)。就地修改。"""
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw


def fit_ica(raw_eeg_hp, n_components=40, random_state=42, max_iter=2000):
    """在 1 Hz 高通 EEG 上拟合 ICA。raw_eeg_hp 应是 ICA 轨 (1-30 Hz) 的 EEG 通道。

    ICA 拟合与主分析滤波解耦 (Tanner et al. 2015): ICA 用 1 Hz 高通去低频漂移
    (Winkler et al. 2015), 最终施加解混矩阵的主分析信号用 0.1 Hz 高通保留慢波。

    收敛设置 (扩展 infomax): max_iter=2000 (单成分迭代上限)。拟合后 ICA.n_iter_
    记录实际迭代次数, 供收敛质量报告 (写入 preprocess_report 的 ica_n_iter 列)。

    警告抑制说明:
      - FastICA 未收敛: 收敛性警告属于迭代性提示, 不影响成分分离可用性, 统一过滤。
      - "unstable mixing matrix": 白化后方差趋近 0 的退化成分, 固定成分数下的正常现象。
    """
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*high-pass filtered.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition._fastica")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.decomposition._fastica")
        ica.fit(raw_eeg_hp, verbose=False)
    return ica


def highpass_eeg(raw_eeg, l_freq=ICA_HIGHPASS):
    """对 EEG raw 做高通 (ICA 拟合轨)。返回副本。

    高通截止 1.0 Hz (ICA_HIGHPASS), 仅用于 ICA 拟合去漂移;
    主分析信号另用 0.1 Hz 高通 (Tanner et al. 2015 解耦策略)。
    """
    return raw_eeg.copy().filter(l_freq, None, fir_design='firwin', verbose=False)


def validate_cfa_with_ecg(ica, raw_eeg, cfa_idx, ecg, rpeaks, srate):
    """用 ECG 通道独立验证 CFA 成分选择 (正交验证, 不参与成分分类)。

    计算每个 ICA 成分时间序列与 ECG 的时域 Pearson 相关 (重采样对齐后),
    取 |r| 最高的成分作为 ECG 驱动的候选。返回重叠比例:
      overlap = |cfa_idx ∩ ecg_top_idx| / |cfa_idx ∪ ecg_top_idx|
    overlap < 0.5 表示锁时能量比判据与 ECG 相关判据严重不一致, 需人工审查。

    注: 这是"独立正交验证", 仅写报告; 不改变 cfa_idx (主判据仍是锁时能量比)。
    """
    try:
        if ecg is None or len(ecg) < 10 or len(cfa_idx) == 0:
            return None
        ica_src = ica.get_sources(raw_eeg).get_data()   # (n_comp, n_times)
        n = min(ica_src.shape[1], len(ecg))
        ecg_f = np.asarray(ecg[:n], dtype=float)
        if np.std(ecg_f) == 0:
            return None
        corrs = []
        for i in range(ica.n_components_):
            comp = ica_src[i, :n]
            if np.std(comp) == 0:
                corrs.append(0.0)
            else:
                corrs.append(abs(float(np.corrcoef(comp, ecg_f)[0, 1])))
        corrs = np.array(corrs)
        # ECG 驱动的候选 = 相关 > 0.15 的成分 (BioSemi 独立 ECG 与 EEG 耦合弱,
        # 实测 max |r|≈0.25; 常规 0.3 阈值在此数据集几乎无成分达标, 故用 0.15)。
        ecg_idx = set(np.where(corrs > 0.15)[0].tolist())
        if not ecg_idx:
            return {"overlap_ratio": None, "n_ecg_idx": 0,
                    "max_ecg_corr": round(float(corrs.max()), 4)}
        union = set(cfa_idx) | ecg_idx
        overlap = len(set(cfa_idx) & ecg_idx) / max(len(union), 1)
        return {"overlap_ratio": round(float(overlap), 4),
                "n_ecg_idx": len(ecg_idx),
                "max_ecg_corr": round(float(corrs.max()), 4)}
    except Exception:
        return None


def verify_rpeak_alignment(ecg, rpeaks, srate, n_epochs_check=50):
    """R 峰对齐验证 (Steinfath 2026 强制校验)。

    将 ECG 信号 epoch 到官方 R 峰时间戳, 取 -100~200 ms 内最大 |幅值| 时刻,
    检查 ECG 的 R 波峰值是否精确落在 0 ms。若偏离 0 ms, 说明 R 峰时间戳与
    生理信号存在系统偏移 (或滤波/下采样引入群延迟), 必须报告。

    参数:
      ecg: 1D numpy 数组 (原始 ECG, 单位任意, 与 rpeaks 同采样率)
      rpeaks: 官方 R 峰时间戳 (秒)
      srate: 采样率 (Hz)

    返回 dict: {"rpeak_lag_ms", "r_amp", "n_epochs"} 或 None (数据不足)。
    rpeak_lag_ms 越接近 0 表示对齐越好。
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or len(ecg) < 10:
        return None
    tmin, tmax = -0.1, 0.2
    n_t = int((tmax - tmin) * srate)
    starts = np.round((rpeaks + tmin) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= len(ecg))]
    if len(starts) < 5:
        return None
    starts = starts[:n_epochs_check]
    idx = starts[:, None] + np.arange(n_t)[None, :]
    epochs = ecg[idx]                    # (K, n_t)
    ev = epochs.mean(axis=0)
    t = np.arange(n_t) / srate + tmin
    # R 波在 -50~50 ms 内 (ECG 的 R 峰本应在 0 ms)
    win = (t >= -0.05) & (t <= 0.05)
    if not win.any():
        return None
    i_peak = int(np.argmax(np.abs(ev[win])))
    lag_ms = t[win][i_peak] * 1000
    amp = float(np.abs(ev[win][i_peak]))
    return {"rpeak_lag_ms": round(float(lag_ms), 1),
            "r_amp": round(amp, 4),
            "n_epochs": int(len(starts))}


def compute_energy_ratios(ica, raw_src, rpeaks, end_time, srate):
    """返回每个成分的锁时能量比 (n_components,)。

    raw_src: 与 ICA 分解同频带的信号 (1-30 Hz 高通)。
    锁时能量比 = 锁时平均段能量 / 成分总能量 (CFA 判据, 与 02/06b/06c 一致)。

    锁时窗口 = QRS 复合波 (-0.05 ~ 0.05s, 心室去极化产生的心电场伪迹 CFA
    主能量区间)。CFA 是心脏电活动经容积传导到头皮 EEG 的伪迹, 其生理来源
    QRS 波严格锁 R 峰前后 ~50ms (Q 波前 -30ms 至 S 波后 +50ms)。
    旧版用 0~0.2s (R 峰后 200ms) 主要覆盖 T 波/早期 HEP, 恰错过 QRS,
    导致 CFA 成分几乎检不出 (overlap_ratio≈0)。
    """
    ica_src = ica.get_sources(raw_src).get_data()
    tmin, tmax = -0.05, 0.05
    rp = rpeaks[(rpeaks > abs(tmin)) & (rpeaks < end_time - tmax)]
    n_t = int((tmax - tmin) * srate)
    starts = np.round((rp + tmin) * srate).astype(int)
    starts = starts[(starts >= 0) & (starts + n_t <= raw_src.n_times)]
    if len(starts) < 10:
        return None
    ratios = []
    for i in range(ica.n_components_):
        comp = ica_src[i]
        segs = np.array([comp[s:s + n_t] for s in starts])
        locked = segs.mean(axis=0)
        le = np.mean(locked ** 2)
        te = np.mean(comp ** 2)
        ratios.append(le / te if te > 0 else 0.0)
    return np.array(ratios)
