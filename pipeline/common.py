#!/usr/bin/env python3
"""共享常量与工具函数 (BBBD experiment2 分析管线)。

统一各脚本的:
  - 路径 (ROOT / EXP2 / PROC / OUT)
  - 会话/刺激/排除清单
  - 采样率 (从 BDF/JSON 读取, 不硬编码)
  - read_rpeak / filter_rr / get_end_time / roi_idx 等重复函数
"""

import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

# 强制 joblib 使用线程后端 (避免临时 mmap 文件):
#   MNE 并行计算 (ICA 拟合/滤波等) 默认经 joblib 多进程后端在
#   mne/__pycache__/ 下创建 tmpXXXX 临时 mmap 文件并在结束后删除;
#   该批量删除会被外层安全护栏拦截, 导致全量重跑时整步被杀。
#   设 JOBLIB_MULTIPROCESSING=0 后 joblib 走 threading 后端, 完全不产生
#   这些临时文件。需在 common import 时 (任何 Parallel 调用前) 设置。
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
#   - RAW_ROOT: BBBD 原始数据目录 (experiment2/3/4/5 等, 原始 BDF/derivatives)。
#   - BASE:     pipeline/pipeline_data/ (所有新产出 PROC/OUT/FIG 写到这里),
#               便于后期分析统一调用。
_PIPELINE_DIR = Path(__file__).resolve().parent         # .../pipeline/
RAW_ROOT = _PIPELINE_DIR.parent                          # pipeline/ -> BBBD/ (原始数据根)
ROOT = RAW_ROOT                                          # 兼容旧变量名 (含义 = 原始数据根)

# 数据输出根: pipeline/pipeline_data/ (所有新产出写到这里)。
BASE = _PIPELINE_DIR / "pipeline_data"

# 实验切换: 优先级 = 命令行 --experiment > 环境变量 BBBD_EXPERIMENT > 默认 "2"。
# 用法 (手动点运行, 无需 export):
#   python3 02_preprocess.py --all --experiment 3        # 只跑 exp3
#   python3 02_preprocess.py --all --experiment=3        # 同上
#   python3 02_preprocess.py --all --experiment all      # 依次跑 2/3/4/5
# 命令行参数在 import common 时即被读取 (见 _exp_from_argv), 决定本次运行的实验。
def _exp_from_argv():
    """从 sys.argv 解析 --experiment / --experiment=, 返回实验 ID 或 None。"""
    for i, a in enumerate(sys.argv):
        if a == "--experiment" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--experiment="):
            return a.split("=", 1)[1]
    return None

_REQUESTED = (_exp_from_argv() or os.environ.get("BBBD_EXPERIMENT", "2")).strip()
# "all" 表示依次运行所有实验 (由 run_all_or_once 用 subprocess 分派)。
# 模块级常量需要一个具体实验才能计算, 故 all 模式下回退到默认 "2"。
REQUESTED_ALL = (_REQUESTED == "all")
_CURRENT_EXP = "2" if REQUESTED_ALL else _REQUESTED

# 每个实验的真实设计 (刺激数 / 排除被试 / 会话语义)。
# 会话语义按各实验真实设计标注, 而非统一套用 attentive/distracted:
#   - exp2: incidental learning,  5 刺激, ses-01 attentive, ses-02 distracted
#   - exp3: intentional learning, 6 刺激, ses-01 attentive, ses-02 distracted
#   - exp4: 5视频双注意条件,      6 刺激, ses-01 attentive1, ses-02 attentive2 (均注意)
#   - exp5: 金钱激励干预,         3 刺激, ses-01 attentive,  ses-02 incentive (金钱激励)
_EXPERIMENTS = {
    "2": {
        "name": "experiment2",
        "n_stim": 5,
        # sub-08 缺 R 峰 derivatives, sub-28 缺 EEG (.bdf), 均无法进入主分析。
        "exclude": {"sub-08", "sub-28"},
        "sessions": {"ses-01": "attentive", "ses-02": "distracted"},
    },
    "3": {
        "name": "experiment3",
        "n_stim": 6,
        # sub-15 几乎无 R 峰 derivative (仅 1 个), 无法做心搏锁定分析, 排除。
        "exclude": {"sub-15"},
        "sessions": {"ses-01": "attentive", "ses-02": "distracted"},
    },
    "4": {
        "name": "experiment4",
        "n_stim": 6,
        "exclude": set(),
        "sessions": {"ses-01": "attentive1", "ses-02": "attentive2"},
        # 会话-刺激映射: 每个会话只录部分刺激 (README: ses-01 看 Stim 4/5/6,
        # ses-02 看 Stim 1/2/3)。缺省时每会话 = 全局 STIMS。
        "session_stims": {
            "ses-01": ["stim04", "stim05", "stim06"],
            "ses-02": ["stim01", "stim02", "stim03"],
        },
    },
    "5": {
        "name": "experiment5",
        "n_stim": 3,
        "exclude": set(),
        "sessions": {"ses-01": "attentive", "ses-02": "incentive"},
    },
}

if _CURRENT_EXP not in _EXPERIMENTS:
    raise SystemExit(
        f"未知实验标识 BBBD_EXPERIMENT={_CURRENT_EXP!r}, 可用: {sorted(_EXPERIMENTS)}")

_CFG = _EXPERIMENTS[_CURRENT_EXP]

EXP_NAME = _CFG["name"]
STIMS = [f"stim{i:02d}" for i in range(1, _CFG["n_stim"] + 1)]
EXCLUDE = _CFG["exclude"]

# 会话语义: session 目录 -> 语义标签 (供 condition 列与配对/对比使用)。
SESSIONS = _CFG["sessions"]
SESSIONS_LIST = list(SESSIONS.keys())

# 会话顺序 (配对/对比时确定"基准"与"对照"会话): 显式给出, 避免 dict 无序隐患。
SESSION_ORDER = sorted(SESSIONS.keys())
BASE_SESSION = SESSION_ORDER[0]      # 基准会话 (attentive / attentive1)
CONTRAST_SESSION = SESSION_ORDER[1]  # 对照会话 (distracted / attentive2 / incentive)

# 会话语义标签 (配对/对比时使用的 condition 字符串), 供下游脚本动态读取,
# 避免在 result 脚本中 hardcode "attentive"/"distracted" (exp4/exp5 语义不同)。
BASE_LABEL = SESSIONS[BASE_SESSION]          # "attentive" (exp2/3/5) 或 "attentive1" (exp4)
CONTRAST_LABEL = SESSIONS[CONTRAST_SESSION]  # "distracted" / "attentive2" / "incentive"

# 数据目录: 指向当前实验 (保留 EXP2 变量名以兼容所有脚本, 含义为"原始数据目录")
EXP2 = ROOT / EXP_NAME

# 预处理产物目录: 写回 pipeline/pipeline_data/, 按实验隔离。
PROC = BASE / "derivatives_bhci" / f"processed_{EXP_NAME}"

# 分析输出目录: 写回 pipeline/pipeline_data/, 按实验隔离。
OUT = BASE / "data" / EXP_NAME                          # pipeline_data/data/experimentX/
FIG = BASE / "figures" / EXP_NAME                       # pipeline_data/figures/experimentX/

# 当前实验 ID (供脚本写入输出列 / 元数据)。
EXPERIMENT_ID = _CURRENT_EXP

# 可用实验 ID (供各脚本 argparse --experiment choices)。
EXPERIMENT_CHOICES = sorted(_EXPERIMENTS.keys()) + ["all"]

# 每会话刺激映射 (缺省每会话 = 全局 STIMS)。
# 部分实验 (如 exp4) 每个会话只录部分刺激, 用此字典覆盖。
SESSION_STIMS = _CFG.get("session_stims")


def get_session_stims(ses):
    """返回某会话的刺激列表 (会话特有或全局)。"""
    if SESSION_STIMS and ses in SESSION_STIMS:
        return SESSION_STIMS[ses]
    return STIMS


def run_all_or_once(main_func):
    """脚本入口包装器: 若 --experiment all, 依次以 2/3/4/5 重跑自身, 否则直接跑 main_func。

    用法: 脚本入口改为
        if __name__ == "__main__":
            run_all_or_once(main)
    这样 `python xxx.py --all --experiment all` 会依次产出 4 个实验的结果,
    每个实验在独立子进程中运行 (互不共享内存, 各自写回 experiment{2,3,4,5} 目录)。
    """
    if not REQUESTED_ALL:
        main_func()
        return

    import subprocess
    script = sys.argv[0]
    # 去掉 argv 里的 --experiment all / --experiment=all, 换成逐实验的 --experiment X。
    # 从 argv[1] 开始 (argv[0] 是脚本名, 由 subprocess 的 script 参数单独传入)。
    base_argv = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--experiment":
            i += 2  # 跳过 --experiment 与其值
            continue
        if a.startswith("--experiment="):
            i += 1
            continue
        base_argv.append(a)
        i += 1

    failed = []
    for exp in sorted(_EXPERIMENTS.keys()):
        print(f"\n{'=' * 60}\n>>> 运行实验 {exp} ({_EXPERIMENTS[exp]['name']})\n{'=' * 60}",
              flush=True)
        r = subprocess.run([sys.executable, script, *base_argv, "--experiment", exp])
        if r.returncode != 0:
            failed.append(exp)
            print(f"[警告] 实验 {exp} 退出码 {r.returncode}", flush=True)

    if failed:
        raise SystemExit(f"以下实验运行失败: {failed}")
    print("\n全部实验运行完成。")

TMIN, TMAX = -0.3, 0.6          # HEP 时间窗
MIN_RR = 0.5                    # 最小 RR 间期 (主分析冻结值; 06_multiverse 扫描其他值作敏感性)
# 基线校正: 主分析不做基线校正 (APPLY_BASELINE=False)。
# 原因 (Steinfath 2025; Petzschner 2019; Azzalini 2019): 心跳是周期性的, RR 间期
# 0.6-1.2s 波动, 任何基线窗 (靠后撞当前跳 P 波 CFA, 靠前撞前一跳 HEP 晚期) 都会
# 把污染注入主窗口。主分析仅依赖 0.1 Hz 高通去慢漂移。
# BASELINE_WIN 保留供"corrected"敏感性对照 (审稿人对照), 不进入主结论。
APPLY_BASELINE = False
BASELINE_WIN = (-0.1, -0.025)   # 仅作敏感性对照 (原始 Coll 2021 窗)
MAIN_WIN = (0.2, 0.6)           # 主效应窗口
EARLY_WIN = (0.0, 0.2)          # 早期 CFA 对照窗口
# epoch 振幅拒绝阈值 (μV): 剔除高振幅伪迹 epoch (Virjee 综述第二大影响参数)。
# 实测本数据集 clean/eyeonly 的 epoch 最大峰峰值中位 ~28μV、最大 ~51μV,
# 100μV 完全不拒绝任何 epoch (0/156), 阈值失去作用; 降到 50μV 使拒绝有实际区分度,
# 同时不误杀 (仅剔除峰峰值>50μV 的极少数大伪迹 epoch)。
EPOCH_REJECT_UV = 50.0

ROI = {
    "frontocentral": ['Fp1', 'Fp2', 'AF3', 'AF4', 'F1', 'F2', 'F3', 'F4', 'Fz',
                      'FC1', 'FC2', 'FC3', 'FC4', 'FCz', 'C1', 'C2', 'C3', 'C4', 'Cz',
                      'CP1', 'CP2', 'CP3', 'CP4', 'CPz'],
    "central": ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz', 'FC1', 'FC2', 'FCz',
                'CP1', 'CP2', 'CPz'],
    "parietal": ['P1', 'P2', 'P3', 'P4', 'Pz', 'PO3', 'PO4', 'POz', 'P5', 'P6', 'P7', 'P8'],
}


def list_subjects():
    """列出全部被试目录 (排除 EXCLUDE)。"""
    return sorted(
        p.name for p in EXP2.glob("sub-*")
        if p.is_dir() and p.name not in (".DS_Store",) and p.name not in EXCLUDE
    )


def read_rpeak(sub, ses, stim):
    """读官方 R 峰时间戳 (秒)。"""
    p = EXP2 / "derivatives" / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_desc-rpeak_timestamps.tsv.gz"
    if not p.is_file():
        return None
    with gzip.open(p, "rt") as f:
        lines = [l.strip() for l in f if l.strip()]
    if lines and lines[0].lower().startswith("timestamp"):
        lines = lines[1:]
    return np.array([float(x) for x in lines])


def read_eog(sub, ses, stim):
    """读原始 EOG 信号 (6 通道, μV)。返回 (n_times, 6) 数组, 或 None。

    EOG 独立存储于 beh/*_recording-eog_physio.tsv.gz (BioSemi 6 眼周电极,
    通道名仅 ch1-ch6, 无 VEOG/HEOG 语义), 采样率与 EEG 一致。
    """
    p = EXP2 / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_recording-eog_physio.tsv.gz"
    if not p.is_file():
        return None
    cols = []
    with gzip.open(p, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            try:
                cols.append([float(x) for x in parts])
            except ValueError:
                continue
    if not cols:
        return None
    # 列数一致性: 丢弃与首行列数不一致的行 (文件末尾空行/注释/异常行)
    n_col = len(cols[0])
    cols = [c for c in cols if len(c) == n_col]
    if not cols:
        return None
    return np.array(cols)   # (n_times, n_chan)


def read_ecg(sub, ses, stim):
    """读原始 ECG 信号 (单通道, mV)。返回 (n_times,) 数组, 或 None。"""
    p = EXP2 / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_recording-ecg_physio.tsv.gz"
    if not p.is_file():
        return None
    vals = []
    with gzip.open(p, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line.split("\t")[0]))
            except ValueError:
                continue
    if not vals:
        return None
    return np.array(vals)


def read_respiration(sub, ses, stim):
    """读原始呼吸信号 (胸带张力, 单通道, mV)。返回 (n_times,) 数组, 或 None。

    仅部分实验 (exp4/exp5) 有呼吸信号; exp2/exp3 无此文件时返回 None。
    """
    p = EXP2 / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_recording-respiration_physio.tsv.gz"
    if not p.is_file():
        return None
    vals = []
    with gzip.open(p, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line.split("\t")[0]))
            except ValueError:
                continue
    if not vals:
        return None
    return np.array(vals)


def filter_rr(rpeaks, min_rr=MIN_RR):
    """按最小 RR 间期过滤 R 峰, 返回布尔 mask。

    维护"上一个保留的峰"作为参考, 保证连续剔除时仍能正确保留
    间隔 >= min_rr 的子序列 (而非与紧邻的原始峰比较)。
    """
    keep = np.ones(len(rpeaks), dtype=bool)
    last_kept = rpeaks[0] if len(rpeaks) > 0 else None
    for i in range(1, len(rpeaks)):
        if rpeaks[i] - last_kept < min_rr:
            keep[i] = False
        else:
            last_kept = rpeaks[i]
    return keep


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR 校正。输入 p 值数组, 返回校正后 p 值数组 (NaN 保持 NaN)。"""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid = ~np.isnan(pvals)
    if not valid.any():
        return out
    p = pvals[valid]
    n = len(p)
    order = np.argsort(p)
    rank = np.argsort(order)          # 每个原始位置的排名 (0-based)
    sorted_p = p[order]
    bh = sorted_p * n / (np.arange(n) + 1.0)
    # 累积最小值 (从大到小)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    out[valid] = bh[rank]
    return out


def reject_epochs_by_amplitude(epochs, thr_uv):
    """剔除峰峰值超过 thr_uv (μV) 的 epoch, 返回保留的 epoch 索引数组。

    epochs: (n_ep, n_ch, n_t) 三维数组, μV 单位。
    逐 epoch 取所有通道的最大峰峰值, 超过阈值的剔除 (Virjee 综述:
    振幅拒绝阈值是 HEP 第二大影响参数)。
    """
    pp = np.ptp(epochs, axis=-1)      # (n_ep, n_ch)
    max_pp = pp.max(axis=1)           # (n_ep,)
    return np.where(max_pp <= thr_uv)[0]


def get_end_time(sub, ses, stim):
    """从 events.tsv 读 end 事件时间, 兼容 trial_type/value/event 列。"""
    p = EXP2 / sub / ses / "eeg" / f"{sub}_{ses}_task-{stim}_events.tsv"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            val = row.get("trial_type") or row.get("value") or row.get("event")
            if val == "end":
                return float(row["onset"])
    return None


def roi_idx(ch_names, roi):
    """ROI 通道名 -> 索引数组。"""
    m = {c: i for i, c in enumerate(ch_names)}
    return np.array([m[c] for c in roi if c in m])


def read_sampling_rate_from_json(sub, ses, stim, kind="eeg"):
    """从 *_eeg.json / *_recording-{ecg,eog,respiration}_physio.json 读 SamplingFrequency。"""
    if kind in ("ecg", "eog", "respiration"):
        p = EXP2 / sub / ses / "beh" / f"{sub}_{ses}_task-{stim}_recording-{kind}_physio.json"
    else:
        p = EXP2 / sub / ses / "eeg" / f"{sub}_{ses}_task-{stim}_eeg.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("SamplingFrequency")


# ---- 供 paper_results 脚本复用的 CFA/HEP 公共辅助函数 ----

FP_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF8', 'F7', 'F8']
EYE_TOPO_THRESH = 0.3


def detect_eye_idx(ica, raw_eeg):
    """地形图判据识别眼动成分。raw_eeg 仅用其 ch_names (频带无关)。

    前额极通道 (Fp1/Fp2/AF7/AF8/F7/F8) 权重均值 > 0.3 判定为眼动成分。
    若所有前额极通道均缺失 (fp_idx 为空), 直接返回空集 (无眼动成分可判)。
    """
    fp_idx = [raw_eeg.ch_names.index(c) for c in FP_CHANNELS if c in raw_eeg.ch_names]
    if not fp_idx:
        return set()
    comp_weights = ica.get_components()
    if comp_weights.shape[0] != ica.n_components_:
        comp_weights = comp_weights.T
    eye_idx = set()
    for i in range(ica.n_components_):
        fp_weight = np.abs(comp_weights[i, fp_idx]).mean()
        if fp_weight > EYE_TOPO_THRESH:
            eye_idx.add(i)
    return eye_idx


def cfa_spatial_weight(ch_names, montage):
    """真实 CFA 空间梯度: 前中央(Y大)强, 顶/枕区(Y小)弱。"""
    pos = montage.get_positions()['ch_pos']
    w = np.zeros(len(ch_names))
    for i, ch in enumerate(ch_names):
        if ch in pos:
            y = pos[ch][1]
            w[i] = float(np.clip((y + 0.08) / (0.05 + 0.08), 0.2, 1.0))
        else:
            w[i] = 0.5
    w = w / np.linalg.norm(w)
    return w


def hep_spatial_weight(ch_names, montage):
    """HEP 空间分布: 前中央(Fz 最强)主导, 顶区弱。

    与 CFA 空间分布不同: HEP 是额中央(Fz/Cz)主导, CFA 是更弥散的前中央+颞区。
    """
    pos = montage.get_positions()['ch_pos']
    w = np.zeros(len(ch_names))
    for i, ch in enumerate(ch_names):
        if ch in pos:
            y = pos[ch][1]
            z = pos[ch][2]
            w[i] = float(np.clip(z / 0.1, 0.2, 1.0) * np.clip((y + 0.02) / (0.06 + 0.02), 0.2, 1.0))
        else:
            w[i] = 0.5
    w = w / np.linalg.norm(w)
    return w
