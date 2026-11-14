# Methods — Pipeline 统一管线总览（冻结版 v0.3）

> 本文档是 `pipeline/` 统一分析管线的**方法学总览**，逐步骤对应 `run_one_experiment.sh` 中
> 00–09 全部脚本，并给出**已冻结的参数取值**与**文献依据**。
> 两篇论文的方法学文档（`methods_paperA_CFA.md` / `methods_paperB_brainheart.md`）
> 分别从 CFA 连续谱与脑-心耦合的视角引用本管线，此处不重复展开。

---

## 0. 管线结构

管线是自包含的：`common.py` 定义路径/常量，`preprocess_utils.py` 集中 EEG 预处理函数，
00–09 脚本按顺序执行，通过环境变量 `BBBD_EXPERIMENT`（或 `--experiment` 参数）切换实验
{2,3,4,5}，产物按实验隔离写入 `pipeline/pipeline_data/`。

```
00_inventory              数据盘点与质量审计
01_rpeak_validation       R 峰复检（官方 vs 自检）
02_preprocess             EEG 预处理（双轨 clean/eyeonly，含因果敏感性轨）
03b_hep_dual              HEP 提取（双轨，ROI×窗口 双报告 mean/RMS）
03c_pseudotrial           Pseudotrial 漂移校正
03d_singletrial           单试次 HEP 长格式（含 RR 间期/呼吸相位协变量）
03e_topography            头皮地形数据导出
04_surrogate_l1           Level 1 surrogate 零分布（时间平移 + 相位随机化）
05_coupling_l2            Level 2 心搏锁定相位一致性（ITPC）
06_multiverse             多宇宙分析（1296 pipelines）
06b_cfa_threshold_scan    CFA 阈值敏感性扫描（6 成分数 × 7 阈值）
06c_energy_ratio_only     CFA 能量比分布（20 成分，逐记录写盘）
06d_cfa_distribution_test CFA 能量比分布检验（dip test）
07_lmm                    LMM 主模型 + 调节 + 等效性 + 单试次
08_hrv                    心率变异性（时域 + Lomb-Scargle 频域）
09_cross_dataset          跨实验效应量汇总
```

> 注：`run_one_experiment.sh` 中 06d 在 06b 之前运行（依赖 06c 的产物），
> 06c/06b 均从原始 BDF 重跑 ICA（不读 02 的 fif），以保证 CFA 分类/能量比
> 与 02 的 clean/eyeonly 双轨在**同一 ICA 分解定义**下一致。

---

## 1. 数据与实验

数据来自 BBBD（Madsen, Kuppa & Parra, 2026, *Scientific Data*）。
四个实验共享 64 通道 BioSemi EEG（128 Hz）+ 单导联 ECG（128 Hz），BIDS 结构一致，
仅实验操纵不同（见下表）。`common.py` 内置各实验真实设计（刺激数、排除被试、会话语义）。

| 实验 | 刺激数 | 会话语义（ses-01 / ses-02） | 排除被试 | 附加模态 |
|------|--------|------------------------------|----------|----------|
| 2 | 5 | attentive / distracted | sub-08, sub-28 | — |
| 3 | 6 | attentive / distracted | sub-15 | — |
| 4 | 6 | attentive1 / attentive2（分刺激）| — | 呼吸 |
| 5 | 3 | attentive / incentive | — | 呼吸 |

> **关键限制**：所有实验被试均按固定顺序完成两个会话，无 counterbalance，
> 故 `condition` 与 `session` 顺序完全共线，主效应只能解释为「会话差异」，
> 不能解释为「注意调制」（详见 `07_lmm.R` 内注释）。

---

## 2. ECG 与 R 峰处理（01）

- 官方 R 峰取自 `derivatives/.../desc-rpeak_timestamps.tsv.gz`（亚采样精度，抛物线插值）。
- 复检（01）：抽样 1/3 被试，用 `neurokit2`（Pan–Tompkins 类）自建检测对比。
  - 数量一致性标准：差异 < 1%。
  - 时间偏移：FFT 互相关估计全局偏移（≤300 ms 搜索窗），扣偏后 Hungarian 分配匹配（容差 50 ms）。
  - 扣偏后残留 RMSE < 10 ms（冻结判定标准）。
- **R 峰对齐验证**（Steinfath et al., 2026 强制校验）：每记录将 raw ECG epoch 到官方 R 峰，
  记录 ±50 ms 内 R 波峰延迟（`rpeak_lag_ms`），确认无滤波/下采样系统偏移。
- 最小 RR 间期 `MIN_RR = 0.5 s`，**在时间窗限制之后**施加，保证不重叠心搏。

---

## 3. EEG 预处理（02）

处理顺序：读 BDF → crop 到 `end` 事件 → 设 montage + 坏导检测/插值 → CAR → 带通滤波 →
ICA → 双轨输出。

**坏导检测**：峰峰值超过 `中位数 + 3 × 1.4826 × MAD`（幅值过大）或低于 0.5 µV（平坦/脱落），
球面样条插值。

**滤波（解耦，核心）**：
- 主分析信号：**0.1–30 Hz** 零相位 FIR（`firwin`，Hamming 窗 ~3.3 s，`filtfilt`），
  保留慢波 HEP（Tanner et al., 2015：高通 ≥0.3 Hz 会引入伪迹）。
- ICA 拟合信号：**1.0–30 Hz**（Winkler et al., 2015：去低频漂移）。
- 最小相位（因果）FIR 变体仅作多宇宙敏感性维度。

**ICA**：20 成分，extended infomax，`random_state=42`，`max_iter='auto'`。
在 1.0 Hz 高通信号上拟合，解混矩阵施加到 0.1 Hz 主分析信号（Tanner/Widmann 解耦策略）。

**成分识别（手动判据，透明可复现）**：
- 眼动：前额极通道（Fp1/Fp2/AF7/AF8/F7/F8）权重均值 > 0.3（`EYE_TOPO_THRESH`）。
- CFA：R 峰锁时（0–200 ms）能量比 > 0.3（`CFA_ENERGY_RATIO_THRESH`）。

**双轨输出**：同一 ICA 分解产出 `clean`（去眼动+去 CFA）与 `eyeonly`（仅去眼动），
另含因果敏感性轨（`*_causal_raw.fif`）。

---

## 4. HEP 提取（03b，Level 1）

- 时间窗 `TMIN/TMAX = −0.3 / 0.6 s`；最小 RR 0.5 s。
- **基线：主分析不做基线校正**（`APPLY_BASELINE = False`）。心跳周期性使任何基线窗
  都与前一搏动的 P 波/晚期 HEP 重叠（RR<0.9 s 占多数），基线校正会注入污染。
  `BASELINE_WIN = (−0.1, −0.025)` 仅作敏感性对照（"corrected"）。
- epoch 振幅拒绝：峰峰值 > 50 µV 剔除（`EPOCH_REJECT_UV = 50`）。
- **ROI × 窗口双报告**：frontocentral / central / parietal 三 ROI，
  主窗口 200–600 ms（`MAIN_WIN`）+ 早期对照窗 0–200 ms（`EARLY_WIN`）。
  每格同时输出**有符号均值电压**（保留极性）与 **RMS**（对极性鲁棒，主指标）。

**窗口依据**：HEP 峰值 >200 ms 额中央主导（Pollatos et al., 2005）；CFA 集中于 0–200 ms
（Park & Blanke, 2019）；早期 HEP 90–365 ms（Kandasamy et al., 2026）；Steinfath et al. (2026)
要求显式报告窗口。

---

## 5. Pseudotrial 校正（03c）

用**全局时间平移**构造伪 R 峰：真实 R 峰序列整体平移 5–15 s（随机，每记录一个值）并循环折回，
再按 0.5 s 最小 RR 过滤。校正 HEP = 真 HEP − 伪 HEP。

> 采用全局平移而非逐峰抖动（±0.25 s），因为逐峰抖动会让伪触发点落在 T 波区
> （QT 间期 250–400 ms），伪 HEP 会捕获倒置的心脏场伪迹（Steinfath et al., 2025）。

---

## 6. 单试次导出（03d）

- 单试次 HEP 窗口 `ST_WIN = (0.25, 0.40) s`（晚期 HEP / T 波核心段，Steinfath 2025），
  frontocentral ROI。
- 协变量：前一 RR 间期（`rr_prev`，基于**未 filter_rr** 的完整序列计算，保证生理真实），
  呼吸相位（`resp_phase`，仅 exp4/5）。
- 呼吸相位：两阶段峰谷插值（谷→峰 = 吸气 0→π，峰→谷 = 呼气 π→2π），
  保留 I:E 不对称（Zaccaro et al., 2018; Kluger et al., 2021）。

---

## 7. Surrogate 检验（04，Level 1）

- **时间平移**：R 峰序列平移 5–60 s（丢弃越界，不折回），`n_perm=1000`。
- **相位随机化**：ROI 通道 FFT → 随机化中间频率相位（DC/Nyquist 保持）→ IFFT，
  `n_perm=1000`（补零到 2 的幂加速）。
- 统计：单侧 `p = (1 + #{null ≥ obs}) / (N+1)`，全记录 BH-FDR 校正。

---

## 8. Level 2 心搏锁定相位一致性（05）

- 频带 delta(0.5–4)/theta(4–8)/alpha(8–13)/beta(13–30 Hz)，零相位 FIR + Hilbert 取相位。
- ITPC(τ) = |(1/K) Σ_k exp(i·φ_k(τ))|，主/早期窗口 + ROI 聚合。
- **surrogate**：R 峰抖动（尺度 = 频带几何中心周期 1 倍，下限 0.2 s），破坏心搏-EEG 对齐，
  `n_perm=1000`。

---

## 9. CFA 连续谱分析（06b/06c/06d，Paper A 核心）

- **阈值扫描**（06b）：阈值 {0.05,0.10,0.15,0.20,0.30,0.40,0.50} × 成分数 {15,20,25,30,40,50}
  交叉，重分类 CFA 并重算 clean RMS。
- **能量比分布**（06c）：固定 20 成分，导出每成分锁时能量比。
- **分布检验**（06d）：KDE 峰值数、Hartigan's dip test、偏度/峰度、阈值敏感性。

---

## 10. 多宇宙分析（06）

维度：track(clean/eyeonly) × 滤波相位(zero/minimum) × min-RR{0.5,0.65,0.7,0.75,0.8,0.88}
× epoch 拒绝{50,75,100 µV} × 基线{corrected,raw,wholecycle} × ROI{3} × 窗口{main,early}
= 2×2×6×3×3×3×2 = **1296 pipelines**。

每 pipeline 的配对状态效应以 Hedges' d_z（`1 − 3/(4(n−1)−1)` 小样本修正）+ 配对 t 检验量化，
全 pipeline BH-FDR 校正。

---

## 11. 统计模型（07_lmm.R）

- 主模型 `bhci ~ condition + (1|sub)`（随机斜率奇异时降级）；等价 `session` 模型做共线诊断。
- 刺激随机效应 `bhci ~ condition + (1|sub) + (1|stim)`。
- 基线敏感性：未校正（主）vs 校正（对照）系数对比。
- Pseudotrial 校正对比。
- 等效性检验（TOST），边界 {0.15, 0.19, 0.25}（Steinfath 2025 的 0.19 为主）。
- 单试次 `hep_amp ~ condition + rr_prev + resp_phase + (1|sub)`（resp_phase 仅 exp4/5）。
- 调节分析（探索性）：Sex/Age/Caffeine/Tired_rank 的 Spearman 相关 + condition×covariate 交互，BH-FDR。

---

## 12. 心率变异性（08_hrv.py）

- 异位搏动剔除：偏离中位数 >30% 的 RR 间隔（Task Force 1996 建议 <20–30%）。
- 时域：SDNN、RMSSD、mean RR、mean HR。
- 频域：Lomb–Scargle 周期图（不重采样），LF(0.04–0.15)/HF(0.15–0.40)，LF/HF 比 + nu 单位。

---

## 13. 跨实验汇总（09）

汇总各实验 LMM condition 系数与多宇宙 dz 分布（中位数/范围/显著 pipeline 数），
生成跨实验对比表。

---

## 14. 冻结参数速查

| 参数 | 值 | 位置 |
|------|-----|------|
| 主分析高通 | 0.1 Hz | `preprocess_utils.BANDPASS_LOW` |
| ICA 拟合高通 | 1.0 Hz | `preprocess_utils.ICA_HIGHPASS` |
| 低通 | 30 Hz | `preprocess_utils.BANDPASS_HIGH` |
| 基线校正 | 不做（主分析） | `common.APPLY_BASELINE` |
| BASELINE_WIN | −0.1 ~ −0.025 s | `common.BASELINE_WIN` |
| TMIN/TMAX | −0.3 / 0.6 s | `common` |
| MAIN_WIN / EARLY_WIN | 0.2–0.6 / 0.0–0.2 s | `common` |
| epoch 拒绝 | 50 µV | `common.EPOCH_REJECT_UV` |
| 最小 RR | 0.5 s | `common.MIN_RR` |
| CFA 能量比阈值 | 0.3 | `preprocess_utils.CFA_ENERGY_RATIO_THRESH` |
| 眼动地形图阈值 | 0.3 | `common.EYE_TOPO_THRESH` |
| ICA 成分数 | 20 | `02_preprocess` |
| 单试次窗口 | 0.25–0.40 s | `03d.ST_WIN` |

---

## 15. 参考文献

1. Steinfath, P., et al. (2026). Heartbeat-evoked responses in M/EEG: A systematic review. *Psychophysiology*, 63(4), e70297.
2. Steinfath, P., et al. (2025). Validating genuine changes in HEPs using pseudotrials and surrogate procedures. *Imaging Neuroscience*, 3, IMAG.a.30.
3. Kandasamy, R., et al. (2026). Silent brains isolate the heartbeat-evoked potential. *bioRxiv*.
4. Park, H.-D., & Blanke, O. (2019). Heartbeat-evoked cortical responses. *NeuroImage*, 197, 502–511.
5. Pollatos, O., et al. (2005). Brain structures involved in interoceptive awareness. *Human Brain Mapping*, 26(1), 54–64.
6. Winkler, I., et al. (2015). On the influence of high-pass filtering on ICA-based artifact reduction. *EMBC*, 4101–4105.
7. Tanner, D., et al. (2015). How inappropriate high-pass filters can produce artifactual effects. *Psychophysiology*, 52(8), 997–1009.
8. Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for electrophysiological data. *Journal of Neuroscience Methods*, 250, 34–46.
9. Coll, M.-P., et al. (2021). Systematic review and meta-analysis of the HEP and interoception. *Neuroscience & Biobehavioral Reviews*, 122, 190–200.
10. Madsen, J., Kuppa, N., & Parra, L. (2026). The Brain, Body, and Behaviour Dataset. *Scientific Data*.
11. Task Force of the ESC/NASPE (1996). Heart rate variability: standards of measurement. *European Heart Journal*, 17(3), 354–381.
12. Zaccaro, A., et al. (2018). How breath-control can change your life. *Frontiers in Human Neuroscience*, 12, 353.
13. Kluger, D. S., et al. (2021). Respiration modulates oscillatory neural network activity. *NeuroImage*, 226, 117556.
