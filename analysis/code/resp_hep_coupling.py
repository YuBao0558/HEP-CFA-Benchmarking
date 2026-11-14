#!/usr/bin/env python3
"""呼吸-心搏双重锁时耦合分析 (BBBD experiment 4/5)。

方向 A：呼吸相位对心搏锁定的脑活动 (HEP) 的调制。

核心命题：HEP 幅度中相当一部分是呼吸锁时的，而非纯粹心搏锁时的；
呼吸相位是一个被现有 HEP 文献普遍忽视的、可能强于 CFA 阈值的 HEP 调制源。

方法 (对齐冻结管线 pipeline/03d_singletrial.py 的 hep_singletrial.csv)：
  1. 单试次 circular-linear 回归 (within-subject z 化)：
       z(hep_amp) ~ sin(resp_phase) + cos(resp_phase) + rr_prev_s
     提取每个被试的 sin/cos/rr 系数，跨被试 one-sample t (meta)。
  2. 呼吸调制幅度 (µV)：用与主回归同口径的完整回归在原始 µV 尺度上估计
     sin/cos 联合幅度，量化呼吸贡献的生理学占比。稳健统计 (中位数 + MAD)，
     并剔除 hep 基线接近 0 的离群被试 (占比除法会爆炸)。
  3. 呼吸-ITPC：R 峰时刻呼吸相位的跨心跳一致性，与"呼吸相位仅随机"的
     均匀分布对照 (Rayleigh 检验)。
  4. 呼吸相位分箱 (8 bins) 的 HEP 均值曲线 (跨被试平均呼吸锁定波形)。
  5. 相对贡献分解：比较 [sin+cos] vs [rr] 的增量 R²，量化呼吸 vs 心搏前间期
     对单试次 HEP 波动的相对解释力。

依赖：pipeline/pipeline_data/data/{experiment4,5}/hep_singletrial.csv
      (列: sub/ses/stim/condition/hep_amp_uv/hep_amp_rms_uv/rr_prev_s/resp_phase)

产出 (写到 analysis/data/respiration_coupling/)：
  - resp_modulation_subject.csv    每被试 sin/cos/rr 系数 + 拟合统计
  - resp_modulation_meta.csv       跨被试 meta (one-sample t)
  - resp_amplitude_contribution.csv 呼吸峰-峰幅度 vs HEP 均值 (稳健)
  - resp_itpc.csv                  呼吸相位 ITPC + Rayleigh 检验
  - resp_relative_contribution.csv sin/cos vs rr 的增量 R² 分解
  - resp_phase_binned_hep.csv      呼吸相位分箱 HEP 曲线
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

PIPE = Path("/Users/yubao/CODE/public/BBBD/pipeline/pipeline_data/data")
OUT = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling")
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["experiment4", "experiment5"]
N_BINS = 8


def load_singletrial(exp):
    """读单试次 HEP 长表，只保留呼吸相位与 rr_prev 均非空的试次。"""
    p = PIPE / exp / "hep_singletrial.csv"
    df = pd.read_csv(p)
    df = df.dropna(subset=["hep_amp_uv", "resp_phase", "rr_prev_s"])
    return df


def circular_linear_per_subject(df):
    """每被试: z(hep) ~ sin(phase) + cos(phase) + rr_prev_s。返回系数 DataFrame。"""
    rows = []
    for sub, g in df.groupby("sub"):
        sd = g["hep_amp_uv"].std()
        if sd == 0 or np.isnan(sd):
            continue
        z = (g["hep_amp_uv"] - g["hep_amp_uv"].mean()) / sd
        sinp = np.sin(g["resp_phase"].values)
        cosp = np.cos(g["resp_phase"].values)
        rr = g["rr_prev_s"].values.astype(float)
        X = sm.add_constant(np.column_stack([sinp, cosp, rr]))
        m = sm.OLS(z.values, X).fit()
        rows.append({
            "sub": sub,
            "sin_coef": m.params[1],
            "cos_coef": m.params[2],
            "rr_coef": m.params[3],
            "sin_se": m.bse[1],
            "cos_se": m.bse[2],
            "rr_se": m.bse[3],
            "sin_t": m.tvalues[1],
            "cos_t": m.tvalues[2],
            "rr_t": m.tvalues[3],
            "n_trials": len(g),
            "r2": m.rsquared,
        })
    return pd.DataFrame(rows)


def meta_one_sample(sub_df):
    """跨被试 one-sample t + 稳健汇总统计。"""
    out = {}
    for name in ["sin_coef", "cos_coef", "rr_coef"]:
        v = sub_df[name].values
        t, p = stats.ttest_1samp(v, 0)
        out[f"{name}_mean"] = np.mean(v)
        out[f"{name}_median"] = np.median(v)
        out[f"{name}_sd"] = np.std(v, ddof=1)
        out[f"{name}_t"] = t
        out[f"{name}_p"] = p
        out[f"{name}_pct_same_sign"] = float((np.sign(v) == np.sign(np.mean(v))).mean())
    out["n_subjects"] = len(sub_df)
    out["n_trials_total"] = int(sub_df["n_trials"].sum())
    return out


def amplitude_contribution(df, hep_floor_uv=0.02):
    """呼吸调制幅度 (µV 峰-峰) vs HEP 幅度 (RMS)。

    用与主回归同口径的完整回归 (sin + cos + rr) 在原始 µV 尺度估计：
      hep_amp_uv ~ sin(phase) + cos(phase) + rr_prev_s
    sin/cos 联合幅度 = 2 * sqrt(sin^2 + cos^2) (峰-峰近似)。

    参照: 呼吸调制幅度 vs HEP 的 RMS (hep_amp_rms_uv, 单试次波动幅度) 与
    被试内 SD。有符号均值 (hep_amp_uv.mean) 因正负抵消趋近 0, 不适合当分母；
    用 RMS / SD 才能正确量化"呼吸调制在 HEP 波动中占多大比例"。
    """
    rows = []
    for sub, g in df.groupby("sub"):
        sinp = np.sin(g["resp_phase"].values)
        cosp = np.cos(g["resp_phase"].values)
        rr = g["rr_prev_s"].values.astype(float)
        X = sm.add_constant(np.column_stack([sinp, cosp, rr]))
        m = sm.OLS(g["hep_amp_uv"].values, X).fit()
        sin_uv = m.params[1]
        cos_uv = m.params[2]
        joint_amp_uv = 2.0 * np.sqrt(sin_uv ** 2 + cos_uv ** 2)  # 峰-峰
        rms = g["hep_amp_rms_uv"].mean()
        sd = g["hep_amp_uv"].std()
        rows.append({
            "sub": sub,
            "resp_sin_coef_uv": sin_uv,
            "resp_cos_coef_uv": cos_uv,
            "resp_joint_pp_uv": joint_amp_uv,
            "hep_mean_uv": g["hep_amp_uv"].mean(),
            "hep_rms_uv": rms,
            "hep_sd_uv": sd,
            "n_trials": len(g),
        })
    r = pd.DataFrame(rows)
    # 占比: 呼吸峰-峰幅度 / HEP 的 RMS (单试次波动), 用 RMS 作参照
    # (RMS 始终 > 0, 无符号抵消问题; 仅在 RMS 过小时剔除)
    r["valid_pct"] = r["hep_rms_uv"] >= hep_floor_uv
    r["resp_pp_pct_of_hep_rms"] = np.where(
        r["valid_pct"],
        r["resp_joint_pp_uv"] / r["hep_rms_uv"] * 100.0,
        np.nan,
    )
    return r


def itpc(df):
    """R 峰时刻呼吸相位的跨心跳一致性 (ITPC) + Rayleigh 检验。

    ITPC = |(1/K) sum_k exp(i·phase_k)|。相位是每个 R 峰的呼吸相位，
    若呼吸相位与心跳完全无关 (随机均匀)，ITPC 期望趋近 0。
    """
    rows = []
    for sub, g in df.groupby("sub"):
        ph = g["resp_phase"].values.astype(float)
        z = np.exp(1j * ph)
        itpc = np.abs(z.mean())
        k = len(ph)
        # Rayleigh 检验 (均匀分布零假设)
        r = itpc
        p = _rayleigh_p(r, k)
        rows.append({
            "sub": sub,
            "itpc": itpc,
            "mean_phase": float(np.angle(z.mean())),
            "n_trials": k,
            "rayleigh_p": p,
        })
    r = pd.DataFrame(rows)
    return r


def _rayleigh_p(r, n):
    """Rayleigh 检验 p 值 (大样本近似, Fisher 1993)。"""
    # p = exp(-n r^2) 是标准近似 (n 大时)
    p = np.exp(-n * r ** 2)
    return float(min(max(p, 0.0), 1.0))


def relative_contribution(df):
    """sin/cos (呼吸) vs rr (心搏前间期) 对单试次 HEP 波动的相对贡献。

    每个被试: 完整模型 z(hep) ~ sin+cos+rr 的 R²；
              去掉 sin/cos 后 (仅 rr) 的 R² 增量 = 呼吸的增量贡献；
              去掉 rr 后 (仅 sin/cos) 的 R² 增量 = 心搏前间期的增量贡献。
    """
    rows = []
    for sub, g in df.groupby("sub"):
        sd = g["hep_amp_uv"].std()
        if sd == 0 or np.isnan(sd):
            continue
        z = (g["hep_amp_uv"] - g["hep_amp_uv"].mean()) / sd
        sinp = np.sin(g["resp_phase"].values)
        cosp = np.cos(g["resp_phase"].values)
        rr = g["rr_prev_s"].values.astype(float)

        X_full = sm.add_constant(np.column_stack([sinp, cosp, rr]))
        X_noresp = sm.add_constant(rr)
        X_norr = sm.add_constant(np.column_stack([sinp, cosp]))

        r2_full = sm.OLS(z, X_full).fit().rsquared
        r2_noresp = sm.OLS(z, X_noresp).fit().rsquared
        r2_norr = sm.OLS(z, X_norr).fit().rsquared

        rows.append({
            "sub": sub,
            "r2_full": r2_full,
            "r2_resp_only": r2_full - r2_noresp,   # 呼吸的增量贡献
            "r2_rr_only": r2_full - r2_norr,        # 心搏前间期的增量贡献
            "n_trials": len(g),
        })
    return pd.DataFrame(rows)


def phase_binned_hep(df):
    """呼吸相位 8 分箱的 HEP 均值 (within-subject z 化后平均)。"""
    df = df.copy()
    df["z"] = df.groupby("sub")["hep_amp_uv"].transform(
        lambda x: (x - x.mean()) / x.std())
    df["bin"] = (df["resp_phase"] / (2 * np.pi) * N_BINS).astype(int) % N_BINS
    g = df.groupby("bin")["z"].agg(["mean", "std", "count"]).reset_index()
    g["phase_deg"] = g["bin"] / N_BINS * 360
    return g


def summarize(x):
    """稳健汇总: n / mean / median / sd / MAD。"""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return dict(n=len(x), mean=float(np.mean(x)), median=float(med),
                sd=float(np.std(x, ddof=1)), mad=float(mad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="all")
    ap.add_argument("--hep-floor-uv", type=float, default=0.02,
                    help="|hep_mean_uv| 下限, 低于此值不计算占比 (防除法爆炸)")
    args = ap.parse_args()

    exps = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    meta_rows = []
    contrib_all = []
    itpc_all = []
    rel_all = []
    binned_all = []

    for exp in exps:
        df = load_singletrial(exp)
        sub_df = circular_linear_per_subject(df)
        sub_df.to_csv(OUT / f"{exp}_resp_modulation_subject.csv", index=False)

        meta = meta_one_sample(sub_df)
        meta["experiment"] = exp
        meta_rows.append(meta)

        contrib = amplitude_contribution(df, hep_floor_uv=args.hep_floor_uv)
        contrib["experiment"] = exp
        contrib_all.append(contrib)

        ipc = itpc(df)
        ipc["experiment"] = exp
        itpc_all.append(ipc)

        rel = relative_contribution(df)
        rel["experiment"] = exp
        rel_all.append(rel)

        binned = phase_binned_hep(df)
        binned["experiment"] = exp
        binned_all.append(binned)

        print(f"[{exp}] n_sub={meta['n_subjects']} n_trials={meta['n_trials_total']}")
        print(f"  sin : t={meta['sin_coef_t']:.3f} p={meta['sin_coef_p']:.4g} "
              f"median={meta['sin_coef_median']:.4f}")
        print(f"  cos : t={meta['cos_coef_t']:.3f} p={meta['cos_coef_p']:.4g} "
              f"median={meta['cos_coef_median']:.4f}")
        print(f"  rr  : t={meta['rr_coef_t']:.3f} p={meta['rr_coef_p']:.4g} "
              f"median={meta['rr_coef_median']:.4f}")

        # 稳健幅度占比
        valid = contrib[contrib["valid_pct"]]
        print(f"  呼吸峰-峰幅度 (µV, 中位): {valid['resp_joint_pp_uv'].median():.4f}")
        print(f"  占比 vs HEP-RMS (%, 中位, 剔除 {len(contrib)-len(valid)} 个 hep≈0 被试): "
              f"{valid['resp_pp_pct_of_hep_rms'].median():.1f}")

        # ITPC 汇总
        ipc_sig = (ipc["rayleigh_p"] < 0.05).sum()
        print(f"  ITPC 中位={ipc['itpc'].median():.4f} | Rayleigh p<0.05 的被试数="
              f"{ipc_sig}/{len(ipc)}")

        # 相对贡献汇总
        print(f"  R² 增量: 呼吸(中位)={rel['r2_resp_only'].median():.5f} "
              f"心搏前间期(中位)={rel['r2_rr_only'].median():.5f}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUT / "resp_modulation_meta.csv", index=False)

    contrib_df = pd.concat(contrib_all, ignore_index=True)
    contrib_df.to_csv(OUT / "resp_amplitude_contribution.csv", index=False)

    itpc_df = pd.concat(itpc_all, ignore_index=True)
    itpc_df.to_csv(OUT / "resp_itpc.csv", index=False)

    rel_df = pd.concat(rel_all, ignore_index=True)
    rel_df.to_csv(OUT / "resp_relative_contribution.csv", index=False)

    binned_df = pd.concat(binned_all, ignore_index=True)
    binned_df.to_csv(OUT / "resp_phase_binned_hep.csv", index=False)

    print("\n产出目录:", OUT)
    print("  resp_modulation_subject.csv       每被试 sin/cos/rr 系数")
    print("  resp_modulation_meta.csv          跨被试 meta (one-sample t)")
    print("  resp_amplitude_contribution.csv   呼吸峰-峰幅度 vs HEP (稳健)")
    print("  resp_itpc.csv                     呼吸相位 ITPC + Rayleigh")
    print("  resp_relative_contribution.csv    sin/cos vs rr 增量 R² 分解")
    print("  resp_phase_binned_hep.csv         呼吸相位分箱 HEP 曲线")


if __name__ == "__main__":
    main()
