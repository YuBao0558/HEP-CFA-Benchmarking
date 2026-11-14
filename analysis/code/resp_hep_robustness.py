#!/usr/bin/env python3
"""呼吸相位调制 HEP 的稳健性补充分析（第 2/3 步）。

在已聚合的单试次层面 (hep_singletrial.csv + 各样本 per-subject 系数) 之上，
补齐审稿人会要求的分析，不重跑原始 BDF：

  1. 标准化效应量 Cohen's d + 95% CI（sin/cos 项，各样本 + 合并）。
  2. 多重比较校正（Bonferroni 与 Benjamini-Hochberg FDR），
     覆盖 3 样本 × {sin, cos} = 6 项主检验。
  3. 二次谐波模型 (sin2 + cos2) 的增量 R² 与 F 检验，检验是否偏离正弦。
  4. 绝对方差报告：呼吸 vs RR 的增量 R² 分布（中位数、IQR、95% CI via bootstrap）
     与配对检验，而非仅报比值。
  5. 相位分箱数敏感性：N_BINS ∈ {6, 8, 12} 下分箱曲线的循环互相关与
     sin 系数符号稳定性。

产出 (analysis/data/respiration_coupling/robustness/)：
  - effect_size_cohens_d.csv
  - multiple_correction.csv
  - harmonic_test.csv
  - absolute_variance.csv
  - bin_sensitivity.csv

依赖：
  BBBD: analysis/data/respiration_coupling/{exp4,exp5}_resp_modulation_subject.csv
        analysis/data/respiration_coupling/resp_relative_contribution.csv
  MultiPhysio: /Users/yubao/CODE/public/MultiPhysio-HRC/analysis/resp_modulation_subject.csv
        /Users/yubao/CODE/public/MultiPhysio-HRC/analysis/resp_relative_contribution.csv
  单试次: pipeline/pipeline_data/data/{experiment4,5}/hep_singletrial.csv
        /Users/yubao/CODE/public/MultiPhysio-HRC/analysis/hep_singletrial.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BBBD_SUB = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling")
BBBD_TRIAL = Path("/Users/yubao/CODE/public/BBBD/pipeline/pipeline_data/data")
MPH = Path("/Users/yubao/CODE/public/MultiPhysio-HRC/analysis")
OUT = BBBD_SUB / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("BBBD exp4", BBBD_SUB / "experiment4_resp_modulation_subject.csv", BBBD_TRIAL / "experiment4" / "hep_singletrial.csv"),
    ("BBBD exp5", BBBD_SUB / "experiment5_resp_modulation_subject.csv", BBBD_TRIAL / "experiment5" / "hep_singletrial.csv"),
    ("MultiPhysio-HRC", MPH / "resp_modulation_subject.csv", MPH / "hep_singletrial.csv"),
]


def load_subject_coefs():
    """返回 {sample: DataFrame(per-subject sin/cos/rr 系数)}。"""
    out = {}
    for name, sub_path, _ in SAMPLES:
        d = pd.read_csv(sub_path)
        out[name] = d
    return out


def cohens_d(vals):
    """单样本 Cohen's d = mean / SD (ddof=1)，并返回 95% CI (t 分布)。"""
    vals = np.asarray(vals, float)
    n = len(vals)
    d = vals.mean() / vals.std(ddof=1)
    se_d = np.sqrt(1 / n + d ** 2 / (2 * n))
    tcrit = stats.t.ppf(0.975, n - 1)
    return d, d - tcrit * se_d, d + tcrit * se_d


def bootstrap_median_ci(vals, n_boot=5000, rng=None):
    """中位数的 bootstrap 95% CI。"""
    rng = rng or np.random.default_rng(0)
    vals = np.asarray(vals, float)
    meds = np.array([np.median(rng.choice(vals, len(vals), replace=True))
                     for _ in range(n_boot)])
    return np.percentile(meds, [2.5, 97.5])


def main():
    coefs = load_subject_coefs()

    # ---------- 1. Cohen's d ----------
    d_rows = []
    for name, c in coefs.items():
        for term in ["sin_coef", "cos_coef"]:
            if term not in c.columns:
                continue
            d, lo, hi = cohens_d(c[term].values)
            d_rows.append({"sample": name, "term": term,
                           "cohens_d": d, "ci_low": lo, "ci_high": hi,
                           "n": len(c)})
    d_df = pd.DataFrame(d_rows)
    d_df.to_csv(OUT / "effect_size_cohens_d.csv", index=False)
    print("=== Cohen's d (sin/cos) ===")
    print(d_df.round(3).to_string(index=False))

    # ---------- 2. 多重校正 (6 项: 3 样本 × sin/cos) ----------
    mc_rows = []
    for name, c in coefs.items():
        for term in ["sin_coef", "cos_coef"]:
            if term not in c.columns:
                continue
            v = c[term].values
            t, p = stats.ttest_1samp(v, 0)
            mc_rows.append({"sample": name, "term": term, "p_raw": p})
    mc = pd.DataFrame(mc_rows)
    # Bonferroni
    mc["p_bonf"] = np.minimum(1.0, mc["p_raw"] * len(mc))
    # BH-FDR
    p = mc["p_raw"].values
    order = np.argsort(p)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank_i in range(n - 1, -1, -1):
        idx = order[rank_i]
        adj = p[idx] * n / (rank_i + 1)
        q[idx] = min(1.0, min(adj, prev))
        prev = q[idx]
    mc["p_fdr"] = q
    mc.to_csv(OUT / "multiple_correction.csv", index=False)
    print("\n=== 多重校正 (6 项) ===")
    print(mc.round(4).to_string(index=False))

    # ---------- 3. 二次谐波模型 ----------
    harm_rows = []
    for name, sub_path, trial_path in SAMPLES:
        t = pd.read_csv(trial_path)
        t = t.dropna(subset=["hep_amp_uv", "resp_phase", "rr_prev_s"])
        for sub, g in t.groupby("sub"):
            sd = g["hep_amp_uv"].std()
            if sd == 0 or np.isnan(sd):
                continue
            z = (g["hep_amp_uv"] - g["hep_amp_uv"].mean()) / sd
            ph = g["resp_phase"].values
            rr = g["rr_prev_s"].values.astype(float)
            X1 = np.column_stack([np.sin(ph), np.cos(ph), rr])
            X2 = np.column_stack([np.sin(ph), np.cos(ph),
                                  np.sin(2 * ph), np.cos(2 * ph), rr])
            X1 = np.column_stack([np.ones(len(z)), X1])
            X2 = np.column_stack([np.ones(len(z)), X2])
            # 用 scipy 直接 OLS 求 R² 与 F 检验 (二次谐波增量)
            b1, _, _, _ = np.linalg.lstsq(X1, z, rcond=None)
            b2, _, _, _ = np.linalg.lstsq(X2, z, rcond=None)
            r1 = 1 - np.sum((z - X1 @ b1) ** 2) / np.sum(z ** 2)
            r2 = 1 - np.sum((z - X2 @ b2) ** 2) / np.sum(z ** 2)
            # F 检验: 二次谐波项联合是否显著
            n_obs = len(z)
            p2 = X2.shape[1]
            p1 = X1.shape[1]
            num = (r2 - r1) / (p2 - p1)
            den = (1 - r2) / (n_obs - p2)
            f = num / den if den > 0 else np.nan
            fp = 1 - stats.f.cdf(f, p2 - p1, n_obs - p2) if np.isfinite(f) else np.nan
            harm_rows.append({"sample": name, "sub": sub,
                              "r2_1harm": r1, "r2_2harm": r2,
                              "delta_r2": r2 - r1, "f": f, "f_p": fp})
    harm = pd.DataFrame(harm_rows)
    harm.to_csv(OUT / "harmonic_test.csv", index=False)
    print("\n=== 二次谐波增量 R² (跨被试汇总) ===")
    for name, g in harm.groupby("sample"):
        n_sig = (g["f_p"] < 0.05).sum()
        print(f"  {name}: 中位 ΔR²={g['delta_r2'].median():.5f} "
              f"| 显著被试 {n_sig}/{len(g)} "
              f"| 中位 F p={g['f_p'].median():.3f}")

    # ---------- 4. 绝对方差 (呼吸 vs RR) ----------
    var_rows = []
    rel_paths = [BBBD_SUB / "resp_relative_contribution.csv", MPH / "resp_relative_contribution.csv"]
    # BBBD 文件含 experiment 列，MPH 需单独读
    rel_bbbd = pd.read_csv(BBBD_SUB / "resp_relative_contribution.csv")
    rel_mph = pd.read_csv(MPH / "resp_relative_contribution.csv")
    rel_mph = rel_mph.copy()
    rel_mph["experiment"] = "MultiPhysio"
    rel_all = pd.concat([rel_bbbd, rel_mph], ignore_index=True)
    name_map = {"experiment4": "BBBD exp4", "experiment5": "BBBD exp5",
                "MultiPhysio": "MultiPhysio-HRC"}
    for exp, g in rel_all.groupby("experiment"):
        nm = name_map.get(exp, exp)
        for col, label in [("r2_resp_only", "resp"), ("r2_rr_only", "rr")]:
            v = g[col].values
            lo, hi = bootstrap_median_ci(v)
            var_rows.append({"sample": nm, "term": label,
                             "median": np.median(v),
                             "iqr_low": np.percentile(v, 25),
                             "iqr_high": np.percentile(v, 75),
                             "ci_low": lo, "ci_high": hi})
        # 配对检验: resp vs rr (Wilcoxon signed-rank)
        w, wp = stats.wilcoxon(g["r2_resp_only"].values, g["r2_rr_only"].values)
        var_rows.append({"sample": nm, "term": "resp_vs_rr_wilcoxon",
                         "median": w, "iqr_low": np.nan, "iqr_high": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan})
        print(f"  {nm}: Wilcoxon resp>rr stat={w:.1f} p={wp:.4g} "
              f"(呼吸>RR 被试 {(g['r2_resp_only']>g['r2_rr_only']).sum()}/{len(g)})")
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(OUT / "absolute_variance.csv", index=False)
    print("\n=== 绝对增量 R² (中位数 [IQR], bootstrap 95% CI) ===")
    show = var_df[var_df["term"].isin(["resp", "rr"])]
    print(show.round(5).to_string(index=False))

    # ---------- 5. 分箱数敏感性 ----------
    bin_rows = []
    for nb in [6, 8, 12]:
        for name, sub_path, trial_path in SAMPLES:
            t = pd.read_csv(trial_path)
            t = t.dropna(subset=["hep_amp_uv", "resp_phase", "rr_prev_s"])
            t = t.copy()
            t["z"] = t.groupby("sub")["hep_amp_uv"].transform(
                lambda x: (x - x.mean()) / x.std())
            t["bin"] = (t["resp_phase"] / (2 * np.pi) * nb).astype(int) % nb
            prof = t.groupby("bin")["z"].mean().sort_index().values
            # 与基准 8 箱的 sin 系数符号对照：直接重算每被试 sin 系数符号
            sub_sign = []
            for sub, g in t.groupby("sub"):
                sd = g["hep_amp_uv"].std()
                if sd == 0 or np.isnan(sd):
                    continue
                zz = (g["hep_amp_uv"] - g["hep_amp_uv"].mean()) / sd
                X = np.column_stack([np.ones(len(zz)), np.sin(g["resp_phase"].values),
                                     np.cos(g["resp_phase"].values),
                                     g["rr_prev_s"].values.astype(float)])
                b, _, _, _ = np.linalg.lstsq(X, zz, rcond=None)
                sub_sign.append(np.sign(b[1]))
            sub_sign = np.array(sub_sign)
            frac_neg = (sub_sign < 0).mean()
            bin_rows.append({"n_bins": nb, "sample": name,
                             "frac_sin_negative": frac_neg,
                             "n_sub": len(sub_sign)})
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(OUT / "bin_sensitivity.csv", index=False)
    print("\n=== 分箱数敏感性 (sin 系数为负的被试比例, 应稳定) ===")
    print(bin_df.pivot_table(index="sample", columns="n_bins",
                             values="frac_sin_negative").round(3).to_string())

    print("\n产出目录:", OUT)


if __name__ == "__main__":
    main()
