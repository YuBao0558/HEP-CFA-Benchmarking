#!/usr/bin/env python3
"""三样本 sin 系数 meta 分析的估计器敏感性（DL / REML / Bayesian / 固定效应）。

目的：巩固 k=3 的 meta 结论。DerSimonian-Laird 在小 k 时对 tau² 的估计有偏，
补 REML（限制最大似然）与 Bayesian（半正态先验）估计作为敏感性分析，
并报告 tau² 的多种估计与 I²，说明 I²=0% 的结论对估计器选择稳健。

方法：
  - DerSimonian-Laird（现有主结果）
  - REML：迭代求解 tau²（方差分量最大化限制似然）
  - Bayesian：mu ~ Normal(0, 1e6), tau ~ HalfNormal(0, 1)，
              用网格近似（闭式后验，无 MCMC 依赖）
  - 固定效应：逆方差加权，无 tau²

数据：与 dir1_meta_forest.py 相同。
  BBBD: analysis/data/respiration_coupling/{exp4,exp5}_resp_modulation_subject.csv
  MultiPhysio: /Users/yubao/CODE/public/MultiPhysio-HRC/analysis/resp_modulation_subject.csv

产出: analysis/data/respiration_coupling/robustness/meta_estimator_sensitivity.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar

BBBD = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling")
MPH = Path("/Users/yubao/CODE/public/MultiPhysio-HRC/analysis")
OUT = BBBD / "robustness"
OUT.mkdir(parents=True, exist_ok=True)


def load_sin():
    """返回 (coefs, ses) 数组，MultiPhysio 极性翻转（对齐后）。"""
    coefs, ses = [], []
    for exp in ["experiment4", "experiment5"]:
        d = pd.read_csv(BBBD / f"{exp}_resp_modulation_subject.csv")
        coefs.append(d["sin_coef"].values)
        ses.append(d["sin_se"].values)
    d = pd.read_csv(MPH / "resp_modulation_subject.csv")
    se = np.abs(d["sin_coef"].values / d["sin_t"].values)  # 从 t 反推
    coefs.append(-d["sin_coef"].values)  # 极性翻转
    ses.append(se)
    coefs = np.concatenate(coefs)
    ses = np.concatenate(ses)
    mask = np.isfinite(ses) & (ses > 0)
    return coefs[mask], ses[mask]


def dersimonian_laird(coefs, ses):
    w = 1.0 / ses ** 2
    mu_fixed = np.sum(w * coefs) / np.sum(w)
    Q = np.sum(w * (coefs - mu_fixed) ** 2)
    k = len(coefs)
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_rand = 1.0 / (ses ** 2 + tau2)
    mu = np.sum(w_rand * coefs) / np.sum(w_rand)
    se = np.sqrt(1.0 / np.sum(w_rand))
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    return mu, se, tau2, Q, I2


def reml(coefs, ses):
    """REML 估计 tau²（迭代求解限制对数似然的极值）。"""
    def neg_log_restricted(tau2):
        tau2 = max(tau2, 1e-12)
        v = ses ** 2 + tau2
        w = 1.0 / v
        mu = np.sum(w * coefs) / np.sum(w)
        # 限制对数似然（剖面）
        ll = -0.5 * (np.sum(np.log(v)) + np.sum(w * (coefs - mu) ** 2)
                     + np.log(np.sum(w)))
        return -ll

    res = minimize_scalar(neg_log_restricted, bounds=(0, 1.0), method="bounded")
    tau2 = max(res.x, 0.0)
    v = ses ** 2 + tau2
    w = 1.0 / v
    mu = np.sum(w * coefs) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    Q = np.sum((1.0 / ses ** 2) * (coefs - np.sum(coefs / ses ** 2) / np.sum(1.0 / ses ** 2)) ** 2)
    k = len(coefs)
    I2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0
    return mu, se, tau2, Q, I2


def bayesian(coefs, ses, tau_grid=None):
    """Bayesian 随机效应 meta（网格近似，HalfNormal(0,1) tau 先验）。

    闭式后验：对每个 tau 网格点，mu 的条件后验为正态，
    权重为 tau 的边缘似然。返回后验均值与 95% 最高密度区间。
    """
    if tau_grid is None:
        tau_grid = np.linspace(0, 0.5, 2001)
    log_weights = []
    mu_post = []
    mu_sd_post = []
    for tau in tau_grid:
        v = ses ** 2 + tau ** 2
        w = 1.0 / v
        mu = np.sum(w * coefs) / np.sum(w)
        se_mu = np.sqrt(1.0 / np.sum(w))
        # 边缘似然（积分掉 mu）
        log_lik = -0.5 * np.sum(np.log(2 * np.pi * v)) \
                  - 0.5 * np.sum((coefs - mu) ** 2 / v)
        # HalfNormal(0,1) 先验对数密度
        log_prior = -0.5 * tau ** 2  # 忽略常数
        log_weights.append(log_lik + log_prior)
        mu_post.append(mu)
        mu_sd_post.append(se_mu)
    log_weights = np.array(log_weights)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    mu_post = np.array(mu_post)
    mu_sd_post = np.array(mu_sd_post)
    # 后验 mu 均值（tau 混合后）
    mu_mean = np.sum(weights * mu_post)
    # 后验 tau 均值
    tau_mean = np.sum(weights * tau_grid)
    # mu 的混合方差（within + between）
    mu_var = np.sum(weights * mu_sd_post ** 2) + np.sum(weights * (mu_post - mu_mean) ** 2)
    se = np.sqrt(mu_var)
    # 95% 可信区间（正态近似）
    ci_lo = mu_mean - 1.96 * se
    ci_hi = mu_mean + 1.96 * se
    return mu_mean, se, tau_mean, ci_lo, ci_hi


def fixed_effect(coefs, ses):
    w = 1.0 / ses ** 2
    mu = np.sum(w * coefs) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    return mu, se


def main():
    coefs, ses = load_sin()
    k = len(coefs)
    print(f"纳入样本（被试数）: k={k} (n={len(coefs)} 被试)")

    rows = []

    mu, se, tau2, Q, I2 = dersimonian_laird(coefs, ses)
    z = mu / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows.append({"estimator": "DerSimonian-Laird", "mu": mu, "se": se,
                 "z": z, "p": p, "tau2": tau2, "I2": I2, "Q": Q})
    print(f"\n[DL]   mu={mu:.4f} SE={se:.4f} z={z:.2f} p={p:.4g} "
          f"tau²={tau2:.5f} I²={I2:.0f}% Q={Q:.2f}")

    mu, se, tau2, Q, I2 = reml(coefs, ses)
    z = mu / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows.append({"estimator": "REML", "mu": mu, "se": se,
                 "z": z, "p": p, "tau2": tau2, "I2": I2, "Q": Q})
    print(f"[REML] mu={mu:.4f} SE={se:.4f} z={z:.2f} p={p:.4g} "
          f"tau²={tau2:.5f} I²={I2:.0f}% Q={Q:.2f}")

    mu, se, tau_mean, ci_lo, ci_hi = bayesian(coefs, ses)
    z = mu / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows.append({"estimator": "Bayesian(HalfNormal)", "mu": mu, "se": se,
                 "z": z, "p": p, "tau2": tau_mean ** 2, "I2": np.nan, "Q": np.nan,
                 "ci_lo": ci_lo, "ci_hi": ci_hi})
    print(f"[Bayes] mu={mu:.4f} SE={se:.4f} z={z:.2f} p={p:.4g} "
          f"tau均值={tau_mean:.4f} 95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    mu, se = fixed_effect(coefs, ses)
    z = mu / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows.append({"estimator": "Fixed-effect", "mu": mu, "se": se,
                 "z": z, "p": p, "tau2": 0.0, "I2": np.nan, "Q": Q})
    print(f"[Fixed] mu={mu:.4f} SE={se:.4f} z={z:.2f} p={p:.4g}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "meta_estimator_sensitivity.csv", index=False)
    print("\n产出:", OUT / "meta_estimator_sensitivity.csv")


if __name__ == "__main__":
    main()
