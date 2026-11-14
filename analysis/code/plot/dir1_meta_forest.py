#!/usr/bin/env python3
"""三样本 sin 系数随机效应 meta 分析。

合并 BBBD exp4/exp5 与 MultiPhysio-HRC 的 per-subject sin 系数，
做随机效应 meta（DerSimonian-Laird），检验呼吸相位调制 HEP 的跨样本
合并效应。

注意：MultiPhysio 的 sin 系数方向需先翻转（-1 ×），因为呼吸信号极性相反
（180° 相位平移）。同时报告"翻转前/后"两种 meta，展示方向差异是极性而非
生理差异。

产出：analysis/figures/fig_dir1_meta_forest.pdf + meta 结果打印
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BBBD = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling")
MPH = Path("/Users/yubao/CODE/public/MultiPhysio-HRC/analysis")
FIG = Path("/Users/yubao/CODE/public/BBBD/analysis/figures")


def load_sin():
    datasets = []
    # BBBD exp4/5
    for exp in ["experiment4", "experiment5"]:
        d = pd.read_csv(BBBD / f"{exp}_resp_modulation_subject.csv")
        datasets.append(("BBBD " + exp.replace("experiment", "exp"),
                         d["sin_coef"].values, d["sin_se"].values))
    # MultiPhysio
    d = pd.read_csv(MPH / "resp_modulation_subject.csv")
    # 注意：MultiPhysio 无 sin_se，需从 t 值反推 se = coef/t
    se = d["sin_coef"].values / d["sin_t"].values
    se = np.where(np.isfinite(se), np.abs(se), np.nan)
    datasets.append(("MultiPhysio-HRC", d["sin_coef"].values, se))
    return datasets


def dersimonian_laird(coefs, ses):
    """随机效应 meta（DerSimonian-Laird）。返回合并效应、SE、p、I²。"""
    coefs = np.asarray(coefs, float)
    ses = np.asarray(ses, float)
    mask = np.isfinite(ses) & (ses > 0)
    coefs, ses = coefs[mask], ses[mask]
    w = 1.0 / ses ** 2
    mu_fixed = np.sum(w * coefs) / np.sum(w)
    Q = np.sum(w * (coefs - mu_fixed) ** 2)
    k = len(coefs)
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_rand = 1.0 / (ses ** 2 + tau2)
    mu_rand = np.sum(w_rand * coefs) / np.sum(w_rand)
    se_rand = np.sqrt(1.0 / np.sum(w_rand))
    z = mu_rand / se_rand
    p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    return dict(mu=mu_rand, se=se_rand, z=z, p=p, I2=I2, Q=Q, tau2=tau2, k=k)


def forest(ax, datasets, flip_mph=True, title=""):
    """森林图：每个样本的 sin 系数 ± SE + 合并效应。"""
    ys = []
    labels = []
    coefs_all, ses_all = [], []
    for i, (name, coefs, ses) in enumerate(datasets):
        # 每个样本画一个汇总点（样本均值 ± SE）
        c = coefs.copy()
        s = ses.copy()
        if flip_mph and name == "MultiPhysio-HRC":
            c = -c
        m = np.nanmean(c)
        se = np.nanmean(s)
        ys.append(len(datasets) - i)
        labels.append(name)
        ax.errorbar(m, len(datasets) - i, xerr=se, fmt="o", ms=6,
                    capsize=3, color="#1f77b4")
        coefs_all.extend(c[np.isfinite(s)])
        ses_all.extend(s[np.isfinite(s)])

    meta = dersimonian_laird(np.array(coefs_all), np.array(ses_all))
    # 合并效应（菱形）
    y_meta = 0
    ax.errorbar(meta["mu"], y_meta, xerr=meta["se"], fmt="D", ms=8,
                color="#d62728", capsize=4)
    labels.insert(0, "Random-effects summary")
    ys.insert(0, y_meta)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("sin(resp_phase) coefficient (z)")
    ax.set_title(title)
    return meta


def main():
    datasets = load_sin()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    meta_raw = forest(axes[0], datasets, flip_mph=False,
                      title="Raw (as recorded)")
    meta_flip = forest(axes[1], datasets, flip_mph=True,
                       title="MultiPhysio polarity-corrected")
    fig.suptitle("Respiratory-phase modulation of HEP: 3-sample meta-analysis",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_meta_forest.pdf")
    plt.close(fig)

    print("=== 随机效应 meta（未翻转极性）===")
    print(f"  mu={meta_raw['mu']:.4f} SE={meta_raw['se']:.4f} "
          f"z={meta_raw['z']:.2f} p={meta_raw['p']:.4g} I²={meta_raw['I2']:.0f}%")
    print("=== 随机效应 meta（MultiPhysio 极性翻转）===")
    print(f"  mu={meta_flip['mu']:.4f} SE={meta_flip['se']:.4f} "
          f"z={meta_flip['z']:.2f} p={meta_flip['p']:.4g} I²={meta_flip['I2']:.0f}%")
    print(f"\n图: {FIG / 'fig_dir1_meta_forest.pdf'}")


if __name__ == "__main__":
    main()
