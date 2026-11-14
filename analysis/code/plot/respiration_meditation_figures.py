#!/usr/bin/env python3
"""呼吸-HEP耦合 + 冥想专长 两方向结果图。

产出 (写到 analysis/figures/):
  fig_resp_phase_modulation.pdf   呼吸相位分箱的 HEP 曲线 (exp4/5 叠加)
  fig_resp_meta_coef.pdf          跨被试 sin/cos/rr_prev 系数 (meta)
  fig_meditation_expertise.pdf    ds003816 lt vs st HEP 幅度 + CFA 能量比
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BBBD_ANALYSIS = Path("/Users/yubao/CODE/public/BBBD/analysis")
DS_ANALYSIS = Path("/Users/yubao/CODE/public/ds003816-full/analysis")
FIG = BBBD_ANALYSIS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RESP = BBBD_ANALYSIS / "data" / "respiration_coupling"
MEDIT = DS_ANALYSIS / "results" / "meditation_expertise"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def fig_resp_phase_modulation():
    df = pd.read_csv(RESP / "resp_phase_binned_hep.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"experiment4": "#d62728", "experiment5": "#1f77b4"}
    labels = {"experiment4": "Exp 4 (dual-attention)", "experiment5": "Exp 5 (incentive)"}
    for exp, g in df.groupby("experiment"):
        g = g.sort_values("bin")
        ax.errorbar(g["phase_deg"], g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                    marker="o", ms=4, label=labels[exp], color=colors[exp], lw=1.5)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Respiratory phase (deg)")
    ax.set_ylabel("HEP amplitude (z-scored)")
    ax.set_title("HEP amplitude is modulated by respiratory phase")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_resp_phase_modulation.pdf")
    plt.close(fig)


def fig_resp_meta_coef():
    df = pd.read_csv(RESP / "resp_modulation_meta.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(3)
    width = 0.35
    labels = ["sin(resp phase)", "cos(resp phase)", "rr_prev (cardiac)"]
    for i, exp in enumerate(["experiment4", "experiment5"]):
        r = df[df.experiment == exp].iloc[0]
        means = [r["sin_coef_mean"], r["cos_coef_mean"], r["rr_coef_mean"]]
        sds = [r["sin_coef_sd"], r["cos_coef_sd"], r["rr_coef_sd"]]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sds,
               label=exp.replace("experiment", "Exp "), capsize=2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Subject-level coefficient (z)")
    ax.set_title("Respiratory vs cardiac modulation of HEP")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_resp_meta_coef.pdf")
    plt.close(fig)


def fig_meditation_expertise():
    hgc = pd.read_csv(MEDIT / "hep_group_compare.csv")
    cgc = pd.read_csv(MEDIT / "cfa_group_compare.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # HEP amplitude clean track
    clean = hgc[hgc.track == "clean"]
    x = np.arange(len(clean))
    w = 0.35
    axes[0].bar(x - w/2, clean["lt_mean"], w, yerr=clean["lt_sd"], label="lt (long-term)", capsize=2, color="#2ca02c")
    axes[0].bar(x + w/2, clean["st_mean"], w, yerr=clean["st_sd"], label="st (novice)", capsize=2, color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clean["roi"], rotation=15)
    axes[0].set_ylabel("HEP RMS (µV)")
    axes[0].set_title("HEP amplitude: lt vs st (clean)")
    axes[0].legend(frameon=False)

    # CFA max ratio
    r = cgc.iloc[0]
    axes[1].bar([0, 1], [r["lt_mean"], r["st_mean"]], 0.5,
                yerr=[r["lt_sd"], r["st_sd"]], capsize=4,
                color=["#2ca02c", "#ff7f0e"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["lt (long-term)", "st (novice)"])
    axes[1].set_ylabel("Max CFA energy ratio")
    axes[1].set_title("CFA energy ratio: lt vs st")

    fig.tight_layout()
    fig.savefig(FIG / "fig_meditation_expertise.pdf")
    plt.close(fig)


def main():
    fig_resp_phase_modulation()
    fig_resp_meta_coef()
    fig_meditation_expertise()
    print(f"图产出目录: {FIG}")
    for f in ["fig_resp_phase_modulation.pdf", "fig_resp_meta_coef.pdf",
              "fig_meditation_expertise.pdf"]:
        print("  ", f)


if __name__ == "__main__":
    main()
