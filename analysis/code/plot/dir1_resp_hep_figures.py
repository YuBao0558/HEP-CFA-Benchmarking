#!/usr/bin/env python3
"""方向1 深化：呼吸-心搏-脑耦合的结果图。

产出（写到 analysis/figures/）：
  fig_dir1_phase_modulation.pdf       呼吸相位分箱 HEP 曲线（吸气相下降）
  fig_dir1_relative_contribution.pdf  呼吸 vs 心搏前间期 增量 R² 对比
  fig_dir1_itpc.pdf                   呼吸相位 ITPC 分布（呼吸-心跳不锁相）
  fig_dir1_amplitude_ratio.pdf        呼吸调制幅度 vs HEP-RMS 占比
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESP = Path("/Users/yubao/CODE/public/BBBD/analysis/data/respiration_coupling")
FIG = Path("/Users/yubao/CODE/public/BBBD/analysis/figures")
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

COLORS = {"experiment4": "#d62728", "experiment5": "#1f77b4"}
LABELS = {"experiment4": "Exp 4 (n=43)", "experiment5": "Exp 5 (n=46)"}


def fig_phase_modulation():
    df = pd.read_csv(RESP / "resp_phase_binned_hep.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for exp, g in df.groupby("experiment"):
        g = g.sort_values("bin")
        ax.errorbar(g["phase_deg"], g["mean"],
                    yerr=g["std"] / np.sqrt(g["count"]),
                    marker="o", ms=4, label=LABELS[exp], color=COLORS[exp], lw=1.5)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    # 标注吸气/呼气相
    ax.axvspan(0, 180, color="grey", alpha=0.06)
    ax.text(60, ax.get_ylim()[1] * 0.9, "inspiration", ha="center", fontsize=8, color="grey")
    ax.text(270, ax.get_ylim()[1] * 0.9, "expiration", ha="center", fontsize=8, color="grey")
    ax.set_xlabel("Respiratory phase (deg, 0 = expiratory trough)")
    ax.set_ylabel("HEP amplitude (z-scored)")
    ax.set_title("HEP amplitude modulated by respiratory phase")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_phase_modulation.pdf")
    plt.close(fig)


def fig_relative_contribution():
    df = pd.read_csv(RESP / "resp_relative_contribution.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(2)
    width = 0.35
    for i, exp in enumerate(["experiment4", "experiment5"]):
        g = df[df.experiment == exp]
        resp_med = g["r2_resp_only"].median()
        rr_med = g["r2_rr_only"].median()
        # 用箱线图数据
        vals = [g["r2_resp_only"].values, g["r2_rr_only"].values]
        bp = ax.boxplot(vals, positions=x + (i - 0.5) * width, widths=width * 0.9,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[exp])
            patch.set_alpha(0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["respiration\n(sin+cos)", "cardiac\n(rr_prev)"])
    ax.set_ylabel("Incremental R² (single-trial HEP)")
    ax.set_title("Respiration contributes more than cardiac interval")
    # 图例
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=COLORS["experiment4"], alpha=0.6),
               Patch(facecolor=COLORS["experiment5"], alpha=0.6)],
              [LABELS["experiment4"], LABELS["experiment5"]], frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_relative_contribution.pdf")
    plt.close(fig)


def fig_itpc():
    df = pd.read_csv(RESP / "resp_itpc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, exp in zip(axes, ["experiment4", "experiment5"]):
        g = df[df.experiment == exp]
        ax.hist(g["itpc"], bins=15, color=COLORS[exp], alpha=0.7)
        ax.axvline(g["itpc"].median(), color="black", ls="--", lw=1,
                   label=f"median={g['itpc'].median():.3f}")
        ax.set_xlabel("Respiratory-phase ITPC")
        ax.set_ylabel("Subjects")
        ax.set_title(LABELS[exp])
        ax.legend(frameon=False)
    fig.suptitle("Respiratory phase is not locked to heartbeats (low ITPC)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_itpc.pdf")
    plt.close(fig)


def fig_amplitude_ratio():
    df = pd.read_csv(RESP / "resp_amplitude_contribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, exp in zip(axes, ["experiment4", "experiment5"]):
        g = df[df.experiment == exp]
        valid = g[g["valid_pct"]]
        ax.hist(valid["resp_pp_pct_of_hep_rms"], bins=15, color=COLORS[exp], alpha=0.7)
        ax.axvline(valid["resp_pp_pct_of_hep_rms"].median(), color="black", ls="--",
                   lw=1, label=f"median={valid['resp_pp_pct_of_hep_rms'].median():.1f}%")
        ax.set_xlabel("Respiration amplitude / HEP RMS (%)")
        ax.set_ylabel("Subjects")
        ax.set_title(LABELS[exp])
        ax.legend(frameon=False)
    fig.suptitle("Respiratory modulation as fraction of HEP fluctuation")
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_amplitude_ratio.pdf")
    plt.close(fig)


def main():
    fig_phase_modulation()
    fig_relative_contribution()
    fig_itpc()
    fig_amplitude_ratio()
    print("图产出目录:", FIG)
    for f in ["fig_dir1_phase_modulation.pdf", "fig_dir1_relative_contribution.pdf",
              "fig_dir1_itpc.pdf", "fig_dir1_amplitude_ratio.pdf"]:
        print("  ", f)


if __name__ == "__main__":
    main()
