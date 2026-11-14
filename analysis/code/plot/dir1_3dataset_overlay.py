#!/usr/bin/env python3
"""方向1 论文级图：三数据集呼吸相位调制 HEP（统一极性后叠加）。

产出：analysis/figures/fig_dir1_3dataset_overlay.pdf
  - 左：原始（未对齐极性），展示 exp4/exp5 同向、MultiPhysio 反向
  - 右：统一极性（MultiPhysio 翻转 180°）后三条曲线重合

这是方向1 最核心的"跨数据集复现"图，证明呼吸相位调制 HEP 是同一生理现象，
方向差异来自呼吸信号极性定义。
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
FIG.mkdir(parents=True, exist_ok=True)

N_BINS = 8
COLORS = {"experiment4": "#d62728", "experiment5": "#1f77b4", "MultiPhysio": "#2ca02c"}
LABELS = {"experiment4": "BBBD exp4", "experiment5": "BBBD exp5",
          "MultiPhysio": "MultiPhysio-HRC"}


def load_bbbd():
    dfs = []
    for exp in ["experiment4", "experiment5"]:
        d = pd.read_csv(BBBD / "resp_phase_binned_hep.csv")
        d = d[d["experiment"] == exp]
        d["dataset"] = exp
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_mph():
    d = pd.read_csv(MPH / "resp_phase_binned_hep.csv")
    d["dataset"] = "MultiPhysio"
    return d


def plot_curve(ax, g, flip=False):
    g = g.sort_values("bin")
    vals = g["mean"].values
    if flip:
        vals = np.roll(vals, 4)  # 180° = 4 bin
    ph = g["phase_deg"].values
    sem = g["std"].values / np.sqrt(g["count"].values)
    ax.errorbar(ph, vals, yerr=sem, marker="o", ms=4,
                color=COLORS[g["dataset"].iloc[0]], lw=1.5,
                label=LABELS[g["dataset"].iloc[0]])


def main():
    bb = load_bbbd()
    mp = load_mph()
    all_df = pd.concat([bb, mp], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左：原始（未对齐）
    for key, g in all_df.groupby("dataset"):
        plot_curve(axes[0], g, flip=False)
    axes[0].axhline(0, color="grey", lw=0.8, ls="--")
    axes[0].set_xlabel("Respiratory phase (deg)")
    axes[0].set_ylabel("HEP amplitude (z)")
    axes[0].set_title("Raw (phase-zero as recorded)")
    axes[0].set_xticks([0, 90, 180, 270, 360])
    axes[0].legend(frameon=False, fontsize=8)

    # 右：统一极性（MultiPhysio 翻转 180°）
    for key, g in all_df.groupby("dataset"):
        plot_curve(axes[1], g, flip=(key == "MultiPhysio"))
    axes[1].axhline(0, color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("Respiratory phase (deg)")
    axes[1].set_ylabel("HEP amplitude (z)")
    axes[1].set_title("MultiPhysio phase-inverted (180°)")
    axes[1].set_xticks([0, 90, 180, 270, 360])
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Respiratory-phase modulation of HEP is consistent across 3 datasets",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dir1_3dataset_overlay.pdf")
    plt.close(fig)
    print("图:", FIG / "fig_dir1_3dataset_overlay.pdf")


if __name__ == "__main__":
    main()
