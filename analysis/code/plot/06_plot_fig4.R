#!/usr/bin/env Rscript
# Figure 4: 多宇宙分析规范曲线 (Specification Curve)
# 输入: data/multiverse_effects.csv (Step6 输出)
# 输出: figures/fig4_specification_curve.pdf
#
# 数据结构 (来自 06_multiverse.py):
#   multiverse_effects.csv: pipeline_id, track, roi, window, hedges_dz, n_subjects
#   multiverse_params.csv:  pipeline_id, track, roi, window

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
# ---- 实验路径 (与 common.py / 07_lmm.R 对齐) ----
BBBD_EXPERIMENT <- Sys.getenv("BBBD_EXPERIMENT", unset = "2")
EXP_DIR <- switch(BBBD_EXPERIMENT,
                  "2" = "experiment2",
                  "3" = "experiment3",
                  "4" = "experiment4",
                  "5" = "experiment5",
                  "experiment2")
SCRIPT_DIR <- normalizePath(dirname(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)[1])))
ANALYSIS_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
DATA_DIR <- file.path(ANALYSIS_DIR, "data", EXP_DIR)
FIG_DIR  <- file.path(ANALYSIS_DIR, "figures", EXP_DIR)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)


library(showtext)
showtext_auto()


theme_bhci <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      axis.line = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0),
      legend.position = "none"
    )
}

if (!file.exists(file.path(DATA_DIR, "multiverse_effects.csv"))) {
  cat("multiverse_effects.csv 不存在, 跳过 Fig4 (待 Step6)\n")
  quit(save = "no", status = 0)
}

eff <- read.csv(file.path(DATA_DIR, "multiverse_effects.csv"), stringsAsFactors = FALSE)
param <- read.csv(file.path(DATA_DIR, "multiverse_params.csv"), stringsAsFactors = FALSE)

# 按效应量排序, 生成 rank
eff <- eff %>%
  arrange(desc(hedges_dz)) %>%
  mutate(rank = row_number())

# 显著判定: 用 p_val (若有) 否则用 |hedges_dz| > 0.3 (SESOI 阈值)
if ("p_val" %in% names(eff)) {
  eff$significant <- ifelse(!is.na(eff$p_val) & eff$p_val < 0.05, "yes", "no")
} else {
  eff$significant <- ifelse(!is.na(eff$hedges_dz) & abs(eff$hedges_dz) > 0.3, "yes", "no")
}

# 4A: specification curve
p4a <- ggplot(eff, aes(x = rank, y = hedges_dz)) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.3, ymax = 0.3),
            fill = "grey", alpha = 0.25, inherit.aes = FALSE) +
  geom_hline(yintercept = 0.3, linetype = "dashed", color = "red") +
  geom_hline(yintercept = -0.3, linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
  geom_point(aes(color = significant), size = 1.2) +
  scale_color_manual(values = c("yes" = "#0072B2", "no" = "grey75")) +
  labs(x = "分析流程 (按效应量排序)", y = "ΔBHCI (Hedges' dz)",
       title = "Figure 4A: Specification Curve (红虚线 = SESOI ±0.3, 灰区 = ROPE)") +
  theme_bhci()

# 4B: 参数指纹矩阵 (与 4A 对齐, 按 pipeline_id 映射到 rank)
# 多宇宙维度: track/filt/min_rr/baseline/roi/window
param_cols <- intersect(c("track", "filt", "min_rr", "baseline", "roi", "window"),
                        names(param))
param <- param %>%
  left_join(eff %>% select(pipeline_id, rank), by = "pipeline_id")

param_long <- param %>%
  pivot_longer(all_of(param_cols), names_to = "param", values_to = "value") %>%
  mutate(rank = as.numeric(rank),
         value = as.character(value))

p4b <- ggplot(param_long, aes(x = rank, y = param, fill = value)) +
  geom_tile() +
  scale_fill_manual(values = c(
    "clean" = "#0072B2", "eyeonly" = "#D55E00",
    "zerophase" = "#2ca02c", "causal" = "#9467bd",
    "frontocentral" = "#1f77b4", "central" = "#2ca02c", "parietal" = "#ff7f0e",
    "main_200_600" = "#7f7f7f", "early_0_200" = "#bcbd22",
    "corrected" = "#e377c2", "raw" = "#17becf", "wholecycle" = "#8c564b"
  )) +
  labs(x = "", y = "参数维度", title = "Figure 4B: 参数指纹矩阵") +
  theme_bhci() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

p_comb <- p4a / p4b + plot_layout(heights = c(3, 2))

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "fig4_specification_curve.pdf"), p_comb,
       width = 18, height = 14, units = "cm")
cat("已保存 figures/fig4_specification_curve.pdf\n")
