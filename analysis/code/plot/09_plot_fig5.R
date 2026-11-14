#!/usr/bin/env Rscript
# Figure 5: 跨数据集泛化森林图
# 输入: data/cross_dataset_effects.csv (Step9 输出)
# 输出: figures/fig5_cross_dataset_forest.pdf
#
# 注: 依赖 Step9 跨数据集分析。数据未就绪时不运行。

suppressMessages({
  library(ggplot2)
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
      plot.title = element_text(face = "bold", hjust = 0)
    )
}

if (!file.exists(file.path(DATA_DIR, "cross_dataset_effects.csv"))) {
  cat("cross_dataset_effects.csv 不存在, 跳过 Fig5 (待 Step9)\n")
  quit(save = "no", status = 0)
}

d <- read.csv(file.path(DATA_DIR, "cross_dataset_effects.csv"), stringsAsFactors = FALSE)
d$experiment <- factor(d$experiment, levels = c("2", "3", "4", "5"))

# 森林图: 各实验 LMM condition 主效应 (β ± 95% CI)
d$lmm_beta <- as.numeric(d$lmm_beta)
d$lmm_ci_low <- as.numeric(d$lmm_ci_low)
d$lmm_ci_high <- as.numeric(d$lmm_ci_high)

p <- ggplot(d, aes(x = lmm_beta, y = experiment)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lmm_ci_low, xmax = lmm_ci_high), height = 0.25) +
  geom_point(size = 3, color = "#0072B2") +
  scale_y_discrete(labels = c("2" = "exp2 (意外学习)",
                              "3" = "exp3 (有意学习)",
                              "4" = "exp4 (双注意)",
                              "5" = "exp5 (金钱激励)")) +
  labs(x = "LMM condition 主效应 β (95% CI, μV)", y = "",
       title = "Figure 5: 跨实验主效应森林图") +
  theme_bhci()

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "fig5_cross_dataset_forest.pdf"), p, width = 14, height = 8, units = "cm"
       )
cat("已保存 figures/fig5_cross_dataset_forest.pdf\n")
