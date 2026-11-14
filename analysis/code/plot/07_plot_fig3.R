#!/usr/bin/env Rscript
# Figure 3: 状态调制主效应 + LMM 推断
# 输入: data/hep_dual.csv (Step3双轨, 含 roi 列) + data/lmm_coefs.csv (Step7)
# 输出: figures/fig3A_paired.pdf + figures/fig3B_forest.pdf

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

# 会话语义 (与 common.py / 07_lmm.R 对齐): base=基准会话, contrast=对照会话
SESS_LABELS <- switch(BBBD_EXPERIMENT,
                      "2" = c(base = "attentive", contrast = "distracted"),
                      "3" = c(base = "attentive", contrast = "distracted"),
                      "4" = c(base = "attentive1", contrast = "attentive2"),
                      "5" = c(base = "attentive", contrast = "incentive"),
                      c(base = "attentive", contrast = "distracted"))
BASE_LBL <- unname(SESS_LABELS["base"])
CONT_LBL <- unname(SESS_LABELS["contrast"])
COND_COLORS <- setNames(c("#0072B2", "#D55E00"), c(BASE_LBL, CONT_LBL))


library(showtext)
showtext_auto()


theme_bhci <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      axis.line = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0),
      legend.position = "right"
    )
}

if (!file.exists(file.path(DATA_DIR, "hep_dual.csv"))) {
  cat("hep_dual.csv 不存在, 跳过 Fig3\n")
  quit(save = "no", status = 0)
}

d <- read.csv(file.path(DATA_DIR, "hep_dual.csv"), stringsAsFactors = FALSE)

# ============ 3A: 配对散点-箱型 (frontocentral ROI, clean 轨道) ============
# 先聚合到被试×条件 (每个被试每条件 1 个值), 再画配对箱线。
# 避免把 stim 级重复当作独立观测 (样本量虚高)。
d_clean <- d %>% filter(track == "clean", roi == "frontocentral")

d_sub <- d_clean %>%
  group_by(sub, condition) %>%
  summarise(main_rms = mean(main_200_600_rms_uv, na.rm = TRUE), .groups = "drop")
d_sub$condition <- factor(d_sub$condition, levels = c(BASE_LBL, CONT_LBL))

# 配对宽表 (供线段)
d_pair <- d_sub %>%
  pivot_wider(names_from = condition, values_from = main_rms)

p3a <- ggplot(d_sub, aes(x = condition, y = main_rms)) +
  geom_boxplot(aes(fill = condition), alpha = 0.4, outlier.shape = NA) +
  geom_point(aes(color = condition), position = position_jitter(width = 0.15),
             size = 1.5, alpha = 0.7) +
  geom_segment(data = d_pair,
               aes(x = 1, y = .data[[BASE_LBL]], xend = 2, yend = .data[[CONT_LBL]]),
               inherit.aes = FALSE, color = "grey50", alpha = 0.4) +
  labs(x = "条件", y = "额中央 HEP 主窗口 RMS (μV)",
       title = sprintf("Figure 3A: 条件对比 (%s vs %s)", BASE_LBL, CONT_LBL)) +
  scale_fill_manual(values = COND_COLORS) +
  scale_color_manual(values = COND_COLORS) +
  theme_bhci() + theme(legend.position = "none")

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "fig3A_paired.pdf"), p3a, width = 10, height = 9, units = "cm"
       )
cat("已保存 figures/fig3A_paired.pdf\n")

# ============ 3B: LMM 森林图 (读 lmm_coefs.csv) ============
if (file.exists(file.path(DATA_DIR, "lmm_coefs.csv"))) {
  coefs <- read.csv(file.path(DATA_DIR, "lmm_coefs.csv"), stringsAsFactors = FALSE)
  coefs$term <- factor(coefs$term, levels = rev(coefs$term))
  if (!"significant" %in% names(coefs)) coefs$significant <- FALSE

  p3b <- ggplot(coefs, aes(x = Estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.25) +
    geom_point(aes(color = significant), size = 2.5) +
    scale_color_manual(values = c("TRUE" = "#0072B2", "FALSE" = "grey60")) +
    labs(x = "标准化 β (95% CI)", y = "",
         title = "Figure 3B: LMM 固定效应森林图") +
    theme_bhci() + theme(legend.position = "none")

  ggsave(file.path(FIG_DIR, "fig3B_forest.pdf"), p3b, width = 12, height = 7, units = "cm"
         )
  cat("已保存 figures/fig3B_forest.pdf\n")
} else {
  cat("lmm_coefs.csv 不存在, 跳过 Fig3B (待 Step7)\n")
}
