#!/usr/bin/env Rscript
# Figure 2A: 心搏锁定 EEG 相位一致性动态 (ITPC)
# 输入: data/coupling_timefreq.csv (Step5 输出)
# 输出: figures/fig2A_timefreq.pdf
#
# 数据结构: sub × ses × stim × band(theta/alpha/beta) × time_s × itpc
# 分频带展示 ITPC 随时间动态, 组平均 (time_s 为 R 峰后延迟)

suppressMessages({
  library(ggplot2)
  library(dplyr)
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

# 会话语义 (与 common.py 对齐)
SESS_LABELS <- switch(BBBD_EXPERIMENT,
                      "4" = c("attentive1", "attentive2"),
                      "5" = c("attentive", "incentive"),
                      c("attentive", "distracted"))
COND_COLORS <- setNames(c("#0072B2", "#D55E00"), SESS_LABELS)


library(showtext)
showtext_auto()


theme_bhci <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      axis.line = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0),
      strip.background = element_blank()
    )
}

if (!file.exists(file.path(DATA_DIR, "coupling_timefreq.csv"))) {
  cat("coupling_timefreq.csv 不存在, 跳过 Fig2A (待 Step5)\n")
  quit(save = "no", status = 0)
}

d <- read.csv(file.path(DATA_DIR, "coupling_timefreq.csv"), stringsAsFactors = FALSE)

# 频带顺序
d$band <- factor(d$band,
                 levels = c("delta", "theta", "alpha", "beta"),
                 labels = c("delta (0.5-4Hz)", "theta (4-8Hz)",
                            "alpha (8-13Hz)", "beta (13-30Hz)"))

# 两阶段平均: 先按被试平均, 再跨被试平均 (避免被试间试次数不均)
d_mean <- d %>%
  group_by(sub, condition, band, time_s) %>%
  summarise(itpc = mean(itpc, na.rm = TRUE), .groups = "drop") %>%
  group_by(condition, band, time_s) %>%
  summarise(itpc = mean(itpc, na.rm = TRUE), .groups = "drop")

# 图: R峰后延迟 × ITPC 折线, 分频带 + 条件
p <- ggplot(d_mean, aes(x = time_s, y = itpc, color = condition)) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~ band, ncol = 1, scales = "free_y") +
  scale_color_manual(values = COND_COLORS) +
  labs(x = "R 峰后延迟 (s)", y = "ITPC (心搏锁定相位一致性)", color = "条件",
       title = "Figure 2A: 心搏锁定 EEG 相位一致性 (跨心跳 ITPC)") +
  theme_bhci()

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "fig2A_timefreq.pdf"), p, width = 16, height = 14, units = "cm"
       )
cat("已保存 figures/fig2A_timefreq.pdf\n")
