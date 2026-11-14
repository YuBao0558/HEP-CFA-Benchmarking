#!/usr/bin/env Rscript
# Figure S1: R 峰复检 Bland-Altman 图
# 输入: data/rpeak_validation.csv (由 01_rpeak_validation.py 生成)
# 输出: figures/figS1_rpeak_blandaltman.pdf

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
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

d <- read.csv(file.path(DATA_DIR, "rpeak_validation.csv"), stringsAsFactors = FALSE)

# Bland-Altman: x = 平均数量, y = 数量差
d <- d %>%
  mutate(avg_n = (n_official + n_self) / 2,
         diff_n = n_official - n_self,
         diff_pct = n_diff_pct)

# 数量差散点
p1 <- ggplot(d, aes(x = avg_n, y = diff_n)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_hline(yintercept = c(-1, 1), linetype = "dotted", color = "grey60") +
  geom_point(alpha = 0.6, size = 1.5) +
  labs(x = "平均 R 峰数 (官方+自检)/2", y = "数量差 (官方−自检)",
       title = "S1a: R 峰数量 Bland-Altman") +
  theme_bhci()

# 系统偏移 (mean_diff_ms) 分布: 展示固定偏移 vs 随机误差
fixed_offset <- median(d$mean_diff_ms, na.rm = TRUE)
p2 <- ggplot(d, aes(x = mean_diff_ms)) +
  geom_histogram(bins = 30, fill = "grey70", color = "grey40") +
  geom_vline(xintercept = 0, color = "grey40", linetype = "dashed") +
  geom_vline(xintercept = fixed_offset, color = "blue", linetype = "dashed") +
  labs(x = "时间戳偏移 mean_diff (ms)", y = "计数",
       title = sprintf("S1b: 时间戳系统偏移 (蓝线 = 中位数 %.1f ms, 亚采样精度所致)",
                       fixed_offset)) +
  theme_bhci()

# 扣偏后 RMSE (残留随机误差) 分布: 这才是真正与 10ms 标准比较的量
d_usable <- d %>% filter(!is.na(rmse_centered_ms))
p3 <- ggplot(d_usable, aes(x = rmse_centered_ms)) +
  geom_histogram(bins = 20, fill = "grey70", color = "grey40") +
  geom_vline(xintercept = 10, color = "red", linetype = "dashed") +
  labs(x = "扣偏后 RMSE (ms)", y = "计数",
       title = "S1c: 扣偏后残留随机误差 (红线 = 10ms 冻结标准)") +
  theme_bhci()

# 数量差百分比分布
p4 <- ggplot(d, aes(x = n_diff_pct)) +
  geom_histogram(bins = 20, fill = "grey70", color = "grey40") +
  geom_vline(xintercept = 1, color = "red", linetype = "dashed") +
  labs(x = "R 峰数量差 (%)", y = "计数",
       title = "S1d: R 峰数量差百分比分布 (红线 = 1% 冻结标准)") +
  theme_bhci()

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "figS1_rpeak_blandaltman.pdf"),
       arrangeGrob(p1, p2, p3, p4, ncol = 1), width = 12, height = 26, units = "cm"
       )
cat("已保存 figures/figS1_rpeak_blandaltman.pdf\n")
