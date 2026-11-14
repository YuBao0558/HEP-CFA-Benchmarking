#!/usr/bin/env Rscript
# Figure 2B + S4: Surrogate 零分布直方图
# 输入: data/surrogate_null_dist.csv + data/surrogate_l1.csv
# 输出: figures/figS4_surrogate_null.pdf

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

null <- read.csv(file.path(DATA_DIR, "surrogate_null_dist.csv"), stringsAsFactors = FALSE)
obs  <- read.csv(file.path(DATA_DIR, "surrogate_l1.csv"), stringsAsFactors = FALSE)

# 每个记录的观测值
obs_line <- obs %>% select(sub, ses, stim, obs_rms_uv)

# 合并，画三类 surrogate 的 null 分布（当前只有 time_shift + phase_rand）
# 按记录分面（示例：只看 sub-01）。将 obs 合并进 null 表, 供 facet 内 vline 使用。
null_sub <- null %>%
  filter(sub == "sub-01") %>%
  left_join(obs_line %>% filter(sub == "sub-01") %>%
              select(ses, stim, obs_rms_uv),
            by = c("ses", "stim"))

p <- ggplot(null_sub, aes(x = rms_uv)) +
  geom_histogram(bins = 40, fill = "grey80", color = "grey40", alpha = 0.7) +
  geom_vline(aes(xintercept = obs_rms_uv), color = "red", linewidth = 0.8) +
  facet_grid(ses + stim ~ type) +
  labs(x = "null 分布 RMS (μV)", y = "计数",
       title = "Figure S4: Surrogate 零分布 vs 观测 HEP (红线)") +
  theme_bhci()

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "figS4_surrogate_null.pdf"), p, width = 18, height = 14, units = "cm"
       )
cat("已保存 figures/figS4_surrogate_null.pdf\n")
