#!/usr/bin/env Rscript
# Figure 1: HEP 双轨波形 + 窗口着色
# 输入: data/hep_waveform.csv (由 03b_hep_dual.py 生成)
# 输出: figures/fig1_hep_dual.pdf

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


# 统一主题
theme_bhci <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(family = "sans"),
      axis.line = element_line(linewidth = 0.3),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.key.size = unit(0.6, "cm"),
      plot.title = element_text(face = "bold", hjust = 0),
      strip.background = element_blank()
    )
}

# 读取数据
wave <- read.csv(file.path(DATA_DIR, "hep_waveform.csv"), stringsAsFactors = FALSE)

# 两阶段平均: 先按 (sub, 条件, 通道, track, 时间) 平均各被试, 再跨被试平均。
# 避免被试间试次数不均导致的加权偏差。
wave_sum <- wave %>%
  group_by(sub, condition, channel, track, time_ms) %>%
  summarise(amp_uv = mean(amp_uv, na.rm = TRUE), .groups = "drop") %>%
  group_by(condition, channel, track, time_ms) %>%
  summarise(amp_uv = mean(amp_uv, na.rm = TRUE), .groups = "drop")

# 窗口着色背景
windows <- data.frame(
  xmin = c(-100, 0, 200),
  xmax = c(0, 200, 600),
  label = c("baseline", "early CFA", "main 200-600ms")
)

# Fig 1A: HEP 双轨波形（按 track 分色）
p1a <- ggplot(wave_sum, aes(x = time_ms, y = amp_uv, color = track)) +
  geom_rect(aes(xmin = 0, xmax = 200), ymin = -Inf, ymax = Inf,
            fill = "red", alpha = 0.06, inherit.aes = FALSE) +
  geom_rect(aes(xmin = 200, xmax = 600), ymin = -Inf, ymax = Inf,
            fill = "green", alpha = 0.10, inherit.aes = FALSE) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.4) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
  geom_line(linewidth = 0.7) +
  facet_grid(channel ~ condition) +
  scale_color_manual(values = c("clean" = "#0072B2", "eyeonly" = "#D55E00"),
                     labels = c("clean" = "去 CFA", "eyeonly" = "保留 CFA")) +
  labs(x = "时间 (ms)", y = "振幅 (μV)", color = "CFA 处理",
       title = "Figure 1: HEP 双轨波形（去/不去 CFA）") +
  theme_bhci()

# 保存 (cairo_pdf 以正确渲染中文标签)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "fig1_hep_dual.pdf"), p1a, width = 18.3, height = 12,
       units = "cm")

cat("已保存 figures/fig1_hep_dual.pdf\n")
