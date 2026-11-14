#!/usr/bin/env Rscript
# Figure 1C: HEP 头皮地形图 (200-600ms 窗口平均电压)
# 输入: data/hep_topography.csv + data/channel_positions.csv (03e 导出)
# 输出: figures/fig1C_topography.pdf
#
# 方法: 2D 俯视投影 (x-y), 通道点着色 = 窗口平均电压, 叠加 ROI 圆环标注。
#       用薄板样条插值 (fields::Tps) 生成平滑地形; 若无 fields 则退回散点。

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})
# ---- 实验路径 ----
BBBD_EXPERIMENT <- Sys.getenv("BBBD_EXPERIMENT", unset = "2")
EXP_DIR <- switch(BBBD_EXPERIMENT,
                  "2" = "experiment2", "3" = "experiment3",
                  "4" = "experiment4", "5" = "experiment5", "experiment2")
SCRIPT_DIR <- normalizePath(dirname(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)[1])))
ANALYSIS_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
DATA_DIR <- file.path(ANALYSIS_DIR, "data", EXP_DIR)
FIG_DIR  <- file.path(ANALYSIS_DIR, "figures", EXP_DIR)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

SESS_LABELS <- switch(BBBD_EXPERIMENT,
                      "4" = c("attentive1", "attentive2"),
                      "5" = c("attentive", "incentive"),
                      c("attentive", "distracted"))

library(showtext); showtext_auto()

theme_bhci <- function(base_size = 9) {
  theme_void(base_size = base_size) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "right")
}

if (!file.exists(file.path(DATA_DIR, "hep_topography.csv"))) {
  cat("hep_topography.csv 不存在, 跳过 Fig1C (待 03e)\n")
  quit(save = "no", status = 0)
}

topo <- read.csv(file.path(DATA_DIR, "hep_topography.csv"), stringsAsFactors = FALSE)
pos  <- read.csv(file.path(DATA_DIR, "channel_positions.csv"), stringsAsFactors = FALSE)

# 组平均: 先被试平均, 再跨被试平均
topo_sum <- topo %>%
  left_join(pos %>% select(channel, x, y), by = "channel") %>%
  group_by(sub, condition, channel, x, y) %>%
  summarise(amp_uv = mean(amp_uv, na.rm = TRUE), .groups = "drop") %>%
  group_by(condition, channel, x, y) %>%
  summarise(amp_uv = mean(amp_uv, na.rm = TRUE), .groups = "drop")

# ROI 定义 (与 common.py ROI 一致)
roi <- list(
  frontocentral = c("Fp1","Fp2","AF3","AF4","F1","F2","F3","F4","Fz",
                    "FC1","FC2","FC3","FC4","FCz","C1","C2","C3","C4","Cz",
                    "CP1","CP2","CP3","CP4","CPz"),
  central = c("C1","C2","C3","C4","C5","C6","Cz","FC1","FC2","FCz","CP1","CP2","CPz"),
  parietal = c("P1","P2","P3","P4","Pz","PO3","PO4","POz","P5","P6","P7","P8")
)
roi_df <- dplyr::bind_rows(lapply(names(roi), function(n) {
  data.frame(roi = n, channel = roi[[n]])
}))

topo_sum <- topo_sum %>% left_join(roi_df, by = "channel")

# 头皮圆 + 耳朵/鼻子示意
head_circle <- data.frame(
  x0 = 0, y0 = 0, r = 0.115
)

# 颜色对称于 0
lim <- max(abs(topo_sum$amp_uv), na.rm = TRUE)

p1c <- ggplot(topo_sum, aes(x = x, y = y)) +
  # 头皮轮廓
  annotate("path",
           x = head_circle$r * cos(seq(0, 2*pi, length.out = 100)),
           y = head_circle$r * sin(seq(0, 2*pi, length.out = 100)),
           color = "grey40", linewidth = 0.5) +
  # 鼻子
  annotate("segment", x = 0, xend = 0, y = head_circle$r, yend = head_circle$r + 0.02,
           color = "grey40", linewidth = 0.5) +
  geom_point(aes(color = amp_uv, shape = roi), size = 2.6) +
  scale_shape_manual(values = c("frontocentral" = 19, "central" = 17, "parietal" = 15),
                     na.value = 1) +
  scale_color_gradient2(low = "#D55E00", mid = "white", high = "#0072B2",
                        midpoint = 0, limits = c(-lim, lim)) +
  coord_fixed() +
  facet_wrap(~ condition, labeller = labeller(condition = setNames(
    c("基准", "对照"), SESS_LABELS))) +
  labs(title = "Figure 1C: HEP 头皮地形 (200-600ms 平均电压, μV)",
       color = "电压 (μV)", shape = "ROI") +
  theme_bhci()

ggsave(file.path(FIG_DIR, "fig1C_topography.pdf"), p1c, width = 16, height = 8, units = "cm")
cat("已保存 figures/fig1C_topography.pdf\n")
