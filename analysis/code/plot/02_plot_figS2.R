#!/usr/bin/env Rscript
# Figure S2 + S3: 预处理报告 (ICA 成分 + 剔除 epoch 数)
# 输入: data/preprocess_report.csv (Step2 输出)
# 输出: figures/figS2_preprocess.pdf
#
# 注: 依赖 Step2 全量完成后生成的 preprocess_report.csv。

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
      plot.title = element_text(face = "bold", hjust = 0)
    )
}

if (!file.exists(file.path(DATA_DIR, "preprocess_report.csv"))) {
  cat("preprocess_report.csv 不存在或未完成, 跳过 S2/S3\n")
  quit(save = "no", status = 0)
}

d <- read.csv(file.path(DATA_DIR, "preprocess_report.csv"), stringsAsFactors = FALSE)

# 剔除处理失败 (error 非空) 的记录
if ("error" %in% names(d)) {
  d <- d %>% filter(is.na(error) | error == "")
}

# S2: 每个被试的眼动/CFA/坏导成分数分布
d_long <- d %>%
  select(sub, condition, n_bad_channels, n_eye_components, n_cfa_components) %>%
  pivot_longer(-c(sub, condition), names_to = "type", values_to = "n") %>%
  mutate(type = recode(type,
    n_bad_channels = "坏导", n_eye_components = "眼动成分", n_cfa_components = "CFA成分"))

p_s2 <- ggplot(d_long, aes(x = condition, y = n, fill = condition)) +
  geom_boxplot(alpha = 0.4, outlier.size = 0.8) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.5) +
  facet_wrap(~type, scales = "free_y") +
  labs(x = "条件", y = "数量", title = "Figure S2: 预处理伪迹成分统计") +
  scale_fill_manual(values = COND_COLORS) +
  theme_bhci() + theme(legend.position = "none")

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(FIG_DIR, "figS2_preprocess.pdf"), p_s2, width = 16, height = 8, units = "cm"
       )
cat("已保存 figures/figS2_preprocess.pdf\n")
