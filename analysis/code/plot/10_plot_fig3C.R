#!/usr/bin/env Rscript
# Figure 3C: 呼吸/RR 代理控制图 (偏回归)
# 输入: data/hep_singletrial.csv (03d 导出)
# 输出: figures/fig3C_resp_control.pdf
#
# 目的: 展示控制 RR 间期 (心跳重叠/RSA 代理) 后 condition 效应是否仍保留。
# 方法: 单试次 hep_amp ~ condition + rr_prev_s + (1|sub), 对比
#       未控制 (仅 condition) vs 控制 (condition + rr_prev_s) 的 condition 系数。

suppressMessages({
  library(ggplot2)
  library(dplyr)
  library(lme4)
  library(lmerTest)
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
                      "4" = c(base = "attentive1", contrast = "attentive2"),
                      "5" = c(base = "attentive", contrast = "incentive"),
                      c(base = "attentive", contrast = "distracted"))
BASE_LBL <- unname(SESS_LABELS["base"])
CONT_LBL <- unname(SESS_LABELS["contrast"])

library(showtext); showtext_auto()

theme_bhci <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(axis.line = element_line(linewidth = 0.3),
          panel.grid = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0))
}

if (!file.exists(file.path(DATA_DIR, "hep_singletrial.csv"))) {
  cat("hep_singletrial.csv 不存在, 跳过 Fig3C (待 03d)\n")
  quit(save = "no", status = 0)
}

d <- read.csv(file.path(DATA_DIR, "hep_singletrial.csv"), stringsAsFactors = FALSE)
d$condition <- factor(d$condition, levels = c(BASE_LBL, CONT_LBL))
d$rr_prev_s <- suppressWarnings(as.numeric(d$rr_prev_s))
d$resp_phase <- suppressWarnings(as.numeric(d$resp_phase))
d <- d %>% filter(!is.na(rr_prev_s) & rr_prev_s > 0)

# 两个模型: 未控制 vs 控制 RR
m0 <- tryCatch(lmer(hep_amp_uv ~ condition + (1 | sub), data = d), error = function(e) NULL)
m1 <- tryCatch(lmer(hep_amp_uv ~ condition + rr_prev_s + (1 | sub), data = d), error = function(e) NULL)

if (is.null(m0) || is.null(m1)) {
  cat("模型失败, 跳过 Fig3C\n")
  quit(save = "no", status = 0)
}

# 提取 condition 系数
get_cond <- function(m, label) {
  co <- as.data.frame(summary(m)$coefficients)
  tr <- grep("condition", rownames(co), value = TRUE)
  if (length(tr) == 0) return(NULL)
  data.frame(model = label, term = tr[1],
             beta = co[tr[1], "Estimate"],
             se = co[tr[1], "Std. Error"],
             p = co[tr[1], "Pr(>|t|)"])
}
cmp <- rbind(get_cond(m0, "未控制 RR"), get_cond(m1, "控制 RR 后"))
cmp$ci_low <- cmp$beta - 1.96 * cmp$se
cmp$ci_high <- cmp$beta + 1.96 * cmp$se
cmp$model <- factor(cmp$model, levels = c("未控制 RR", "控制 RR 后"))

p3c <- ggplot(cmp, aes(x = model, y = beta)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15) +
  geom_point(size = 3, color = "#0072B2") +
  labs(x = "", y = "condition 主效应 β (μV)",
       title = sprintf("Figure 3C: 控制 RR 间期后的 condition 效应 (%s vs %s)",
                       BASE_LBL, CONT_LBL)) +
  theme_bhci()

ggsave(file.path(FIG_DIR, "fig3C_resp_control.pdf"), p3c, width = 12, height = 8, units = "cm")
cat("已保存 figures/fig3C_resp_control.pdf\n")

# 打印数值
print(cmp)
