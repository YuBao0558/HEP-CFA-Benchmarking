#!/usr/bin/env Rscript
# Step 7: 状态调制主模型 (LMM) + 个体差异调节分析
# 输入: data/<experiment>/hep_dual.csv (Step3双轨, 全量后) + <experiment>/participants.tsv
# 输出: data/<experiment>/lmm_coefs.csv (主模型) + data/<experiment>/lmm_moderation.csv (调节分析)
#
# 实验切换: 通过环境变量 BBBD_EXPERIMENT 选择数据集 (默认 "2")。
#   Rscript pipeline/07_lmm.R  (默认 exp2)
#   BBBD_EXPERIMENT=4 Rscript pipeline/07_lmm.R
# 路径: 用脚本自身位置 (--file=) 定位 paper_results/ (脚本的上一级目录),
#       与 Python common.py 的 BASE 一致; 也可用 BBBD_ANALYSIS_DIR 显式覆盖。
#
# 模型 (操作定义 §7.1):
#   主模型:   BHCI ~ condition + (1 | sub)
#   调节分析: BHCI ~ condition * covariate + (1 | sub)
#     - 探索性: 查 per-subject Δ 的异质性是否由协变量解释
#
# 注: 预注册主假说已冻结 (condition 主效应)。调节分析是探索性补充,
#     不挤占预注册结论, 论文中需明确标注。

suppressMessages({
  library(lme4)
  library(lmerTest)   # Satterthwaite 自由度 Pr(>|t|)
  library(dplyr)
  library(tidyr)
})

# ---- 实验切换 (与 Python common.py 的 BBBD_EXPERIMENT 对齐) ----
BBBD_EXPERIMENT <- Sys.getenv("BBBD_EXPERIMENT", unset = "2")
EXP <- list(
  "2" = list(dir = "experiment2",
             sessions = c("ses-01" = "attentive", "ses-02" = "distracted"),
             base = "attentive", contrast = "distracted"),
  "3" = list(dir = "experiment3",
             sessions = c("ses-01" = "attentive", "ses-02" = "distracted"),
             base = "attentive", contrast = "distracted"),
  "4" = list(dir = "experiment4",
             sessions = c("ses-01" = "attentive1", "ses-02" = "attentive2"),
             base = "attentive1", contrast = "attentive2"),
  "5" = list(dir = "experiment5",
             sessions = c("ses-01" = "attentive", "ses-02" = "incentive"),
             base = "attentive", contrast = "incentive")
)
if (!(BBBD_EXPERIMENT %in% names(EXP))) {
  stop(sprintf("未知实验 BBBD_EXPERIMENT=%s, 可用: %s",
               BBBD_EXPERIMENT, paste(names(EXP), collapse = ",")))
}
cfg <- EXP[[BBBD_EXPERIMENT]]

# ---- 路径: 优先用环境变量 BBBD_ANALYSIS_DIR (即 pipeline/pipeline_data/), 否则用脚本位置 ----
# Rscript 下命令行含 --file=<脚本路径>。若无 (如 RStudio source), 报错而非猜测。
args_file <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_file, value = TRUE)
if (length(file_arg) > 0) {
  SCRIPT_PATH <- sub("^--file=", "", file_arg[1])
  SCRIPT_DIR <- normalizePath(dirname(SCRIPT_PATH))
} else {
  SCRIPT_DIR <- getwd()
}
analysis_override <- Sys.getenv("BBBD_ANALYSIS_DIR", unset = "")
if (nzchar(analysis_override)) {
  ANALYSIS_DIR <- normalizePath(analysis_override)
} else {
  ANALYSIS_DIR <- normalizePath(file.path(SCRIPT_DIR, "pipeline_data"))  # pipeline/ -> pipeline/pipeline_data/
}
BBBD_DIR <- normalizePath(file.path(ANALYSIS_DIR, "..", ".."))          # pipeline_data/ -> BBBD/
DATA_DIR <- file.path(ANALYSIS_DIR, "data", cfg$dir)
FIG_DIR  <- file.path(ANALYSIS_DIR, "figures", cfg$dir)
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR,  showWarnings = FALSE, recursive = TRUE)

# ---- 数据准备 ----
d <- read.csv(file.path(DATA_DIR, "hep_dual.csv"), stringsAsFactors = FALSE)
d_clean <- d %>% filter(track == "clean", roi == "frontocentral")

# 每个被试×会话×刺激的主窗口 RMS (保留刺激水平, 供随机效应模型)。
# 主分析不做基线校正 (Steinfath 2025/Petzschner 2019/Azzalini 2019): 基线窗会
# 重叠前一跳 HEP 晚期 (RR<0.9s 占 79%), 故 bhci 直接取未校正值; 校正值
# (bhci_corrected) 降级为敏感性对照 (审稿人对照)。
d_stim <- d_clean %>%
  mutate(condition = factor(condition, levels = c(cfg$base, cfg$contrast)),
         session = ifelse(ses == "ses-01", 0, 1)) %>%
  select(sub, ses, stim, session, condition,
         bhci = main_200_600_rms_uv_raw,
         bhci_corrected = main_200_600_rms_uv)

# 被试×会话水平 (主模型, 平均掉刺激)
d_bhci <- d_stim %>%
  group_by(sub, ses, condition, session) %>%
  summarise(bhci = mean(bhci, na.rm = TRUE),
            bhci_corrected = mean(bhci_corrected, na.rm = TRUE), .groups = "drop")

if (nrow(d_bhci) < 10 || length(unique(d_bhci$sub)) < 3) {
  cat("数据不足 (需全量), 跳过 LMM\n")
  quit(save = "no", status = 0)
}

# ---- 协变量 (participants.tsv) ----
part <- read.csv(file.path(BBBD_DIR, cfg$dir, "participants.tsv"),
                 sep = "\t", stringsAsFactors = FALSE)
names(part)[1] <- "sub"
# 清洗协变量
part$Sex <- factor(part$Sex, levels = c("Female", "Male"))
part$Age <- as.numeric(part$Age)
part$Caffeine <- suppressWarnings(as.numeric(part$Caffeine))
# Tired 归一化: Awake < Moderate < Tired < Very Tired/Very awake
tired_rank <- c("Awake" = 0, "Very awake" = 0, "Moderate" = 1, "Tired" = 2, "Very Tired" = 3)
part$Tired_rank <- tired_rank[part$Tired]

d_bhci <- d_bhci %>% left_join(part, by = "sub")

# ---- 0. epoch 数描述统计 (Steinfath: epoch 不足是 HEP 统计效力低的主因) ----
cat("\n=== epoch 数描述统计 (每被试×条件) ===\n")
ep <- d_clean %>%
  group_by(sub, condition) %>%
  summarise(n_epochs = mean(n_epochs, na.rm = TRUE), .groups = "drop")
if (nrow(ep) > 0) {
  cat(sprintf("  平均 epoch 数: %.1f (中位数 %.1f, 范围 [%.0f, %.0f])\n",
              mean(ep$n_epochs), median(ep$n_epochs), min(ep$n_epochs), max(ep$n_epochs)))
  n_low <- sum(ep$n_epochs < 30)
  cat(sprintf("  epoch < 30 的被试×条件组合: %d / %d%s\n", n_low, nrow(ep),
              ifelse(n_low > 0, "  (需在论文标注)", "")))
}

# ---- 1. 主模型 (预注册冻结) ----
# 重要限制 (设计层面, 非统计可解):
#   本数据集所有被试均按固定顺序完成两个会话 (ses-01 -> ses-02), 无 counterbalance。
#   因此 "condition" 与 "session 顺序" 完全共线, condition 主效应同时承载:
#     注意状态 + 时间/疲劳/习惯化/练习效应。
#   故该主效应只能解释为 "会话(session)差异", 不能解释为 "注意调制"。
#   下面同时拟合 condition 与 session 两个等价模型, 并报告共线诊断。
# 随机效应结构: 先尝试随机斜率 (1 + condition | sub), 奇异则降级为随机截距。
m_main_slope <- tryCatch(
  lmer(bhci ~ condition + (1 + condition | sub), data = d_bhci),
  error = function(e) NULL)
if (!is.null(m_main_slope) && !isSingular(m_main_slope)) {
  m_main <- m_main_slope
  cat("=== 主模型: bhci ~ condition + (1 + condition | sub) [仅可解释为 session 差异] ===\n")
} else {
  m_main <- lmer(bhci ~ condition + (1 | sub), data = d_bhci)
  cat("=== 主模型: bhci ~ condition + (1 | sub) [随机斜率奇异, 降级] ===\n")
}
print(summary(m_main)$coefficients)

# 共线验证: session 与 condition 完全共线, 系数应一致 (仅符号/截距基准不同)
m_ses <- lmer(bhci ~ session + (1 | sub), data = d_bhci)
cat("\n=== 等价模型: bhci ~ session + (1|sub) (验证共线) ===\n")
print(summary(m_ses)$coefficients)

# 刺激随机效应模型 (刺激水平数据, 推广到刺激层面)
cat("\n=== 刺激随机效应模型: bhci ~ condition + (1|sub) + (1|stim) ===\n")
m_stim <- tryCatch(
  lmer(bhci ~ condition + (1 | sub) + (1 | stim), data = d_stim),
  error = function(e) { cat("  [跳过] 模型失败:", conditionMessage(e), "\n"); NULL })
if (!is.null(m_stim)) {
  print(summary(m_stim)$coefficients)
  vc <- as.data.frame(VarCorr(m_stim))
  stim_var <- vc$vcov[vc$grp == "stim"]
  if (isSingular(m_stim)) {
    cat("  [警告] m_stim 奇异拟合, (1|stim) 方差不可靠 (刺激间方差可忽略)\n")
    cat("  (1|stim) 方差:", stim_var, " (stim 方差占比: n/a, 奇异无法可靠估计)\n")
  } else {
    cat("  (1|stim) 方差:", stim_var,
        " (stim 方差占比:",
        round(100 * stim_var / sum(vc$vcov), 1), "%)\n")
  }
}

# 模型诊断: 奇异拟合 + 收敛
cat("\n=== 主模型诊断 ===\n")
if (isSingular(m_main)) cat("  [警告] 主模型奇异拟合 (随机效应方差接近 0)\n")
cat("  收敛: ", tryCatch(m_main@optinfo$conv$opt == 0, error = function(e) NA), "\n")
# 残差近似正态 (Shapiro 只对非极小数)
res <- resid(m_main)
if (length(res) >= 3 && length(res) <= 5000) {
  sh <- tryCatch(shapiro.test(res)$p.value, error = function(e) NA)
  cat(sprintf("  残差 Shapiro-Wilk p = %.4f%s\n", sh, ifelse(!is.na(sh) && sh < 0.05, "  (偏离正态)", "")))
}

# ---- 1a. 基线校正敏感性 (Steinfath 2026: 心跳周期性使基线窗与前一跳重叠) ----
# 主分析为未校正(bhci), 校正(bhci_corrected)作为敏感性对照, 报告两者系数对比。
cat("\n=== 基线校正敏感性: 未校正(主) vs 校正(对照) 主效应 ===\n")
m_corr <- tryCatch(lmer(bhci_corrected ~ condition + (1 | sub), data = d_bhci),
                   error = function(e) NULL)
if (!is.null(m_corr)) {
  co_main <- summary(m_main)$coefficients
  co_corr <- summary(m_corr)$coefficients
  cond_rows <- grep("condition", rownames(co_main), value = TRUE)
  for (tr in cond_rows) {
    b_main <- co_main[tr, "Estimate"]; p_main <- co_main[tr, "Pr(>|t|)"]
    b_corr <- co_corr[tr, "Estimate"]; p_corr <- co_corr[tr, "Pr(>|t|)"]
    cat(sprintf("  %s: 未校正 β=%.4f (p=%.4f) | 校正 β=%.4f (p=%.4f)\n",
                tr, b_main, p_main, b_corr, p_corr))
  }
  write.csv(data.frame(
    term = cond_rows,
    beta_raw = co_main[cond_rows, "Estimate"],
    p_raw = co_main[cond_rows, "Pr(>|t|)"],
    beta_corrected = co_corr[cond_rows, "Estimate"],
    p_corrected = co_corr[cond_rows, "Pr(>|t|)"]
  ), file.path(DATA_DIR, "lmm_baseline_sensitivity.csv"), row.names = FALSE)
  cat(sprintf("已保存 %s/lmm_baseline_sensitivity.csv\n", DATA_DIR))
}

# ---- 1b. Pseudotrial 校正对比 (Steinfath 2025) ----
# 用 hep_pseudotrial.csv 的 corrected 列跑主模型, 对比校正前后主效应。
# 若校正后效应消失 → 原始效应是心跳无关的; 若保留/增强 → 真正心跳锁定活动。
cat("\n=== Pseudotrial 校正对比 ===\n")
pseudo_path <- file.path(DATA_DIR, "hep_pseudotrial.csv")
if (file.exists(pseudo_path)) {
  d_pseudo <- read.csv(pseudo_path, stringsAsFactors = FALSE)
  d_pseudo_fc <- d_pseudo %>%
    filter(roi == "frontocentral") %>%
    mutate(condition = factor(condition, levels = c(cfg$base, cfg$contrast)),
           session = ifelse(ses == "ses-01", 0, 1)) %>%
    group_by(sub, ses, condition, session) %>%
    summarise(bhci_corr = mean(main_200_600_corrected, na.rm = TRUE),
              bhci_true = mean(main_200_600_true, na.rm = TRUE), .groups = "drop")
  m_pseudo <- tryCatch(lmer(bhci_corr ~ condition + (1 | sub), data = d_pseudo_fc),
                       error = function(e) NULL)
  m_true <- tryCatch(lmer(bhci_true ~ condition + (1 | sub), data = d_pseudo_fc),
                     error = function(e) NULL)
  if (!is.null(m_pseudo) && !is.null(m_true)) {
    co_ps <- summary(m_pseudo)$coefficients
    co_tr <- summary(m_true)$coefficients
    for (tr in grep("condition", rownames(co_ps), value = TRUE)) {
      cat(sprintf("  %s: 未校正(真) β=%.4f (p=%.4f) | pseudotrial校正 β=%.4f (p=%.4f)\n",
                  tr, co_tr[tr, "Estimate"], co_tr[tr, "Pr(>|t|)"],
                  co_ps[tr, "Estimate"], co_ps[tr, "Pr(>|t|)"]))
    }
    write.csv(data.frame(
      term = grep("condition", rownames(co_ps), value = TRUE),
      beta_true = co_tr[grep("condition", rownames(co_tr), value = TRUE), "Estimate"],
      p_true = co_tr[grep("condition", rownames(co_tr), value = TRUE), "Pr(>|t|)"],
      beta_corrected = co_ps[grep("condition", rownames(co_ps), value = TRUE), "Estimate"],
      p_corrected = co_ps[grep("condition", rownames(co_ps), value = TRUE), "Pr(>|t|)"]
    ), file.path(DATA_DIR, "lmm_pseudotrial_compare.csv"), row.names = FALSE)
    cat(sprintf("已保存 %s/lmm_pseudotrial_compare.csv\n", DATA_DIR))
  }
} else {
  cat("  [跳过] 未找到 hep_pseudotrial.csv (需先运行 03c_pseudotrial.py)\n")
}

coefs <- as.data.frame(summary(m_main)$coefficients)
coefs$term <- rownames(coefs)
coefs$ci_low <- coefs$Estimate - 1.96 * coefs$`Std. Error`
coefs$ci_high <- coefs$Estimate + 1.96 * coefs$`Std. Error`
coefs$significant <- coefs$`Pr(>|t|)` < 0.05
coefs$note <- "condition 与 session 顺序完全共线(无 counterbalance), 只能解释为 session 差异"
write.csv(coefs, file.path(DATA_DIR, "lmm_coefs.csv"), row.names = FALSE)
cat(sprintf("已保存 %s/lmm_coefs.csv\n\n", DATA_DIR))

# ---- 2. 个体差异调节分析 (探索性) ----
# 每个被试的 Δ = 对照会话 - 基准会话 (clean, frontocentral)
d_wide <- d_bhci %>%
  select(sub, condition, bhci) %>%
  pivot_wider(names_from = condition, values_from = bhci)
missing_cols <- setdiff(c(cfg$base, cfg$contrast), names(d_wide))
if (length(missing_cols) > 0) {
  stop(sprintf("缺少 condition 列: %s (部分被试可能只完成一个会话)",
               paste(missing_cols, collapse = ", ")))
}
n_incomplete <- sum(is.na(d_wide[[cfg$contrast]]) | is.na(d_wide[[cfg$base]]))
if (n_incomplete > 0) {
  cat(sprintf("  [提示] %d 个被试只有单会话, Δ 为 NA, 已从调节分析排除\n", n_incomplete))
}
d_wide <- d_wide %>%
  mutate(delta = .data[[cfg$contrast]] - .data[[cfg$base]]) %>%
  left_join(part, by = "sub")

# ---- 1b. 等效性检验 (TOST, Steinfath 2025) ----
# 主结论若为"无显著差异", 等效性检验可强化零结果: 检验效应是否小于最小有意义边界。
# 边界 Hedges' dz = 0.19 取自 Steinfath et al. 2025; 若无预注册 SESOI, 应做
# 边界敏感性分析 (0.15/0.19/0.25), 报告不同边界下 TOST 结论是否稳健。
EQUIV_BOUNDS <- c(0.15, 0.19, 0.25)
cat(sprintf("\n=== 等效性检验 (TOST, 边界敏感性 %s) ===\n",
            paste(EQUIV_BOUNDS, collapse = "/")))
delta_vals <- d_wide$delta[!is.na(d_wide$delta)]
if (length(delta_vals) >= 3) {
  n_eq <- length(delta_vals)
  d_mean <- mean(delta_vals)
  d_sd <- sd(delta_vals)
  dz <- d_mean / d_sd
  dz_corr <- dz * (1 - 3 / (4 * (n_eq - 1) - 1))  # Hedges 小样本修正 (配对, df=n-1)
  # 配对设计下 dz 的标准误 (近似)
  se_dz <- sqrt((1 / n_eq) + dz_corr^2 / (2 * n_eq))
  # 90% CI (TOST 用 90% 双侧 = 两个 95% 单侧)
  ci90_low <- dz_corr - 1.645 * se_dz
  ci90_high <- dz_corr + 1.645 * se_dz
  cat(sprintf("  配对 Hedges' dz = %.4f (n=%d)\n", dz_corr, n_eq))
  cat(sprintf("  90%% CI = [%.4f, %.4f]\n", ci90_low, ci90_high))

  equiv_tbl <- data.frame()
  for (b in EQUIV_BOUNDS) {
    t_upper <- (dz_corr - b) / se_dz
    t_lower <- (dz_corr + b) / se_dz
    p_upper <- pt(t_upper, df = n_eq - 1, lower.tail = FALSE)
    p_lower <- pt(t_lower, df = n_eq - 1)
    p_equiv <- max(p_upper, p_lower)
    equiv_tbl <- rbind(equiv_tbl, data.frame(boundary = b, p_equiv = round(p_equiv, 4)))
    cat(sprintf("  边界 ±%.2f: TOST p = %.4f  -> %s\n", b, p_equiv,
                ifelse(p_equiv < 0.05, "等价(支持零结果)", "不能排除真实效应")))
  }
  write.csv(equiv_tbl, file.path(DATA_DIR, "lmm_equivalence.csv"), row.names = FALSE)
  cat(sprintf("已保存 %s/lmm_equivalence.csv\n", DATA_DIR))
} else {
  cat("  数据不足 (需 ≥3 被试), 跳过等效性检验\n")
}

covars <- c("Sex", "Age", "Caffeine", "Tired_rank")
cat(sprintf("=== 个体差异: Δ(%s−%s) 与协变量的 Spearman 相关 ===\n",
            cfg$contrast, cfg$base))
moderation <- data.frame()
for (cv in covars) {
  x <- d_wide[[cv]]
  y <- d_wide$delta
  valid <- !is.na(x) & !is.na(y)
  if (sum(valid) < 5) next
  if (is.factor(x)) {
    # 两样本比较
    g <- split(y[valid], x[valid])
    if (length(g) < 2) next
    p <- wilcox.test(g[[1]], g[[2]])$p.value
    rho <- NA
  } else {
    r <- cor(x[valid], y[valid], method = "spearman")
    rho <- r
    p <- cor.test(x[valid], y[valid], method = "spearman")$p.value
  }
  moderation <- rbind(moderation, data.frame(
    covariate = cv, spearman_rho = rho, p = p))
  cat(sprintf("  %-10s rho=% .3f  p=%.4f\n", cv, ifelse(is.na(rho), 0, rho), p))
}

# LMM 交互检验 (对候选协变量)
cat("\n=== LMM 交互检验: bhci ~ condition * covariate + (1|sub) ===\n")
moderation$lmm_p_interaction <- NA_real_   # 预建列, 避免空表时下标越界
for (cv in covars) {
  if (sum(!is.na(d_bhci[[cv]])) < 10) next
  f <- as.formula(sprintf("bhci ~ condition * %s + (1 | sub)", cv))
  m <- tryCatch(lmer(f, data = d_bhci), error = function(e) NULL)
  if (is.null(m)) next
  co <- as.data.frame(summary(m)$coefficients)
  int_row <- grep("condition.*:", rownames(co), value = TRUE)
  if (length(int_row) == 0) next
  p_int <- co[int_row, "Pr(>|t|)"]
  cat(sprintf("  condition × %-10s : p_interaction = %.4f%s\n",
              cv, p_int, ifelse(p_int < 0.05, "  *", "")))
  if (nrow(moderation) == 0) next   # 防御: 无任何 Spearman 记录时不写交互列
  moderation$lmm_p_interaction[moderation$covariate == cv] <- p_int
}

# 多重比较校正 (FDR / Benjamini-Hochberg) 对 Spearman 相关 p 值与交互 p 值
moderation$p_fdr_spearman <- NA
moderation$p_fdr_interaction <- NA
p_spearman <- if (nrow(moderation) > 0) moderation$p[!is.na(moderation$p)] else numeric(0)
p_interact <- if (nrow(moderation) > 0) moderation$lmm_p_interaction[!is.na(moderation$lmm_p_interaction)] else numeric(0)
if (length(p_spearman) > 0) {
  moderation$p_fdr_spearman[!is.na(moderation$p)] <-
    p.adjust(p_spearman, method = "BH")
}
if (length(p_interact) > 0) {
  moderation$p_fdr_interaction[!is.na(moderation$lmm_p_interaction)] <-
    p.adjust(p_interact, method = "BH")
}
cat("\n=== FDR 校正后 (BH) ===\n")
if (nrow(moderation) > 0) {
  print(moderation[, c("covariate", "p_fdr_spearman", "p_fdr_interaction")])
} else {
  cat("  (无协变量通过筛选, 无 FDR 结果)\n")
}

write.csv(moderation, file.path(DATA_DIR, "lmm_moderation.csv"), row.names = FALSE)
cat(sprintf("\n已保存 %s/lmm_moderation.csv\n", DATA_DIR))

# ---- 3. 单试次 LMM (RR 间期 + 呼吸相位协变量) ----
# 用 hep_singletrial.csv (03d 导出) 做单试次混合模型, 纳入 RR 间期(心跳重叠)
# 与呼吸相位(RSA) 作固定效应协变量, 控制试次间不平衡。
cat("\n=== 单试次 LMM: hep_amp ~ condition + rr_prev + resp_phase + (1|sub) ===\n")
st_path <- file.path(DATA_DIR, "hep_singletrial.csv")
if (file.exists(st_path)) {
  d_st <- read.csv(st_path, stringsAsFactors = FALSE)
  d_st$condition <- factor(d_st$condition, levels = c(cfg$base, cfg$contrast))
  d_st$rr_prev_s <- suppressWarnings(as.numeric(d_st$rr_prev_s))
  d_st$resp_phase <- suppressWarnings(as.numeric(d_st$resp_phase))

  has_resp <- any(!is.na(d_st$resp_phase))
  if (has_resp) {
    f_st <- hep_amp_uv ~ condition + rr_prev_s + resp_phase + (1 | sub)
  } else {
    cat("  [提示] 无呼吸相位数据 (exp2/exp3), 仅纳入 RR 间期协变量\n")
    f_st <- hep_amp_uv ~ condition + rr_prev_s + (1 | sub)
  }
  m_st <- tryCatch(lmer(f_st, data = d_st), error = function(e) NULL)
  if (!is.null(m_st)) {
    print(summary(m_st)$coefficients)
    if (isSingular(m_st)) cat("  [警告] 单试次模型奇异拟合\n")
    co_st <- as.data.frame(summary(m_st)$coefficients)
    co_st$term <- rownames(co_st)
    write.csv(co_st, file.path(DATA_DIR, "lmm_singletrial_coefs.csv"), row.names = FALSE)
    cat(sprintf("已保存 %s/lmm_singletrial_coefs.csv\n", DATA_DIR))
  } else {
    cat("  [跳过] 单试次模型失败\n")
  }
} else {
  cat("  [跳过] 未找到 hep_singletrial.csv (需先运行 03d_singletrial.py)\n")
}
