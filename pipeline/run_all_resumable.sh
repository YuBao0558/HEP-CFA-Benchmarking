#!/usr/bin/env bash
# =============================================================================
# 一键分析总控 (可断点续跑) —— ICA=40 成分统一重跑
#
# 特性:
#   1. 幂等 + 版本感知: 每步产物完成后写版本戳 (ICA 成分数), 中断重启后
#      只补跑"缺失/版本不符"的步骤, 不会重复计算, 也不会误把旧 20/60 成分
#      结果当作完整而跳过。
#   2. 断点续跑: 每个实验的每步独立判断, Ctrl-C / 关机后重跑本脚本即可续跑。
#   3. 自动备份旧 CSV 产物 (体积小), 删除旧 .fif 中间产物 (体积大 40G,
#      由 02_preprocess overwrite 重新生成, 无需备份)。
#   4. 失败不中断: 单步失败只记录, 继续后续步骤 (与旧 run_one_experiment 一致)。
#
# 用法:
#   bash pipeline/run_all_resumable.sh                # 前台跑
#   nohup bash pipeline/run_all_resumable.sh > pipeline_data/run_all.log 2>&1 &
#
# 环境变量:
#   ICA_N_COMPONENTS  ICA 成分数 (默认 40)
#   EXP_IDS           要跑的实验 (默认 "2 3 4 5")
#   SKIP_BACKUP=1      跳过备份 (再次续跑时避免重复备份)
# =============================================================================
set -u

PYTHON=/Users/yubao/anaconda3/envs/EEG/bin/python
RSCRIPT=Rscript
N_COMP="${ICA_N_COMPONENTS:-40}"
EXP_IDS="${EXP_IDS:-2 3 4 5}"

# 关闭 .pyc 生成, 规避 safe-delete 对 __pycache__ 的批量删除拦截。
export PYTHONDONTWRITEBYTECODE=1
# joblib 走线程后端, 不产生 mmap 临时文件 (common.py 已设, 这里兜底)。
export JOBLIB_MULTIPROCESSING=0
export ICA_N_COMPONENTS="$N_COMP"

BBBD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="$BBBD_DIR/pipeline"
PD="$PIPELINE_DIR/pipeline_data"
DATA_ROOT="$PD/data"
DERIV_ROOT="$PD/derivatives_bhci"
BACKUP_DIR="$PD/backup_n${N_COMP}_pre"

cd "$PIPELINE_DIR"

log()  { echo ""; echo "=================================================="; echo "[$(date '+%F %T')] $*"; echo "=================================================="; }

# 各实验有效被试数 (与 common.py EXCLUDE 对齐)
n_subs_of() {
  case "$1" in
    2) echo 29 ;;  # 排除 sub-08, sub-28
    3) echo 28 ;;  # 排除 sub-15
    4) echo 43 ;;
    5) echo 48 ;;
    *) echo 30 ;;
  esac
}

# 统计 CSV 第一列唯一 sub 数
count_subs() { awk -F, 'NR>1 && $1 ~ /^sub-/ {print $1}' "$1" 2>/dev/null | sort -u | wc -l | tr -d ' '; }
count_rows() { awk 'END{print NR-1}' "$1" 2>/dev/null; }

# 写版本戳: 记录本次产物的 ICA 成分数 (戳文件与产物同名 + .icaN 后缀)
stamp_write() { echo "$N_COMP" > "$1.ica$N_COMP"; }
# 读版本戳: 与当前成分数一致返回 0
stamp_ok() { [ -f "$1.ica$N_COMP" ] && [ "$(cat "$1.ica$N_COMP")" = "$N_COMP" ]; }

# 完整性检查: 文件存在 + 行数/sub数达标 + 该文件自身版本戳一致
is_complete() {
  local kind="$1" file="$2" expect="$3"
  [ -f "$file" ] || return 1
  stamp_ok "$file" || return 1
  local n
  if [ "$kind" = "subs" ]; then n=$(count_subs "$file"); else n=$(count_rows "$file"); fi
  [ "${n:-0}" -ge "$expect" ] 2>/dev/null
}

# 跑一步: 完整则跳过; 否则运行, 成功则写该产物自身的版本戳
run_step() {
  local kind="$1" outfile="$2" expect="$3" label="$4"; shift 4
  local path
  path="$DATA_DIR/$outfile"
  if is_complete "$kind" "$path" "$expect"; then
    echo "[跳过] $label (产物完整且版本一致)"
    return 0
  fi
  log "运行: $label ($outfile)"
  if "$@"; then
    stamp_write "$path"
    echo "[完成] $label"
  else
    echo "[失败(继续)] $label —— 命令: $*" >&2
  fi
}

# =============================================================================
# 0. 备份旧产物 (仅首次)
# =============================================================================
if [ "${SKIP_BACKUP:-0}" != "1" ] && [ ! -d "$BACKUP_DIR" ]; then
  log "备份旧 CSV 产物 -> $BACKUP_DIR"
  mkdir -p "$BACKUP_DIR"
  # 备份所有实验的 data (CSV/json 等小文件), 跳过 .fif (体积大, 会重新生成)
  for e in $EXP_IDS; do
    [ -d "$DATA_ROOT/experiment$e" ] && cp -R "$DATA_ROOT/experiment$e" "$BACKUP_DIR/" 2>/dev/null
  done
  # 备份跨实验汇总
  [ -f "$DATA_ROOT/cross_dataset_effects.csv" ] && cp "$DATA_ROOT/cross_dataset_effects.csv" "$BACKUP_DIR/" 2>/dev/null
  # 清理旧 .fif 中间产物 (40G, 由 02_preprocess 重新生成), 释放磁盘
  if [ -d "$DERIV_ROOT" ]; then
    log "清理旧 .fif 中间产物 (将被重新生成) -> 释放磁盘"
    rm -rf "$DERIV_ROOT"
  fi
else
  log "跳过备份 (SKIP_BACKUP=1 或备份已存在)"
fi

# =============================================================================
# 主循环: 逐实验逐步骤
# =============================================================================
log "全量管线开始 (ICA=$N_COMP 成分, 实验: $EXP_IDS)"

for EXP_ID in $EXP_IDS; do
  export BBBD_EXPERIMENT="$EXP_ID"
  N_SUBS=$(n_subs_of "$EXP_ID")
  DATA_DIR="$DATA_ROOT/experiment$EXP_ID"
  mkdir -p "$DATA_DIR"

  log "进入实验 experiment$EXP_ID (有效被试 $N_SUBS)"

  run_step subs subjects.csv                  "1"       "00 数据盘点" \
    $PYTHON 00_inventory.py

  run_step subs rpeak_validation.csv          "1"       "01 R峰验证" \
    $PYTHON 01_rpeak_validation.py --all

  run_step subs preprocess_report.csv         "$N_SUBS" "02 预处理(双轨)" \
    $PYTHON 02_preprocess.py --all

  run_step subs hep_dual.csv                  "$N_SUBS" "03b HEP双轨" \
    $PYTHON 03b_hep_dual.py --all

  run_step subs hep_pseudotrial.csv           "$N_SUBS" "03c pseudotrial" \
    $PYTHON 03c_pseudotrial.py --all

  run_step subs hep_singletrial.csv           "$N_SUBS" "03d 单试次" \
    $PYTHON 03d_singletrial.py --all

  run_step subs hep_topography.csv            "$N_SUBS" "03e 地形图" \
    $PYTHON 03e_topography.py --all

  run_step subs surrogate_l1.csv              "$N_SUBS" "04 surrogate" \
    $PYTHON 04_surrogate_l1.py --all

  run_step subs coupling_long.csv             "$N_SUBS" "05 脑心耦合ITPC" \
    $PYTHON 05_coupling_l2.py --all

  run_step rows multiverse_effects.csv        "1296"    "06 多宇宙" \
    $PYTHON 06_multiverse.py --all

  run_step subs cfa_energy_ratio_distribution.csv "$N_SUBS" "06c 能量比分布" \
    $PYTHON 06c_energy_ratio_only.py --all

  run_step rows cfa_distribution_stats.csv    "1"       "06d CFA分布检验" \
    $PYTHON 06d_cfa_distribution_test.py

  run_step subs cfa_threshold_scan.csv        "$N_SUBS" "06b CFA阈值扫描" \
    $PYTHON 06b_cfa_threshold_scan.py --all

  run_step subs hrv_long.csv                  "$N_SUBS" "08 HRV频域分析" \
    $PYTHON 08_hrv.py --all

  run_step rows lmm_coefs.csv                 "2"       "07 LMM主模型" \
    $RSCRIPT 07_lmm.R

  log "实验 experiment$EXP_ID 完成"
done

# =============================================================================
# 跨实验汇总
# =============================================================================
log "09 跨实验效应量汇总"
if $PYTHON 09_cross_dataset.py; then
  echo "[完成] 09_cross_dataset"
else
  echo "[失败(继续)] 09_cross_dataset" >&2
fi

log "全量管线结束"
echo "输出: $DATA_ROOT/experiment{2,3,4,5}/"
