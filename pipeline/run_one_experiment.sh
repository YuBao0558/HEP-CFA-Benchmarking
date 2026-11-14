#!/usr/bin/env bash
# 单实验完整管线 (幂等, 中间不中断)。
# 由 run_all_pipeline.sh 调用, 或单独: BBBD_EXPERIMENT=3 bash run_one_experiment.sh
#
# 产物完整性判断 (两类):
#   - "subs" 型: 按 CSV 第一列的唯一 sub 数 (>= 有效被试数)
#   - "rows" 型: 按数据行数 (用于 multiverse=1296 固定值, lmm 系数表等)
set -u

PYTHON=/Users/yubao/anaconda3/envs/EEG/bin/python
RSCRIPT=Rscript

BBBD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="$BBBD_DIR/pipeline"
cd "$PIPELINE_DIR"

EXP_ID="${BBBD_EXPERIMENT:?需设置 BBBD_EXPERIMENT}"
case "$EXP_ID" in
  2) N_SUBS=29 ;;
  3) N_SUBS=28 ;;
  4) N_SUBS=43 ;;
  5) N_SUBS=48 ;;
  *) N_SUBS=30 ;;
esac

DATA_DIR="$PIPELINE_DIR/pipeline_data/data/experiment$EXP_ID"

log() { echo ""; echo "=================================================="; echo "[$(date '+%H:%M:%S')] $*"; echo "=================================================="; }

# 统计 CSV 第一列唯一 sub 数 (假设 sub-XX 在第一列)
count_subs() {
  awk -F, 'NR>1 && $1 ~ /^sub-/ {print $1}' "$1" 2>/dev/null | sort -u | wc -l | tr -d ' '
}
count_rows() {
  awk 'END{print NR-1}' "$1" 2>/dev/null
}

# 完整性检查: kind=subs 或 rows, 返回 0=完整 1=不完整/缺失
is_complete() {
  local kind="$1" file="$2" expect="$3"
  [ -f "$file" ] || return 1
  local n
  if [ "$kind" = "subs" ]; then n=$(count_subs "$file"); else n=$(count_rows "$file"); fi
  [ "${n:-0}" -ge "$expect" ] 2>/dev/null
}

# 跑一步: 完整则跳过, 否则运行 (失败只警告不中断)
run_step() {
  local kind="$1" outfile="$2" expect="$3" label="$4"; shift 4
  local path="$DATA_DIR/$outfile"
  if is_complete "$kind" "$path" "$expect"; then
    echo "[跳过] $label (产物已完整: $outfile)"
    return 0
  fi
  log "运行: $label ($outfile)"
  if "$@"; then
    echo "[完成] $label"
  else
    echo "[失败(继续)] $label —— 命令: $*" >&2
  fi
}

log "实验 experiment$EXP_ID 开始 (有效被试 $N_SUBS)"

run_step subs subjects.csv                  "1"       "00 数据盘点" \
  $PYTHON 00_inventory.py

run_step subs rpeak_validation.csv          "1"       "01 R峰验证" \
  $PYTHON 01_rpeak_validation.py --all

run_step subs preprocess_report.csv        "$N_SUBS" "02 预处理(双轨)" \
  $PYTHON 02_preprocess.py --all

run_step subs hep_dual.csv                 "$N_SUBS" "03b HEP双轨" \
  $PYTHON 03b_hep_dual.py --all

run_step subs hep_pseudotrial.csv          "$N_SUBS" "03c pseudotrial" \
  $PYTHON 03c_pseudotrial.py --all

run_step subs hep_singletrial.csv          "$N_SUBS" "03d 单试次" \
  $PYTHON 03d_singletrial.py --all

run_step subs hep_topography.csv           "$N_SUBS" "03e 地形图" \
  $PYTHON 03e_topography.py --all

run_step subs surrogate_l1.csv             "$N_SUBS" "04 surrogate" \
  $PYTHON 04_surrogate_l1.py --all

run_step subs coupling_long.csv            "$N_SUBS" "05 脑心耦合ITPC" \
  $PYTHON 05_coupling_l2.py --all

run_step rows multiverse_effects.csv       "1296"    "06 多宇宙" \
  $PYTHON 06_multiverse.py --all

run_step subs cfa_energy_ratio_distribution.csv "$N_SUBS" "06c 能量比(20成分)" \
  $PYTHON 06c_energy_ratio_only.py --all

run_step rows cfa_distribution_stats.csv   "1"       "06d CFA分布检验" \
  $PYTHON 06d_cfa_distribution_test.py

run_step subs cfa_threshold_scan.csv       "$N_SUBS" "06b CFA阈值扫描" \
  $PYTHON 06b_cfa_threshold_scan.py --all

run_step subs hrv_long.csv                "$N_SUBS" "08 HRV频域分析" \
  $PYTHON 08_hrv.py --all

run_step rows lmm_coefs.csv                "2"       "07 LMM主模型" \
  $RSCRIPT 07_lmm.R

log "实验 experiment$EXP_ID 完成"
