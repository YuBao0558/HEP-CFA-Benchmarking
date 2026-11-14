#!/usr/bin/env bash
# ICA 40 成分全量重跑驱动脚本 (跳过与 ICA 无关的 00/01/08, 以及自包含的 06b 扫描)
# 20 成分旧结果已备份至 pipeline_data/data_n20_backup/
set -u

# 关键: 关闭 .pyc 字节码生成, 避免触发 safe-delete 对 __pycache__ 清理的拦截。
export PYTHONDONTWRITEBYTECODE=1
export ICA_N_COMPONENTS=40
export BBBD_EXPERIMENT_ALL=1
PYTHON=/Users/yubao/anaconda3/envs/EEG/bin/python
RSCRIPT=Rscript

cd /Users/yubao/CODE/public/BBBD/pipeline
LOG=/Users/yubao/CODE/public/BBBD/pipeline/pipeline_data/rerun_n40.log

echo "[$(date '+%F %T')] ICA 40 成分全量重跑开始 (PID $$)" | tee -a "$LOG"

for EXP in 2 3 4 5; do
    export BBBD_EXPERIMENT=$EXP
    for STEP in 02_preprocess 03b_hep_dual 03c_pseudotrial 03d_singletrial 03e_topography \
                04_surrogate_l1 05_coupling_l2 06_multiverse 06c_energy_ratio_only; do
        echo "[$(date '+%F %T')] EXP$EXP $STEP 开始" >> "$LOG"
        if $PYTHON ${STEP}.py --all >> "$LOG" 2>&1; then
            echo "[$(date '+%F %T')] EXP$EXP $STEP 完成" >> "$LOG"
        else
            echo "[$(date '+%F %T')] [失败] EXP$EXP $STEP —— 继续下一步" >> "$LOG"
        fi
    done
    # 06d: 分布检验 (无 --all 参数)
    echo "[$(date '+%F %T')] EXP$EXP 06d_cfa_distribution_test 开始" >> "$LOG"
    $PYTHON 06d_cfa_distribution_test.py >> "$LOG" 2>&1 \
        && echo "[$(date '+%F %T')] EXP$EXP 06d 完成" >> "$LOG" \
        || echo "[$(date '+%F %T')] [失败] EXP$EXP 06d" >> "$LOG"
    # 07: LMM 主模型 (R)
    echo "[$(date '+%F %T')] EXP$EXP 07_lmm 开始" >> "$LOG"
    $RSCRIPT --vanilla 07_lmm.R >> "$LOG" 2>&1 \
        && echo "[$(date '+%F %T')] EXP$EXP 07_lmm 完成" >> "$LOG" \
        || echo "[$(date '+%F %T')] [失败] EXP$EXP 07_lmm" >> "$LOG"
done

echo "[$(date '+%F %T')] 全部完成" >> "$LOG"
