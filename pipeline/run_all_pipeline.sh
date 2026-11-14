#!/usr/bin/env bash
# 总控脚本: 依次跑 4 个实验 (exp2/3/4/5) 的完整管线, 最后跨实验汇总。
# 中间不中断 (每步失败只记录继续), 幂等 (已完整产物自动跳过)。
#
# 用法:
#   bash pipeline/run_all_pipeline.sh
#   nohup bash pipeline/run_all_pipeline.sh > run_all_pipeline.log 2>&1 &
set -u

PYTHON=/Users/yubao/anaconda3/envs/EEG/bin/python
BBBD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="$BBBD_DIR/pipeline"
cd "$PIPELINE_DIR"

echo "################# 全量管线开始 $(date '+%Y-%m-%d %H:%M:%S') #################"

for EXP_ID in 2 3 4 5; do
  echo ""
  echo "############################################################"
  echo "### 进入实验 experiment$EXP_ID"
  echo "############################################################"
  BBBD_EXPERIMENT=$EXP_ID bash run_one_experiment.sh
  echo "### 实验 experiment$EXP_ID 流程结束 (退出码 $?)"
done

echo ""
echo "############################################################"
echo "### 09 跨实验效应量汇总"
echo "############################################################"
$PYTHON 09_cross_dataset.py || echo "[失败(继续)] 09_cross_dataset"

echo ""
echo "################# 全量管线结束 $(date '+%Y-%m-%d %H:%M:%S') #################"
echo "输出: pipeline/pipeline_data/data/experiment{2,3,4,5}/ 与 pipeline/pipeline_data/derivatives_bhci/"
