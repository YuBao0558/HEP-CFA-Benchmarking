#!/usr/bin/env bash
# 后台监控 + 自动触发脚本:
#   等待 pipeline/run_all_resumable.sh 跑完, 然后自动依次触发:
#     1. paper_results/code/run_all_results.sh   (主分析结果)
#     2. paper_results/code/run_supp_analyses.sh (补充分析)
#
# 用法: nohup bash pipeline/watch_and_trigger.sh > pipeline/pipeline_data/watch_trigger.log 2>&1 &

set -u
PIPELINE_LOG="/Users/yubao/CODE/public/BBBD/pipeline/pipeline_data/run_all.log"
CODE_DIR="/Users/yubao/CODE/public/BBBD/paper_results/code"

echo "[watch] 监控开始 $(date '+%F %T')"

# 等待 run_all_resumable.sh 进程结束 (轮询, 每 60s)
while true; do
  if ! pgrep -f "run_all_resumable.sh" > /dev/null 2>&1; then
    # 确认日志里有"全量管线结束"标记
    if grep -q "全量管线结束" "$PIPELINE_LOG" 2>/dev/null; then
      echo "[watch] pipeline 全部完成 $(date '+%F %T')"
      break
    else
      # run_all_resumable.sh 进程没了但没标记完成 → 可能异常退出, 再等一次
      echo "[watch] run_all_resumable.sh 进程消失但未见完成标记, 等待 30s 复检"
      sleep 30
    fi
  fi
  sleep 60
done

# 触发主分析结果
echo "[watch] 开始触发 run_all_results.sh $(date '+%F %T')"
cd "$CODE_DIR"
bash run_all_results.sh 2>&1 | tee "$CODE_DIR/run_all_results.log"

# 触发补充分析
echo "[watch] 开始触发 run_supp_analyses.sh $(date '+%F %T')"
bash run_supp_analyses.sh 2>&1 | tee "$CODE_DIR/run_supp_analyses.log"

echo "[watch] 全部触发完成 $(date '+%F %T')"
