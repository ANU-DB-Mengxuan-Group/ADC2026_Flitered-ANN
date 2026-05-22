#!/bin/bash
# V2 ML prep — Job B (在 weirdo 2 跑, 跟 Job A 并行)
# 处理 2 个 768d 大数据集: synth_768d_hc, yahoo800k
#
# 用法 (在 weirdo 2, conda env, screen 内):
#   bash analysis/run_v2_ml_prep_jobB.sh

set -e
cd ~/benchmarks/discrete

DATASETS_B=(synth_768d_hc yahoo800k)
LOG_DIR=analysis/v2_prep_logs_jobB_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Job B — 处理: ${DATASETS_B[@]}"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi
HOST=$(hostname); if [[ "$HOST" =~ login ]]; then echo "❌ 在登录节点 ($HOST)"; exit 1; fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "Step B.1: per-query recall (本 job 的 2 个数据集)"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "${DATASETS_B[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2_jobB.csv 2>&1 | tee "$LOG_DIR/perquery.log"
echo "  ⏱  $((($(date +%s) - START))) s"

echo ""
echo "Step B.2: dataset_metrics × 2 串行"
for ds in "${DATASETS_B[@]}"; do
    OUT="analysis/${ds}_metrics.json"
    if [ -s "$OUT" ]; then
        echo "  [skip] $ds: 已有 $OUT"; continue
    fi
    echo "  --- $ds ---"
    START=$(date +%s)
    python tools/dataset_metrics.py \
        --data-root ~/benchmarks/datasets/discrete \
        --dataset "$ds" 2>&1 | tee "$LOG_DIR/metrics_${ds}.log" || echo "  ⚠️  $ds 失败, 继续"
    echo "  ⏱  $ds: $((($(date +%s) - START))) s"
done

echo ""
echo "Step B.3: SIEVE per-query (本 job 的 2 个数据集)"
START=$(date +%s)
python analysis/run_sieve_perquery.py \
    --dataset "${DATASETS_B[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobB.csv 2>&1 | tee "$LOG_DIR/sieve.log"
echo "  ⏱  Step B.3 SIEVE: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "Job B 完成: $(date)"
echo "输出:"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_jobB.csv (UNG + Post-filter)"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobB.csv (SIEVE)"
for ds in "${DATASETS_B[@]}"; do
    [ -s "analysis/${ds}_metrics.json" ] && echo "  ✅ analysis/${ds}_metrics.json" || echo "  ❌ analysis/${ds}_metrics.json"
done
echo "============================================================"
