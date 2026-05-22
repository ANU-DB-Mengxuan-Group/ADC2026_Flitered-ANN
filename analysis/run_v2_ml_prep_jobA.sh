#!/bin/bash
# V2 ML prep — Job A (在 weirdo 1 跑)
# 处理 3 个数据集: synth_192d, synth_512d, dbpedia560k
#
# Job A + Job B 并行跑, 完了再用 finalize 脚本合并 + 训练.
#
# 用法 (在 weirdo 1, conda env, screen 内):
#   bash analysis/run_v2_ml_prep_jobA.sh

set -e
cd ~/benchmarks/discrete

DATASETS_A=(synth_192d synth_512d dbpedia560k)
LOG_DIR=analysis/v2_prep_logs_jobA_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Job A — 处理: ${DATASETS_A[@]}"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi
HOST=$(hostname); if [[ "$HOST" =~ login ]]; then echo "❌ 在登录节点 ($HOST)"; exit 1; fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "Step A.1: per-query recall (本 job 的 3 个数据集)"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "${DATASETS_A[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2_jobA.csv 2>&1 | tee "$LOG_DIR/perquery.log"
echo "  ⏱  $((($(date +%s) - START))) s"

echo ""
echo "Step A.2: dataset_metrics × 3 串行"
for ds in "${DATASETS_A[@]}"; do
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
echo "Step A.3: SIEVE per-query (本 job 的 3 个数据集)"
START=$(date +%s)
python analysis/run_sieve_perquery.py \
    --dataset "${DATASETS_A[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobA.csv 2>&1 | tee "$LOG_DIR/sieve.log"
echo "  ⏱  Step A.3 SIEVE: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "Job A 完成: $(date)"
echo "输出:"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_jobA.csv (UNG + Post-filter)"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobA.csv (SIEVE)"
for ds in "${DATASETS_A[@]}"; do
    [ -s "analysis/${ds}_metrics.json" ] && echo "  ✅ analysis/${ds}_metrics.json" || echo "  ❌ analysis/${ds}_metrics.json"
done
echo "============================================================"
