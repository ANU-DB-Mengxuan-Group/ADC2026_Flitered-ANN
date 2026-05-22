#!/bin/bash
# V2 ML prep — 单 weirdo 处理所有 5 个数据集 (用 multiprocessing 并行 DF)
#
# 跟 jobA/jobB/finalize 三件套等价, 但:
#   - 1 个 weirdo 跑全部 (省 SLURM 资源)
#   - dataset_metrics 用 16 进程并行 DF (省时间)
#   - SIEVE per-query 不在本脚本 (建议另开 weirdo 单独跑)
#
# 用法 (在 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   screen -S v2_combined
#   bash analysis/run_v2_ml_prep_combined.sh
#   Ctrl+A D 脱离
#
# 预估: 15-30 分钟 (vs 老 jobA+jobB 串行 70-150 分钟)
#   - extract_v2_perquery_recall: 5-10 分钟
#   - dataset_metrics × 5 串行 (但每个 DF 16 进程并行): 10-20 分钟

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
LOG_DIR=analysis/v2_prep_logs_combined_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "V2 ML prep — 合并版 (单 weirdo + 并行 DF)"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi
HOST=$(hostname); if [[ "$HOST" =~ login ]]; then echo "❌ 在登录节点 ($HOST)"; exit 1; fi
echo "  ✅ env=benchmark, host=$HOST"

# 防止 dataset_metrics 多进程内每进程再开 16 个 MKL 线程导致严重 oversubscribe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "  ✅ OMP/MKL/OPENBLAS = 1 (避免子进程 thread 爆炸)"

echo ""
echo "============================================================"
echo "Step 1: V2 per-query recall (UNG + Post-filter, 全 5 数据集)"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "${V2_DATASETS[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2.csv 2>&1 | tee "$LOG_DIR/perquery.log"
echo "  ⏱  Step 1: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "Step 2: dataset_metrics × 5 (DF 用 16 进程并行)"
echo "============================================================"
for ds in "${V2_DATASETS[@]}"; do
    OUT="analysis/${ds}_metrics.json"
    if [ -s "$OUT" ]; then
        echo "  [skip] $ds: 已有 $OUT"; continue
    fi
    echo ""
    echo "  --- $ds ---"
    START=$(date +%s)
    python tools/dataset_metrics.py \
        --data-root ~/benchmarks/datasets/discrete \
        --dataset "$ds" 2>&1 | tee "$LOG_DIR/metrics_${ds}.log" || echo "  ⚠️  $ds 失败, 继续"
    echo "  ⏱  $ds: $((($(date +%s) - START))) s"
done

echo ""
echo "============================================================"
echo "完成: $(date)"
echo "============================================================"
echo ""
echo "输出:"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
for ds in "${V2_DATASETS[@]}"; do
    [ -s "analysis/${ds}_metrics.json" ] && echo "  ✅ analysis/${ds}_metrics.json" || echo "  ❌ analysis/${ds}_metrics.json"
done
echo ""
echo "下一步:"
echo "  1. 等另一个 weirdo 的 SIEVE per-query 完成"
echo "     → analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv"
echo "  2. 跑 finalize 合并 SIEVE + 训练:"
echo "     bash analysis/run_v2_ml_prep_finalize_combined.sh"
