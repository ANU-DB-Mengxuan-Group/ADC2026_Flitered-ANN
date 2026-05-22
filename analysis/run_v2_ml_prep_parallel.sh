#!/bin/bash
# V2 ML router prep — 并行版本 (省时间)
#
# 并行结构:
#   Phase 1: 同时跑
#     - extract_v2_perquery_recall.py        (5-10 分钟, 单进程)
#     - dataset_metrics × 5 (并行, 5 个数据集同时算)
#   Phase 2: 等 Phase 1 全部完成后跑
#     - build_ml_validation_data
#     - train_ml_router_v2
#
# 用法 (在 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   screen -S v2_prep
#   bash analysis/run_v2_ml_prep_parallel.sh
#   Ctrl+A D 脱离
#
# 总预估: 30 分钟 - 1.5 小时 (受 dataset_metrics 最慢那个限制)
#   并行省了串行时的 2-3 小时

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
LOG_DIR=analysis/v2_prep_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "V2 ML router prep — 并行版"
echo "总开始: $(date)"
echo "日志目录: $LOG_DIR"
echo "============================================================"

# 环境检查
if [ -z "${CONDA_DEFAULT_ENV}" ] || [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then
    echo "❌ conda env 不是 benchmark"; exit 1
fi
HOST=$(hostname)
if [[ "$HOST" =~ login ]]; then
    echo "❌ 当前在登录节点 ($HOST)"; exit 1
fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "============================================================"
echo "Phase 1: 并行启动 (per-query recall + 5 dataset_metrics)"
echo "============================================================"
PHASE1_START=$(date +%s)

# ── 并行任务 1: per-query recall ──
echo "  → 启动 extract_v2_perquery_recall.py (后台)"
(
    LOG="$LOG_DIR/perquery_recall.log"
    python analysis/ml_router/extract_v2_perquery_recall.py > "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] perquery_recall 完成 (exit=$?)"
) &
PID_PQ=$!

# ── 并行任务 2-6: dataset_metrics × 5 ──
declare -A PIDS
for ds in "${V2_DATASETS[@]}"; do
    OUT="analysis/${ds}_metrics.json"
    if [ -s "$OUT" ]; then
        echo "  → [skip] $ds: metrics 已有"
        continue
    fi
    echo "  → 启动 dataset_metrics: $ds (后台)"
    (
        LOG="$LOG_DIR/metrics_${ds}.log"
        python tools/dataset_metrics.py \
            --data-root ~/benchmarks/datasets/discrete \
            --dataset "$ds" > "$LOG" 2>&1
        echo "[$(date +%H:%M:%S)] metrics $ds 完成 (exit=$?)"
    ) &
    PIDS[$ds]=$!
done

# ── 等所有后台任务 ──
echo ""
echo "  等待所有 Phase 1 任务... (用 'tail -f $LOG_DIR/*.log' 看进度)"
wait "$PID_PQ"
for ds in "${!PIDS[@]}"; do
    wait "${PIDS[$ds]}" || echo "  ⚠️  $ds 退出非 0"
done

PHASE1_END=$(date +%s)
echo "  ⏱  Phase 1 总用时: $((PHASE1_END - PHASE1_START))s"

# 检查 Phase 1 输出
echo ""
echo "  Phase 1 输出检查:"
PERQUERY_CSV="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
if [ -s "$PERQUERY_CSV" ]; then
    echo "    ✅ $PERQUERY_CSV ($(wc -l < $PERQUERY_CSV) 行)"
else
    echo "    ❌ $PERQUERY_CSV 缺/空"
fi
for ds in "${V2_DATASETS[@]}"; do
    OUT="analysis/${ds}_metrics.json"
    if [ -s "$OUT" ]; then
        echo "    ✅ $OUT"
    else
        echo "    ❌ $OUT 缺/空"
    fi
done

echo ""
echo "============================================================"
echo "Phase 2: 串行 (build + train)"
echo "============================================================"
PHASE2_START=$(date +%s)

ML_V2_DATA="analysis/ml_v2_data.csv"
OUT_DIR="analysis/ml_router/results/v2_on_v2"

echo ""
echo "Step 2.1: build_ml_validation_data"
START=$(date +%s)
python analysis/ml_router/build_ml_validation_data.py \
    --data-root ~/benchmarks/datasets/discrete \
    --recall-csv "$PERQUERY_CSV" \
    --output "$ML_V2_DATA" \
    --datasets "${V2_DATASETS[@]}" 2>&1 | tee "$LOG_DIR/build_ml_v2.log"
END=$(date +%s)
echo "  ⏱  Step 2.1 用时: $((END - START))s"

echo ""
echo "Step 2.2: train_ml_router_v2"
mkdir -p "$OUT_DIR"
START=$(date +%s)
python analysis/ml_router/train_ml_router_v2.py \
    --mode full \
    --val-recall "$PERQUERY_CSV" \
    --val-features "$ML_V2_DATA" \
    --feature-set minimal \
    --regression \
    --output-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/train_ml_v2.log"
END=$(date +%s)
echo "  ⏱  Step 2.2 用时: $((END - START))s"

PHASE2_END=$(date +%s)
echo "  ⏱  Phase 2 总用时: $((PHASE2_END - PHASE2_START))s"

echo ""
echo "============================================================"
echo "全部完成: $(date)"
echo "总用时: $((PHASE2_END - PHASE1_START))s"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  - $PERQUERY_CSV"
echo "  - analysis/{ds}_metrics.json (5 个)"
echo "  - $ML_V2_DATA"
echo "  - $OUT_DIR/full_train_val_results_*.json"
echo ""
echo "看 ML router recall:"
echo "  cat $OUT_DIR/full_train_val_results_minimal_reg.json | python3 -m json.tool"
echo ""
echo "并行任务的详细日志:"
ls -la "$LOG_DIR/"
