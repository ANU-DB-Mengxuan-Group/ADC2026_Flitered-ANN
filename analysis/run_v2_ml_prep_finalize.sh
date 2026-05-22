#!/bin/bash
# V2 ML prep — Finalize (Job A + Job B 都完成后跑)
#
# 步骤:
#   1. 合并 jobA + jobB 的 per-query recall CSV
#   2. build_ml_validation_data
#   3. train_ml_router_v2
#
# 用法 (在任一 weirdo, conda env 内):
#   bash analysis/run_v2_ml_prep_finalize.sh

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
PERQUERY_A="analysis/ml_router/perquery_recall/perquery_recall_v2_jobA.csv"
PERQUERY_B="analysis/ml_router/perquery_recall/perquery_recall_v2_jobB.csv"
PERQUERY_SIEVE_A="analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobA.csv"
PERQUERY_SIEVE_B="analysis/ml_router/perquery_recall/perquery_recall_v2_sieve_jobB.csv"
PERQUERY_SIEVE_SINGLE="analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv"  # 备选: 单一文件
PERQUERY_MERGED="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
ML_V2_DATA="analysis/ml_v2_data.csv"
OUT_DIR="analysis/ml_router/results/v2_on_v2"

LOG_DIR=analysis/v2_prep_logs_finalize_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Finalize: 合并 + build + train"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

# 检查前置
[ -s "$PERQUERY_A" ] || { echo "❌ 缺 $PERQUERY_A — 先跑 Job A"; exit 1; }
[ -s "$PERQUERY_B" ] || { echo "❌ 缺 $PERQUERY_B — 先跑 Job B"; exit 1; }
SIEVE_FILES=()
if [ -s "$PERQUERY_SIEVE_A" ] && [ -s "$PERQUERY_SIEVE_B" ]; then
    SIEVE_FILES=("$PERQUERY_SIEVE_A" "$PERQUERY_SIEVE_B")
    echo "  ✅ SIEVE: jobA + jobB 双文件"
elif [ -s "$PERQUERY_SIEVE_SINGLE" ]; then
    SIEVE_FILES=("$PERQUERY_SIEVE_SINGLE")
    echo "  ✅ SIEVE: 单一文件 ($PERQUERY_SIEVE_SINGLE)"
else
    echo "⚠️  SIEVE per-query 缺失 (无 jobA/B 也无单一文件):"
    echo "    [ ] $PERQUERY_SIEVE_A"
    echo "    [ ] $PERQUERY_SIEVE_B"
    echo "    [ ] $PERQUERY_SIEVE_SINGLE"
    echo "    → ML router 训练只能用 UNG + PF, 没 SIEVE 候选"
    echo "    → 继续 (用部分数据训练)"
fi

MISSING_METRICS=()
for ds in "${V2_DATASETS[@]}"; do
    [ -s "analysis/${ds}_metrics.json" ] || MISSING_METRICS+=("$ds")
done
if [ ${#MISSING_METRICS[@]} -gt 0 ]; then
    echo "⚠️  缺 dataset metrics: ${MISSING_METRICS[@]}"
    echo "   build_ml_validation_data 会用 default 值, 影响特征质量"
    echo "   建议补齐 (在 jobA/jobB 中重跑对应 dataset_metrics)"
fi

echo ""
echo "Step F.1: 合并 perquery CSV (UNG/PF + SIEVE)"
head -1 "$PERQUERY_A" > "$PERQUERY_MERGED"
tail -n +2 "$PERQUERY_A" >> "$PERQUERY_MERGED"
tail -n +2 "$PERQUERY_B" >> "$PERQUERY_MERGED"
for f in "${SIEVE_FILES[@]}"; do
    tail -n +2 "$f" >> "$PERQUERY_MERGED"
done
echo "  ✅ $PERQUERY_MERGED ($(wc -l < $PERQUERY_MERGED) 行)"

echo ""
echo "Step F.2: build_ml_validation_data"
START=$(date +%s)
python analysis/ml_router/build_ml_validation_data.py \
    --data-root ~/benchmarks/datasets/discrete \
    --recall-csv "$PERQUERY_MERGED" \
    --output "$ML_V2_DATA" \
    --datasets "${V2_DATASETS[@]}" 2>&1 | tee "$LOG_DIR/build_ml_v2.log"
echo "  ⏱  $((($(date +%s) - START))) s"
[ -s "$ML_V2_DATA" ] || { echo "❌ $ML_V2_DATA 空"; exit 1; }
echo "  ✅ $ML_V2_DATA ($(wc -l < $ML_V2_DATA) 行)"

echo ""
echo "Step F.3: train_ml_router_v2"
mkdir -p "$OUT_DIR"
START=$(date +%s)
python analysis/ml_router/train_ml_router_v2.py \
    --mode full \
    --val-recall "$PERQUERY_MERGED" \
    --val-features "$ML_V2_DATA" \
    --feature-set minimal \
    --regression \
    --output-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/train.log"
echo "  ⏱  $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "Finalize 完成: $(date)"
echo "============================================================"
echo ""
echo "结果:"
echo "  $PERQUERY_MERGED"
echo "  $ML_V2_DATA"
ls "$OUT_DIR"/full_train_val_results_*.json 2>/dev/null || echo "  (没生成结果 JSON?)"
echo ""
echo "看 ML router recall:"
echo "  cat $OUT_DIR/full_train_val_results_minimal_reg.json | python3 -m json.tool"
