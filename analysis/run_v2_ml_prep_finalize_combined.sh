#!/bin/bash
# V2 ML prep — finalize for combined version
#
# 用于 run_v2_ml_prep_combined.sh + 单独 SIEVE 已完成 后的收尾.
#
# 输入 (检查):
#   analysis/ml_router/perquery_recall/perquery_recall_v2.csv          (UNG + PF, combined 输出)
#   analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv    (SIEVE, 另一 weirdo 输出)
#   analysis/{ds}_metrics.json × 5
#
# 步骤:
#   F.1 合并 SIEVE 行进 perquery_recall_v2.csv
#   F.2 build_ml_validation_data
#   F.3 train_ml_router_v2

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
PERQUERY_MERGED="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
PERQUERY_SIEVE="analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv"
ML_V2_DATA="analysis/ml_v2_data.csv"
OUT_DIR="analysis/ml_router/results/v2_on_v2"

LOG_DIR=analysis/v2_prep_logs_finalize_combined_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Finalize Combined — 合并 SIEVE + build + train"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

# 检查前置
[ -s "$PERQUERY_MERGED" ] || { echo "❌ 缺 $PERQUERY_MERGED — 先跑 combined.sh"; exit 1; }
echo "  ✅ $PERQUERY_MERGED ($(wc -l < $PERQUERY_MERGED) 行, UNG+PF)"

if [ -s "$PERQUERY_SIEVE" ]; then
    echo "  ✅ $PERQUERY_SIEVE ($(wc -l < $PERQUERY_SIEVE) 行, SIEVE)"
else
    echo "  ⚠️  缺 $PERQUERY_SIEVE — ML router 训练只能 2 候选 (UNG+PF)"
fi

MISSING=()
for ds in "${V2_DATASETS[@]}"; do
    [ -s "analysis/${ds}_metrics.json" ] || MISSING+=("$ds")
done
[ ${#MISSING[@]} -gt 0 ] && echo "  ⚠️  缺 metrics: ${MISSING[@]}"

echo ""
echo "Step F.1: 合并 SIEVE 进 perquery_recall_v2.csv"
if [ -s "$PERQUERY_SIEVE" ]; then
    # 用临时文件 in-place 合并 (避免破坏原文件)
    TMP="${PERQUERY_MERGED}.tmp"
    cp "$PERQUERY_MERGED" "$TMP"
    # 检查 SIEVE 行是否已合并过 (避免重复 append)
    if grep -q ",SIEVE," "$TMP"; then
        echo "  [skip] $PERQUERY_MERGED 已含 SIEVE 行"
    else
        tail -n +2 "$PERQUERY_SIEVE" >> "$TMP"
        mv "$TMP" "$PERQUERY_MERGED"
        echo "  ✅ 合并完: $PERQUERY_MERGED ($(wc -l < $PERQUERY_MERGED) 行)"
    fi
else
    echo "  [skip] 没有 SIEVE 文件, 跳过合并"
fi

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
echo "全部完成: $(date)"
echo "============================================================"
echo ""
echo "结果:"
ls "$OUT_DIR"/full_train_val_results_*.json 2>/dev/null
echo ""
echo "看 ML router recall:"
echo "  cat $OUT_DIR/full_train_val_results_minimal_reg.json | python3 -m json.tool"
