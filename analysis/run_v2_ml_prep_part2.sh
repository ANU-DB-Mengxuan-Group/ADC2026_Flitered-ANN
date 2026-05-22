#!/bin/bash
# V2 ML router prep — Part 2: build validation data + train ML router
#
# 跑前必须先跑过 part1 (生成 per-query recall + dataset metrics).
#
# 用法 (在 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash analysis/run_v2_ml_prep_part2.sh
#
# 预估: 5-15 分钟
#   - Step 2.1 build_ml_validation_data: 1-5 分钟
#   - Step 2.2 train_ml_router_v2:       1-5 分钟
#
# 输出:
#   analysis/ml_v2_data.csv
#   analysis/ml_router/results/v2_on_v2/full_train_val_results_*.json

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
PERQUERY_CSV="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
ML_V2_DATA="analysis/ml_v2_data.csv"
OUT_DIR="analysis/ml_router/results/v2_on_v2"

echo "============================================================"
echo "Part 2: V2 ML router prep (训练部分)"
echo "开始: $(date)"
echo "============================================================"

# 检查 part1 输出
if [ ! -s "$PERQUERY_CSV" ]; then
    echo "❌ 缺 $PERQUERY_CSV - 先跑 part1"
    exit 1
fi
echo "  ✅ per-query recall: $PERQUERY_CSV ($(wc -l < $PERQUERY_CSV) 行)"

MISSING=()
for ds in "${V2_DATASETS[@]}"; do
    if [ ! -s "analysis/${ds}_metrics.json" ]; then
        MISSING+=("$ds")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "⚠️  缺 dataset metrics: ${MISSING[@]}"
    echo "   build_ml_validation_data 会用 default 值, 可能影响特征质量"
    echo "   建议先跑 part1 补齐"
fi

echo ""
echo "============================================================"
echo "Step 2.1: build_ml_validation_data 组装"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/build_ml_validation_data.py \
    --data-root ~/benchmarks/datasets/discrete \
    --recall-csv "$PERQUERY_CSV" \
    --output "$ML_V2_DATA" \
    --datasets "${V2_DATASETS[@]}"
END=$(date +%s)
echo "  ⏱  Step 2.1 用时: $((END - START))s"

if [ ! -s "$ML_V2_DATA" ]; then
    echo "❌ 输出空: $ML_V2_DATA"
    exit 1
fi
echo "  ✅ 输出: $ML_V2_DATA ($(wc -l < $ML_V2_DATA) 行)"

echo ""
echo "============================================================"
echo "Step 2.2: train_ml_router_v2 (full mode + V2 validation)"
echo "============================================================"
mkdir -p "$OUT_DIR"
START=$(date +%s)
python analysis/ml_router/train_ml_router_v2.py \
    --mode full \
    --val-recall "$PERQUERY_CSV" \
    --val-features "$ML_V2_DATA" \
    --feature-set minimal \
    --regression \
    --output-dir "$OUT_DIR"
END=$(date +%s)
echo "  ⏱  Step 2.2 用时: $((END - START))s"

echo ""
echo "============================================================"
echo "Part 2 完成. 结束: $(date)"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  - $ML_V2_DATA"
echo "  - $OUT_DIR/full_train_val_results_*.json"
echo ""
echo "看结果:"
echo "  cat $OUT_DIR/full_train_val_results_minimal_reg.json | python3 -m json.tool"
