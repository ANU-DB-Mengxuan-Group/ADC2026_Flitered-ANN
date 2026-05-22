#!/bin/bash
# V2 PF rebuild 完成后的 ML 收尾脚本
#
# 用于: 3 个 PF rebuild (synth_192d/synth_512d/dbpedia560k) + PF rescue
#       (synth_768d_hc/yahoo800k) + SIEVE per-query 全部跑完 之后.
#
# 步骤:
#   F.0 prereq check (.bin × 5 datasets, dataset_metrics × 5, SIEVE perquery)
#   F.1 全局 re-extract perquery_recall_v2.csv (UNG + PF for 全 5 datasets)
#   F.2 合并 SIEVE 进 perquery_recall_v2.csv
#   F.3 build_ml_validation_data → ml_v2_data.csv
#   F.4 train_ml_router_v2 (mode=full, V1 train + V2 val)
#
# 用法 (在 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash analysis/run_v2_pf_finalize_ml.sh
#
# 预估: 15-30 分钟 (主要是 build_ml_validation_data 加载 label_base.txt × 5)

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)
PERQUERY_V2="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
PERQUERY_SIEVE="analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv"
ML_V2_DATA="analysis/ml_v2_data.csv"
OUT_DIR="analysis/ml_router/results/v2_on_v2"
LOG_DIR="analysis/v2_pf_finalize_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "============================================================"
echo "V2 PF rebuild 收尾 — 全局 re-extract + ML training"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi

echo ""
echo "============================================================"
echo "Step F.0: 前置检查 (.bin × 5 datasets, dataset_metrics × 5, SIEVE)"
echo "============================================================"
ALL_OK=1

# (a) .bin 文件: 全 5 数据集 × 3 场景都有
echo "  --- .bin files (per-query top-k) ---"
for ds in "${V2_DATASETS[@]}"; do
    for sc in and or equal; do
        n=$(find faiss/results_postfilter/$ds/$sc -name "M=64_efc=400_ef=*_idx_uint32.bin" 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then
            echo "    ❌ $ds/$sc: 0 bin files"; ALL_OK=0
        else
            echo "    ✅ $ds/$sc: $n bin files"
        fi
    done
done

# (b) dataset_metrics × 5
echo "  --- dataset_metrics (features) ---"
for ds in "${V2_DATASETS[@]}"; do
    if [ -s "analysis/${ds}_metrics.json" ]; then
        echo "    ✅ analysis/${ds}_metrics.json"
    else
        echo "    ❌ analysis/${ds}_metrics.json 缺"; ALL_OK=0
    fi
done

# (c) SIEVE per-query
echo "  --- SIEVE per-query ---"
if [ -s "$PERQUERY_SIEVE" ]; then
    echo "    ✅ $PERQUERY_SIEVE ($(wc -l < $PERQUERY_SIEVE) 行)"
else
    echo "    ⚠️  $PERQUERY_SIEVE 缺 — ML router 只能 2 候选 (UNG+PF)"
fi

# (d) V1 训练数据 (train_ml_router_v2 默认从这俩读)
echo "  --- V1 train data ---"
for f in analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv analysis/ml_training_data.csv; do
    if [ -s "$f" ]; then
        echo "    ✅ $f"
    else
        echo "    ❌ $f 缺 (V1 训练数据)"; ALL_OK=0
    fi
done

if [ "$ALL_OK" -ne 1 ]; then
    echo ""
    echo "❌ 前置检查失败. 修补后再跑."
    exit 1
fi
echo "  ✅ 前置全 OK"

echo ""
echo "============================================================"
echo "Step F.1: 全局 re-extract perquery_recall_v2.csv (UNG + PF × 5)"
echo "============================================================"
# 备份旧文件 (如果存在)
if [ -s "$PERQUERY_V2" ]; then
    cp "$PERQUERY_V2" "${PERQUERY_V2}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "  备份: ${PERQUERY_V2}.bak.*"
fi
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "${V2_DATASETS[@]}" \
    --output "$PERQUERY_V2" 2>&1 | tee "$LOG_DIR/extract.log"
echo "  ⏱  $((($(date +%s) - START))) s"
[ -s "$PERQUERY_V2" ] || { echo "❌ $PERQUERY_V2 空"; exit 1; }
echo "  ✅ $PERQUERY_V2 ($(wc -l < $PERQUERY_V2) 行)"
# 校验: 应含 UNG + Post-filter, 且 dataset 覆盖全 5
echo "  方法分布:"
awk -F, 'NR>1 {print $4}' "$PERQUERY_V2" | sort | uniq -c
echo "  dataset 分布:"
awk -F, 'NR>1 {print $2}' "$PERQUERY_V2" | sort | uniq -c

echo ""
echo "============================================================"
echo "Step F.2: 合并 SIEVE 行进 perquery_recall_v2.csv"
echo "============================================================"
if [ -s "$PERQUERY_SIEVE" ]; then
    if grep -q ",SIEVE," "$PERQUERY_V2"; then
        echo "  [skip] $PERQUERY_V2 已含 SIEVE 行 (不应该, 因为刚 re-extract 的)"
    else
        tail -n +2 "$PERQUERY_SIEVE" >> "$PERQUERY_V2"
        echo "  ✅ 合并完: $PERQUERY_V2 ($(wc -l < $PERQUERY_V2) 行)"
        echo "  方法分布 (合并后):"
        awk -F, 'NR>1 {print $4}' "$PERQUERY_V2" | sort | uniq -c
    fi
else
    echo "  [skip] 没 SIEVE per-query, ML 只能 2 候选"
fi

echo ""
echo "============================================================"
echo "Step F.3: build_ml_validation_data → ml_v2_data.csv"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/build_ml_validation_data.py \
    --data-root ~/benchmarks/datasets/discrete \
    --recall-csv "$PERQUERY_V2" \
    --output "$ML_V2_DATA" \
    --datasets "${V2_DATASETS[@]}" 2>&1 | tee "$LOG_DIR/build_ml_v2.log"
echo "  ⏱  $((($(date +%s) - START))) s"
[ -s "$ML_V2_DATA" ] || { echo "❌ $ML_V2_DATA 空"; exit 1; }
echo "  ✅ $ML_V2_DATA ($(wc -l < $ML_V2_DATA) 行)"

echo ""
echo "============================================================"
echo "Step F.4: train_ml_router_v2 (V1 train + V2 val)"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/train_ml_router_v2.py \
    --mode full \
    --val-recall "$PERQUERY_V2" \
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
echo "  cat $OUT_DIR/full_train_val_results_minimal_reg.json | python3 -m json.tool | head -50"
