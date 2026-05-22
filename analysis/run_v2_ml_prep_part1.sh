#!/bin/bash
# V2 ML router prep — Part 1: per-query recall + dataset metrics (慢的部分串行)
#
# 跑完这个再跑 part2 (assembly + train).
#
# 用法 (在 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash analysis/run_v2_ml_prep_part1.sh
#
# 预估: 30 分钟 - 4 小时 (主要看 dataset_metrics 速度)
#   - Step 1.1 per-query recall:    5-10 分钟
#   - Step 1.2 dataset metrics × 5: 串行 5-30 分钟 each = 25-150 分钟
#
# 输出:
#   analysis/ml_router/perquery_recall/perquery_recall_v2.csv
#   analysis/{ds}_metrics.json × 5

set -e
cd ~/benchmarks/discrete

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)

echo "============================================================"
echo "Part 1: V2 ML router prep (慢部分)"
echo "开始: $(date)"
echo "============================================================"

# 简单环境检查
if [ -z "${CONDA_DEFAULT_ENV}" ] || [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then
    echo "❌ conda env 不是 benchmark, 请先 conda activate benchmark"
    exit 1
fi
HOST=$(hostname)
if [[ "$HOST" =~ login ]]; then
    echo "❌ 当前在登录节点 ($HOST), 请先 srun 进 weirdo"
    exit 1
fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "============================================================"
echo "Step 1.1: V2 per-query recall 提取 (UNG + Post-filter)"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py
END=$(date +%s)
echo "  ⏱  Step 1.1 用时: $((END - START))s"

OUTPUT="analysis/ml_router/perquery_recall/perquery_recall_v2.csv"
if [ ! -s "$OUTPUT" ]; then
    echo "❌ 输出空: $OUTPUT"
    exit 1
fi
echo "  ✅ 输出: $OUTPUT ($(wc -l < $OUTPUT) 行)"

echo ""
echo "============================================================"
echo "Step 1.2: V2 dataset metrics 计算 (5 数据集串行)"
echo "============================================================"
for ds in "${V2_DATASETS[@]}"; do
    OUT="analysis/${ds}_metrics.json"
    if [ -s "$OUT" ]; then
        echo "  [skip] $ds: metrics 已有 ($OUT)"
        continue
    fi
    echo ""
    echo "  --- $ds ---"
    START=$(date +%s)
    python tools/dataset_metrics.py \
        --data-root ~/benchmarks/datasets/discrete \
        --dataset $ds || echo "  ⚠️  $ds dataset_metrics 失败, 继续"
    END=$(date +%s)
    echo "  ⏱  $ds 用时: $((END - START))s"
    if [ -s "$OUT" ]; then
        echo "  ✅ $ds metrics 已生成"
    else
        echo "  ❌ $ds metrics 没生成"
    fi
done

echo ""
echo "============================================================"
echo "Part 1 完成. 结束: $(date)"
echo "============================================================"
echo ""
echo "下一步: bash analysis/run_v2_ml_prep_part2.sh"
