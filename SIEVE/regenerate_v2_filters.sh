#!/bin/bash
# 重生 V2 5 个数据集的 SIEVE base_filters.bin (3 场景 × 5 数据集 = 15 文件)
#
# 用于: rm -rf SIEVE/sieve_labels/ 误伤之后, 重生缺失的 V2 数据.
#
# ⚠️ 关键: V2 数据集 label 是 **0-based** (synth_*/yahoo800k/dbpedia560k),
# 必须用 convert_labels.py (不减 1), 不是 convert_labels_ung.py (UNG 1-based 减 1).
# 老 commit 1af9f2d / 7837fd0 (4/1) 已经验证过这个走法.
# 之前一版用错了 _ung.py 导致 yahoo800k SIEVE 索引 transpose 时 -1 → uint32 OOB segfault.
#
# 用法 (在 cluster 任意节点, conda env 内):
#   cd ~/benchmarks/discrete && bash SIEVE/regenerate_v2_filters.sh
#
# 预估: 5-10 分钟 (主要是 yahoo800k 800K 和 dbpedia560k 560K vectors 的 label 转换)

set -e
DATA_DIR="$HOME/benchmarks/datasets/discrete"
SIEVE_DIR="$HOME/benchmarks/discrete/SIEVE"
LABEL_DIR="$SIEVE_DIR/sieve_labels"

V2_DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)

echo "=== Regenerating SIEVE V2 filters ==="
for ds in "${V2_DATASETS[@]}"; do
    mkdir -p "$LABEL_DIR/$ds"
    echo ""
    echo "--- $ds ---"
    ORIG_BASE="$DATA_DIR/$ds/label_base.txt"

    if [ ! -f "$ORIG_BASE" ]; then
        echo "  ❌ $ORIG_BASE 不存在 → skip $ds"
        continue
    fi

    for sc_pair in "eq:equal" "and:and" "or:or"; do
        sc_short="${sc_pair%%:*}"      # eq / and / or
        sc_data="${sc_pair##*:}"        # equal / and / or
        OUT_BASE="$LABEL_DIR/$ds/original_${sc_short}_base_filters.bin"
        OUT_QUERY="$LABEL_DIR/$ds/original_${sc_short}_query_filters.pkl"
        QUERY_LABELS="$DATA_DIR/$ds/${ds}_query_${sc_data}.txt"

        if [ -f "$OUT_BASE" ] && [ -f "$OUT_QUERY" ]; then
            echo "  [SKIP] ${sc_short}: 已存在"
            continue
        fi
        if [ ! -f "$QUERY_LABELS" ]; then
            echo "  ❌ ${sc_short}: $QUERY_LABELS 不存在"
            continue
        fi
        echo "  [CONVERT] ${sc_short}: $QUERY_LABELS"
        # 0-based — 用 convert_labels.py (NOT convert_labels_ung.py)
        python "$SIEVE_DIR/convert_labels.py" \
            --base_labels "$ORIG_BASE" \
            --query_labels "$QUERY_LABELS" \
            --output_base "$OUT_BASE" \
            --output_query "$OUT_QUERY"
    done
done

echo ""
echo "=== 验证 ==="
for ds in "${V2_DATASETS[@]}"; do
    echo "$ds:"
    for sc in eq and or; do
        f="$LABEL_DIR/$ds/original_${sc}_base_filters.bin"
        if [ -f "$f" ]; then
            echo "  ✅ $(basename $f) ($(du -h $f | cut -f1))"
        else
            echo "  ❌ $(basename $f) 缺"
        fi
    done
done

echo ""
echo "=== Done ==="
echo "下一步: 重跑 SIEVE per-query for yahoo800k + dbpedia560k"
echo "  python analysis/run_sieve_perquery.py --dataset yahoo800k dbpedia560k --output analysis/perquery_recall/perquery_recall_v2_sieve_yahoo_dbpedia.csv"
echo "然后合并到 perquery_recall_v2_sieve.csv"
