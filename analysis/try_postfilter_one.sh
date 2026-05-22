#!/bin/bash
# Post-filter v2 单配置 smoke test (验证 MKL 修复后能跑通)
#
# 目的: 在开 6-12 小时全 v2 之前, 用最小配置 (yahoo800k, M=32, efc=100)
#      快速验证当前环境能跑通 + binary 不会崩.
#
# 用法: bash analysis/try_postfilter_one.sh [dataset]
#       默认 dataset = yahoo800k; 可换 synth_512d / synth_768d_hc / dbpedia560k
#
# 预估 5-15 分钟 (HNSW build 单配置), 磁盘峰值 < 5GB.

set +e
cd ~/benchmarks/discrete

DATASET=${1:-yahoo800k}
M=32
EFC=100

echo "============================================================"
echo "[1/4] 准备环境"
echo "============================================================"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
    { echo "❌ 找不到 conda.sh"; exit 1; }
conda activate benchmark
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "  conda env: $CONDA_DEFAULT_ENV"
echo "  LD_LIBRARY_PATH 前缀: $CONDA_PREFIX/lib"

echo ""
echo "============================================================"
echo "[2/4] 验证 MKL 链接"
echo "============================================================"
BIN=~/benchmarks/discrete/faiss/tutorial/cpp/build_HNSW_index_static
if ! ldd "$BIN" 2>/dev/null | grep -q "mkl_intel_lp64.*=>.*lib/libmkl_intel"; then
    echo "❌ MKL 仍然找不到, 当前 shell 设 LD_LIBRARY_PATH 没生效"
    ldd "$BIN" 2>/dev/null | grep mkl
    exit 1
fi
echo "  ✅ MKL 链接 OK"

echo ""
echo "============================================================"
echo "[3/4] 跑 ${DATASET} 单配置 (M=$M, efc=$EFC, 3 个场景)"
echo "============================================================"
echo "  开始时间: $(date)"

# 直接 import auto_postfilter_hnsw 内部的逻辑跑 1 个配置
DATA_DIR=~/benchmarks/datasets/discrete
FAISS_DIR=~/benchmarks/discrete/faiss
INDEX_DIR=$FAISS_DIR/data/index_files/hnsw/${DATASET}
# 写到 v2 期望的位置 (而非 _smoketest 子目录), 这样 v2 判跳能直接复用
RESULT_DIR=$FAISS_DIR/results_postfilter/${DATASET}
PROGRESS_FILE=$FAISS_DIR/progress_postfilter_${DATASET}.json
mkdir -p "$INDEX_DIR" "$RESULT_DIR"

FVECS=$DATA_DIR/${DATASET}/${DATASET}_base.fvecs
INDEX_PATH=$INDEX_DIR/M=${M}_efc=${EFC}.json
BUILD_LOG=$INDEX_DIR/M=${M}_efc=${EFC}_build.log

if [ ! -f "$FVECS" ]; then
    echo "❌ fvecs 不存在: $FVECS"
    exit 1
fi

if [ -f "$INDEX_PATH" ]; then
    INDEX_SIZE=$(du -m "$INDEX_PATH" | cut -f1)
    echo "  ⏩ 索引已存在, 跳过 build (${INDEX_SIZE}MB): $INDEX_PATH"
else
    echo "  -> 构建索引..."
    START=$(date +%s)
    "$BIN" "$FVECS" $M $EFC "$FAISS_DIR/data/index_files/hnsw" "$DATASET" \
        > "$BUILD_LOG" 2>&1
    EXIT=$?
    END=$(date +%s)
    BUILD_TIME=$((END - START))

    if [ $EXIT -ne 0 ]; then
        echo "  ❌ build 失败 (exit $EXIT, 耗时 ${BUILD_TIME}s)"
        echo "  build.log 末尾:"
        tail -10 "$BUILD_LOG" | sed 's/^/    /'
        exit $EXIT
    fi

    if [ ! -f "$INDEX_PATH" ]; then
        echo "  ❌ build 命令成功但索引文件没生成: $INDEX_PATH"
        exit 1
    fi
    INDEX_SIZE=$(du -m "$INDEX_PATH" | cut -f1)
    echo "  ✅ build 成功 (${BUILD_TIME}s, ${INDEX_SIZE}MB)"
fi

echo ""
echo "  -> 跑 3 个场景搜索..."
SEARCH_BIN=$FAISS_DIR/tutorial/cpp/search_HNSW_index_static
N=$(python3 -c "import struct,os; f=open('${DATA_DIR}/${DATASET}/${DATASET}_base.fvecs','rb'); d=struct.unpack('I',f.read(4))[0]; print(os.path.getsize('${DATA_DIR}/${DATASET}/${DATASET}_base.fvecs')//(4+d*4))")

SUCCESS_SCENARIOS=()
for SC in equal or and; do
    OUT_CSV=$RESULT_DIR/${SC}/M=${M}_efc=${EFC}_result.csv
    SEARCH_LOG=$RESULT_DIR/${SC}/M=${M}_efc=${EFC}_search.log
    mkdir -p "$RESULT_DIR/${SC}"
    START=$(date +%s)
    # 参数顺序对齐 auto_postfilter_hnsw.py:search_index (13 个位置参数):
    # dataset M efc index_root scenario result_dir base_fvecs label_base query_fvecs query_label gt_path k N
    "$SEARCH_BIN" "$DATASET" $M $EFC "$FAISS_DIR/data/index_files/hnsw" "$SC" \
        "$RESULT_DIR/${SC}/" \
        "$DATA_DIR/${DATASET}/${DATASET}_base.fvecs" \
        "$DATA_DIR/${DATASET}/label_base.txt" \
        "$DATA_DIR/${DATASET}/${DATASET}_query_${SC}.fvecs" \
        "$DATA_DIR/${DATASET}/${DATASET}_query_${SC}.txt" \
        "$DATA_DIR/${DATASET}/${DATASET}_gt_${SC}.txt" \
        10 "$N" \
        > "$SEARCH_LOG" 2>&1
    EXIT=$?
    END=$(date +%s)
    if [ $EXIT -ne 0 ]; then
        echo "    ❌ ${SC} 搜索失败 (exit $EXIT)"
        tail -5 "$SEARCH_LOG" | sed 's/^/      /'
    else
        # 从 search log 抓 recall + qps (auto_postfilter_hnsw 的格式)
        RECALL=$(grep -oP 'recall.*[0-9.]+' "$SEARCH_LOG" 2>/dev/null | tail -1)
        echo "    ✅ ${SC}: $((END - START))s ${RECALL:-(see log)}"
        SUCCESS_SCENARIOS+=("$SC")
    fi
done

# 把成功的配置登记到 progress JSON, 让全量 v2 能跳过这个 build/search
if [ ${#SUCCESS_SCENARIOS[@]} -gt 0 ]; then
    echo ""
    echo "  -> 登记 progress 让 v2 跳过 (${#SUCCESS_SCENARIOS[@]} 场景成功)..."
    python3 - <<PYEOF
import json, os
p = "$PROGRESS_FILE"
data = json.load(open(p)) if os.path.exists(p) else {}
completed = data.setdefault("completed", [])

keys = ["${DATASET}_M=${M}_efc=${EFC}_build"]
for sc in "${SUCCESS_SCENARIOS[@]}".split():
    keys.append(f"${DATASET}_{sc}_M=${M}_efc=${EFC}")

added = []
for k in keys:
    if k not in completed:
        completed.append(k)
        added.append(k)

os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
json.dump(data, open(p, "w"), indent=2)
print(f"    progress 文件: {p}")
print(f"    新登记 {len(added)} 条:")
for k in added:
    print(f"      - {k}")
PYEOF
fi

echo ""
echo "============================================================"
echo "[4/4] 保留索引 + 登记 progress, 让 v2 直接跳过"
echo "============================================================"
echo "  索引保留在 $INDEX_PATH"
echo "  progress 已登记: $PROGRESS_FILE"
echo "  → 之后跑 start_postfilter_v2.sh 会自动 skip 这个配置, 不会重 build"
echo "  如需手动删除索引: rm -f \"$INDEX_PATH\""
echo ""
echo "✅ Smoke test 完成. 如果三个场景都成功就说明环境 OK, 可以开全 v2:"
echo "   screen -S postfilter_v2"
echo "   bash analysis/start_postfilter_v2.sh"
echo ""
echo "结束时间: $(date)"
