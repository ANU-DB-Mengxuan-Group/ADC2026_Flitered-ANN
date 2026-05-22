#!/bin/bash
# V2 Post-filter 单数据集完整 rebuild (build + search + extract per-query)
#
# 用于 3 个索引早删的 V2 数据集 (synth_192d, synth_512d, dbpedia560k).
# 每个数据集一个 weirdo 跑, 3 weirdo 并行 ≈ max 20h.
#
# 用法 (在 weirdo, conda env, screen 内):
#   bash analysis/run_v2_pf_full_rebuild.sh <dataset>
#   例:
#     bash analysis/run_v2_pf_full_rebuild.sh synth_192d
#     bash analysis/run_v2_pf_full_rebuild.sh synth_512d
#     bash analysis/run_v2_pf_full_rebuild.sh dbpedia560k
#
# 步骤:
#   1. 验证 binary 含 .bin 输出
#   2. 清 progress 的 build + search 标记 (强制重跑)
#   3. python auto_postfilter_hnsw.py — build + search 3 场景 (产出 .bin)
#   4. extract per-query recall
#
# 预估 (per dataset):
#   synth_192d:   ~5-8h  (build ~3h + search ~1-2h + extract)
#   synth_512d:   ~13-18h (build ~10h + search ~2-3h + extract)
#   dbpedia560k:  ~18-22h (build ~15h + search ~2-3h + extract)

set -e
cd ~/benchmarks/discrete

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    echo "  dataset: synth_192d / synth_512d / dbpedia560k"
    exit 1
fi
DS="$1"

LOG_DIR="analysis/v2_pf_rebuild_logs_${DS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "V2 Post-filter 完整 rebuild: $DS"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi
HOST=$(hostname); if [[ "$HOST" =~ login ]]; then echo "❌ 在登录节点 ($HOST)"; exit 1; fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "============================================================"
echo "Step 1: 验证 search binary 含 .bin 输出"
echo "============================================================"
BIN_PATH=~/benchmarks/discrete/faiss/tutorial/cpp/search_HNSW_index_static
if ! strings "$BIN_PATH" 2>/dev/null | grep -q "idx_uint32"; then
    NEW_BIN=~/benchmarks/discrete/faiss/build/tutorial/cpp/search_HNSW_index
    if [ -x "$NEW_BIN" ] && strings "$NEW_BIN" 2>/dev/null | grep -q "idx_uint32"; then
        cp "$NEW_BIN" "$BIN_PATH"
        echo "  ✅ 升级 search binary (从 build/ cp 过来)"
    else
        echo "  ❌ 找不到含 idx_uint32 的 binary"
        exit 1
    fi
else
    echo "  ✅ binary 已含 idx_uint32"
fi

echo ""
echo "============================================================"
echo "Step 2: 清 progress 的 build + search 标记 (强制重跑)"
echo "============================================================"
python3 - <<PYEOF
import json, os
ds = "$DS"
p = os.path.expanduser(f"~/benchmarks/discrete/faiss/progress_postfilter_{ds}.json")
if not os.path.exists(p):
    print(f"  [info] {p} 不存在, 创建空 progress")
    json.dump({"completed": [], "failed": []}, open(p, "w"), indent=2)
    raise SystemExit
d = json.load(open(p))
keys_to_remove = {
    f"{ds}_M=64_efc=400_build",
    f"{ds}_and_M=64_efc=400",
    f"{ds}_or_M=64_efc=400",
    f"{ds}_equal_M=64_efc=400",
}
before_c = len(d.get("completed", []))
before_f = len(d.get("failed", []))
d["completed"] = [x for x in d.get("completed", []) if x not in keys_to_remove]
d["failed"] = [x for x in d.get("failed", []) if x not in keys_to_remove]
json.dump(d, open(p, "w"), indent=2)
print(f"  completed: {before_c} → {len(d['completed'])}")
print(f"  failed: {before_f} → {len(d['failed'])}")
PYEOF

echo ""
echo "============================================================"
echo "Step 3: auto_postfilter_hnsw build + search ($DS)"
echo "============================================================"
START=$(date +%s)
python faiss/bash/auto_postfilter_hnsw.py "$DS" 2>&1 | tee "$LOG_DIR/postfilter.log"
echo "  ⏱  total: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "Step 4: 验证 .bin 文件产出"
echo "============================================================"
for sc in and or equal; do
    n=$(find ~/benchmarks/discrete/faiss/results_postfilter/$DS/$sc -name "M=64_efc=400_ef=*_idx_uint32.bin" 2>/dev/null | wc -l)
    echo "  $sc: $n bin files"
done

echo ""
echo "============================================================"
echo "Step 5: extract per-query recall ($DS)"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "$DS" \
    --output "analysis/ml_router/perquery_recall/perquery_recall_v2_pf_${DS}.csv" 2>&1 | tee "$LOG_DIR/extract.log"
echo "  ⏱  extract: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "完成: $(date)"
echo "============================================================"
echo ""
echo "输出:"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_pf_${DS}.csv"
