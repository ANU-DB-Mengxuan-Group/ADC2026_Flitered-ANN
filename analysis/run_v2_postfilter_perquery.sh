#!/bin/bash
# V2 Post-filter per-query 补救脚本 (用 /tmp 备份的 2 个索引)
#
# 步骤:
#   1. 还原 /tmp 备份的 synth_768d_hc + yahoo800k 索引
#   2. 重编 search_HNSW_index_static (旧 binary 没 .bin 输出)
#   3. 清 progress 的 failed 标记 + 把 search 标记从 completed 移除 (强制重跑 search)
#   4. 跑 auto_postfilter_hnsw.py — 自动 skip build (索引存在), 重跑 search 出 .bin
#   5. 跑 extract_v2_perquery_recall.py 加 PF 数据
#
# 用法 (新 weirdo, conda env, screen 内):
#   cd ~/benchmarks/discrete && git pull
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   screen -S v2_pf_perquery
#   bash analysis/run_v2_postfilter_perquery.sh
#   Ctrl+A D
#
# 预估: 1-3 小时
#   - 重编 binary: 5-30 min
#   - 还原索引: 1 min
#   - search × 6 (2 数据集 × 3 场景): 30 min - 2h
#   - extract per-query: 5 min

set -e
cd ~/benchmarks/discrete

DATASETS_PF=(synth_768d_hc yahoo800k)
LOG_DIR=analysis/v2_pf_perquery_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "V2 Post-filter per-query 补救"
echo "处理: ${DATASETS_PF[@]}"
echo "开始: $(date) | log: $LOG_DIR"
echo "============================================================"

# 环境检查
if [ "${CONDA_DEFAULT_ENV}" != "benchmark" ]; then echo "❌ conda env 不是 benchmark"; exit 1; fi
HOST=$(hostname); if [[ "$HOST" =~ login ]]; then echo "❌ 在登录节点 ($HOST)"; exit 1; fi
echo "  ✅ env=benchmark, host=$HOST"

echo ""
echo "============================================================"
echo "Step 1: 还原 /tmp 备份索引到工作目录"
echo "============================================================"
for ds in "${DATASETS_PF[@]}"; do
    SRC="/tmp/${ds}_M=64_efc=400.json.backup"
    DST_DIR="$HOME/benchmarks/discrete/faiss/data/index_files/hnsw/$ds"
    DST="$DST_DIR/M=64_efc=400.json"
    mkdir -p "$DST_DIR"
    # 先检查 DST 是否已存在 (NFS 共享, 也许从登录节点已经 cp 过来了)
    if [ -s "$DST" ]; then
        echo "  [skip] $DST 已存在 ($(du -h "$DST" | cut -f1))"
        continue
    fi
    # DST 不存在, 试 /tmp 备份 (注意: weirdo 的 /tmp 跟登录节点不共享!)
    if [ ! -s "$SRC" ]; then
        echo "  ❌ /tmp 备份缺: $SRC"
        echo "     说明: weirdo 的 /tmp 跟登录节点 /tmp 不共享."
        echo "     从登录节点先 cp 到 NFS 共享目录:"
        echo "       ssh anu 'cp /tmp/${ds}_M=64_efc=400.json.backup ~/benchmarks/discrete/faiss/data/index_files/hnsw/${ds}/M=64_efc=400.json'"
        exit 1
    fi
    echo "  cp $SRC → $DST ..."
    cp "$SRC" "$DST"
    echo "  ✅ $(du -h "$DST" | cut -f1)"
done

echo ""
echo "============================================================"
echo "Step 2: 升级 search_HNSW_index_static binary (含 .bin 输出)"
echo "============================================================"
BIN_PATH=~/benchmarks/discrete/faiss/tutorial/cpp/search_HNSW_index_static
NEW_BIN=~/benchmarks/discrete/faiss/build/tutorial/cpp/search_HNSW_index

if strings "$BIN_PATH" 2>/dev/null | grep -q "idx_uint32"; then
    echo "  [skip] $BIN_PATH 已含 idx_uint32 输出"
else
    if [ ! -x "$NEW_BIN" ]; then
        echo "  ❌ build dir 没编好的 binary: $NEW_BIN"
        echo "     需要先在 build/ 跑: cmake --build . --target search_HNSW_index"
        exit 1
    fi
    if ! strings "$NEW_BIN" 2>/dev/null | grep -q "idx_uint32"; then
        echo "  ❌ build dir 的 binary 也没 idx_uint32, 源码可能没含此代码"
        exit 1
    fi
    cp "$NEW_BIN" "$BIN_PATH"
    echo "  ✅ cp $NEW_BIN → $BIN_PATH"
    echo "  $(stat -c '%y %s bytes' "$BIN_PATH")"
fi

echo ""
echo "============================================================"
echo "Step 3: 清 progress 的 failed 标记 + 强制重跑 search"
echo "============================================================"
python3 - <<PYEOF
import json, os
for ds in ["synth_768d_hc", "yahoo800k"]:
    p = os.path.expanduser(f"~/benchmarks/discrete/faiss/progress_postfilter_{ds}.json")
    if not os.path.exists(p):
        print(f"  [skip] {ds}: 没 progress 文件")
        continue
    d = json.load(open(p))
    completed = d.get("completed", [])
    failed = d.get("failed", [])

    # 清 failed
    before_failed = len(failed)
    d["failed"] = []

    # 把 search 任务从 completed 移除, 强制重跑 (但保留 build 标记!)
    keys_to_remove = [
        f"{ds}_and_M=64_efc=400",
        f"{ds}_or_M=64_efc=400",
        f"{ds}_equal_M=64_efc=400",
    ]
    before_completed = len(completed)
    d["completed"] = [x for x in completed if x not in keys_to_remove]

    json.dump(d, open(p, "w"), indent=2)
    print(f"  {ds}:")
    print(f"    failed: {before_failed} → 0")
    print(f"    completed: {before_completed} → {len(d['completed'])} (去掉 search 强制重跑)")
PYEOF

echo ""
echo "============================================================"
echo "Step 4: 跑 auto_postfilter_hnsw.py × 2 数据集 (skip build, 重跑 search)"
echo "============================================================"
for ds in "${DATASETS_PF[@]}"; do
    echo ""
    echo "  --- $ds ---"
    START=$(date +%s)
    python faiss/bash/auto_postfilter_hnsw.py "$ds" 2>&1 | tee "$LOG_DIR/postfilter_${ds}.log"
    echo "  ⏱  $ds: $((($(date +%s) - START))) s"
done

echo ""
echo "============================================================"
echo "Step 5: 验证 .bin 文件产出"
echo "============================================================"
for ds in "${DATASETS_PF[@]}"; do
    echo "--- $ds ---"
    for sc in and or equal; do
        n=$(find ~/benchmarks/discrete/faiss/results_postfilter/$ds/$sc -name "M=64_efc=400_ef=*_idx_uint32.bin" 2>/dev/null | wc -l)
        echo "  $sc: $n bin files"
    done
done

echo ""
echo "============================================================"
echo "Step 6: extract_v2_perquery_recall (UNG + PF + 等)"
echo "============================================================"
START=$(date +%s)
python analysis/ml_router/extract_v2_perquery_recall.py \
    --dataset "${DATASETS_PF[@]}" \
    --output analysis/ml_router/perquery_recall/perquery_recall_v2_pf_perquery.csv 2>&1 | tee "$LOG_DIR/extract.log"
echo "  ⏱  extract: $((($(date +%s) - START))) s"

echo ""
echo "============================================================"
echo "完成: $(date)"
echo "============================================================"
echo ""
echo "输出:"
echo "  analysis/ml_router/perquery_recall/perquery_recall_v2_pf_perquery.csv"
echo "    (UNG + PF for synth_768d_hc + yahoo800k)"
echo ""
echo "下一步: 跟之前的 perquery_recall_v2.csv (UNG-only) 和 SIEVE 文件合并"
echo "         然后跑 finalize"
