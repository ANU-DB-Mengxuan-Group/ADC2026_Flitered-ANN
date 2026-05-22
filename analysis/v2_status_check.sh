#!/bin/bash
# 一次性检查 ACORN v2 + Post-filter v2 在 5 个验证数据集上的进度,
# 以及节点是否空闲可以开 Post-filter.
#
# 用法:
#   bash analysis/v2_status_check.sh
#
# 输出三块:
#   [1] ACORN 5 个 V2 数据集的 build/search 完成情况
#   [2] Post-filter 5 个 V2 数据集的 completed/failed 数
#   [3] 节点当前进程 + 内存余量

set +e  # 即便部分命令失败也继续

ACORN_DATA=~/benchmarks/discrete/ACORN/data
FAISS_DIR=~/benchmarks/discrete/faiss
DATASETS=(synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k)

echo "============================================================"
echo "[1] ACORN V2 进度"
echo "============================================================"
for ds in "${DATASETS[@]}"; do
    pf=$ACORN_DATA/param_search_${ds}/progress.json
    if [ ! -f "$pf" ]; then
        echo "  ${ds}: progress.json 不存在"
        continue
    fi
    python3 - <<EOF
import json
with open("$pf") as f: d = json.load(f)
p = d.get("progress", d)
done = sum(1 for v in p.values() if v.get("build_done") is True)
fail = sum(1 for v in p.values() if v.get("build_done") is False)
search = sum(1 for v in p.values() if v.get("search_done"))
print(f"  {'$ds':<18}  build {done:>2} ok / {fail:>2} fail   search_done {search:>3}")
EOF
done

echo ""
echo "============================================================"
echo "[2] Post-filter V2 进度"
echo "============================================================"
for ds in "${DATASETS[@]}"; do
    pf=$FAISS_DIR/progress_postfilter_${ds}.json
    if [ ! -f "$pf" ]; then
        echo "  ${ds}: progress 不存在"
        continue
    fi
    python3 - <<EOF
import json, os
with open("$pf") as f: d = json.load(f)
done = len(d.get("completed", []))
fail = len(d.get("failed", []))
sf = "$FAISS_DIR/results_postfilter/$ds/summary.csv"
rows = 0
if os.path.exists(sf):
    with open(sf) as f: rows = sum(1 for _ in f) - 1
flag = "🔴" if fail > 0 and done == 0 else ("⚠️ " if fail > 0 else "✅" if done > 0 else "  ")
print(f"  {flag} {'$ds':<18}  completed {done:>3}   failed {fail:>3}   summary_rows {rows:>3}")
EOF
done

echo ""
echo "============================================================"
echo "[3] 节点状态"
echo "============================================================"
echo "--- 当前用户的相关进程 (build/search 类) ---"
ps -ef | grep -E "build_acorn|build_HNSW|search_HNSW|auto_param_search|run_postfilter" | grep -v grep | awk '{printf "  PID %s  CPU %s  ELAPSED %s  CMD %.80s\n", $2, $3, $7, $0}'

echo ""
echo "--- 节点 load + 内存 ---"
uptime
echo ""
free -h | head -3

echo ""
echo "============================================================"
echo "判读建议"
echo "============================================================"
cat <<'EOM'
- ACORN 全部 V2 完成: build_done 数 = 27 (除非有 build_failed) 或 21 (M=32,Mb=96 三档已知失败)
- Post-filter 仍 failed: 之前因 MKL 库找不到 (exit 127)。修复方案:
    conda activate benchmark
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python3 analysis/clear_postfilter_failed.py    # 清旧 failed 标记
    screen -S postfilter_v2
    bash analysis/run_postfilter_v2.sh 2>&1 | tee analysis/run_postfilter_v2.log
- 节点上没 ACORN 进程 + 内存余量充足 → 可以直接开 Post-filter
EOM
