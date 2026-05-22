#!/bin/bash
# Post-filter v2 第二波: 并行跑 yahoo800k + dbpedia560k
#
# 用途: 当 round1 (start_postfilter_v2.sh) 还在跑 synth_512d 时,
#       开第二个 SLURM job 同时推进剩下的 yahoo800k + dbpedia560k.
#       SLURM cgroups 强制资源隔离, 两 job 不互相挤资源.
#
# 跳过:
#   - synth_192d (已 completed)
#   - synth_512d (round1 在跑, 不要冲突)
#   - synth_768d_hc (本轮也跳过, 待后续单独处理)
#
# 用法:
#   先在 weirdo (新 SLURM job 内) 起 screen, 然后:
#     bash analysis/start_postfilter_v2_round2.sh
#   预估时间: 2 数据集 × ~15h = 25-30h

set -e

cd ~/benchmarks/discrete

echo "============================================================"
echo "[1/4] 切环境 + 设 LD_LIBRARY_PATH"
echo "============================================================"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
    { echo "❌ 找不到 conda.sh, 请手动 conda activate benchmark 后再 bash 此脚本"; exit 1; }
conda activate benchmark
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "  当前 conda env: $CONDA_DEFAULT_ENV"
echo "  LD_LIBRARY_PATH 前缀: $CONDA_PREFIX/lib"

echo ""
echo "============================================================"
echo "[2/4] 验证 binary 运行时环境"
echo "============================================================"
BIN=~/benchmarks/discrete/faiss/tutorial/cpp/build_HNSW_index_static
SBIN=~/benchmarks/discrete/faiss/tutorial/cpp/search_HNSW_index_static
if [ ! -f "$BIN" ]; then
    echo "❌ build binary 不存在: $BIN"
    exit 1
fi
if [ ! -f "$SBIN" ]; then
    echo "❌ search binary 不存在: $SBIN"
    exit 1
fi

mkl_found=$(ldd "$BIN" | grep -c "libmkl_intel_lp64.*=>.*lib/libmkl_intel_lp64" || true)
if [ "$mkl_found" -eq 0 ]; then
    echo "❌ MKL 库未链接成功! ldd 输出:"
    ldd "$BIN" | grep mkl
    echo "  修复: export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
    exit 1
fi
echo "  ✅ MKL 链接 OK"

LIB_CHECK_OUT=$("$BIN" 2>&1 < /dev/null | head -5)
if echo "$LIB_CHECK_OUT" | grep -qE "GLIBCXX|libmkl|symbol lookup error|undefined symbol|version \`.*' not found|cannot open shared"; then
    echo "❌ binary 启动失败, 运行时库依赖有问题:"
    echo "$LIB_CHECK_OUT" | head -5 | sed 's/^/    /'
    echo "  修复: conda install -y -c conda-forge 'libstdcxx-ng>=14'"
    exit 1
fi
echo "  ✅ Runtime 库 (libstdc++/MKL) 加载 OK"

HOST=$(hostname)
if [[ "$HOST" =~ login ]]; then
    echo ""
    echo "❌ 当前在登录节点 ($HOST), 不应在此跑长任务!"
    echo "   先 salloc + srun 进 db4ai 计算节点再跑."
    exit 1
fi
echo "  ✅ 跑在节点 $HOST (非登录节点)"

echo ""
echo "============================================================"
echo "[3/4] 清空 yahoo800k + dbpedia560k 的旧 build_failed 标记"
echo "============================================================"
# 不调用 clear_postfilter_failed.py (会清全部 4 个 v2 数据集),
# 只清本次要跑的 2 个, 避免和 round1 互相干扰
python3 - <<PYEOF
import json, os
FAISS_DIR = os.path.expanduser("~/benchmarks/discrete/faiss")
for ds in ["yahoo800k", "dbpedia560k"]:
    p = f"{FAISS_DIR}/progress_postfilter_{ds}.json"
    if not os.path.exists(p):
        print(f"  {ds}: 没有 progress 文件 ({p}), skip")
        continue
    with open(p) as f:
        data = json.load(f)
    failed_n = len(data.get("failed", []))
    completed_n = len(data.get("completed", []))
    data["failed"] = []
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {ds}: 清 failed {failed_n} 条, 保留 completed {completed_n} 条")
PYEOF

echo ""
echo "============================================================"
echo "[4/4] 串行跑 yahoo800k → dbpedia560k"
echo "============================================================"
LOGFILE=analysis/run_postfilter_v2_round2_$(date +%Y%m%d_%H%M%S).log
echo "  日志输出到: $LOGFILE"
echo "  开始时间: $(date)"
echo ""
echo "  ⚠️  本脚本只跑 yahoo800k + dbpedia560k"
echo "      (round1 在跑 synth_512d; synth_768d_hc 留待后续)"
echo ""
echo "  Tip: Ctrl+A D 脱离 screen, 后台让它跑"
echo ""
sleep 3

{
    for DS in yahoo800k dbpedia560k; do
        echo ""
        echo "========================================="
        echo "Round2 Post-filter: $DS"
        echo "========================================="
        python faiss/bash/auto_postfilter_hnsw.py "$DS"
    done
    echo ""
    echo "========================================="
    echo "Round2 全部完成: $(date)"
    echo "========================================="
} 2>&1 | tee "$LOGFILE"
