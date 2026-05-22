#!/bin/bash
# Post-filter v2 第三波: synth_512d (只补 search) + synth_768d_hc (完整 build+search)
#
# 用途: round1 在 synth_512d M=64 efc=400 build 成功后被 SLURM time-limit kill,
#       3 个 search 都没跑. 这个脚本补 synth_512d 的 search,
#       并接着跑 synth_768d_hc (round1/round2 都没碰过).
#
# 跳过:
#   - synth_192d (已 completed)
#   - yahoo800k (round2 在跑)
#   - dbpedia560k (round2 在跑)
#
# 用法:
#   先 salloc + srun 进 weirdo, 然后:
#     screen -S postfilter_round3
#     bash analysis/start_postfilter_v2_round3.sh
#     Ctrl+A D 脱离
#
# 预估时间:
#   synth_512d: 仅 3 个 search (build 已存在), ~30-90 分钟
#   synth_768d_hc: 完整 build (~18-22h) + 3 search (~30-90 min)
#   合计 ~20-24h, 建议 salloc --time=30:00:00 留余量

set -e

cd ~/benchmarks/discrete

echo "============================================================"
echo "[1/4] 切环境 + 设 LD_LIBRARY_PATH"
echo "============================================================"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
    { echo "❌ 找不到 conda.sh"; exit 1; }
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
echo "[3/4] 清空 synth_512d + synth_768d_hc 的旧 build_failed 标记"
echo "============================================================"
# 只清这两个数据集的 failed (避免影响 round2 的 yahoo800k / dbpedia560k)
python3 - <<PYEOF
import json, os
FAISS_DIR = os.path.expanduser("~/benchmarks/discrete/faiss")
for ds in ["synth_512d", "synth_768d_hc"]:
    p = f"{FAISS_DIR}/progress_postfilter_{ds}.json"
    if not os.path.exists(p):
        print(f"  {ds}: 没有 progress 文件, skip")
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
echo "[4/4] 串行跑 synth_512d → synth_768d_hc"
echo "============================================================"
LOGFILE=analysis/run_postfilter_v2_round3_$(date +%Y%m%d_%H%M%S).log
echo "  日志输出到: $LOGFILE"
echo "  开始时间: $(date)"
echo ""
echo "  ⚠️  本脚本只跑 synth_512d (补 search) + synth_768d_hc (full build+search)"
echo "      round2 在跑 yahoo800k → dbpedia560k, 不会冲突"
echo ""
echo "  Tip: Ctrl+A D 脱离 screen, 后台让它跑"
echo ""
sleep 3

{
    for DS in synth_512d synth_768d_hc; do
        echo ""
        echo "========================================="
        echo "Round3 Post-filter: $DS"
        echo "========================================="
        python faiss/bash/auto_postfilter_hnsw.py "$DS"
    done
    echo ""
    echo "========================================="
    echo "Round3 全部完成: $(date)"
    echo "========================================="
} 2>&1 | tee "$LOGFILE"
