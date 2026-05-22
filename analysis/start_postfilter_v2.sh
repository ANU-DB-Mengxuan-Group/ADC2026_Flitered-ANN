#!/bin/bash
# 一键准备并启动 Post-filter v2 实验
#
# 用法 (推荐先开 screen 再跑):
#   screen -S postfilter_v2
#   bash analysis/start_postfilter_v2.sh
#   # Ctrl+A D 脱离 screen
#
# 脚本干 4 件事:
#   1. 切到 benchmark conda 环境 + 设 LD_LIBRARY_PATH (修复之前的 MKL 找不到 bug)
#   2. 验证 MKL 可加载 (ldd 检查)
#   3. 清掉之前 27 个 build_failed 标记 (旧 MKL 问题导致的, 修好后要重跑)
#   4. 启动 run_postfilter_v2.sh, 输出 tee 到 log 文件
#
# 预估时间: 4 数据集 × 9 配置 × 10-30 min ≈ 6-12 小时
# 磁盘峰值: 每次只保留 1 个索引文件 (build → 搜索3场景 → 删), 不会爆

set -e

cd ~/benchmarks/discrete

echo "============================================================"
echo "[1/4] 切环境 + 设 LD_LIBRARY_PATH"
echo "============================================================"
# Conda 激活 (脚本里需要 source conda.sh)
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

# 2.1 静态依赖检查 (ldd)
mkl_found=$(ldd "$BIN" | grep -c "libmkl_intel_lp64.*=>.*lib/libmkl_intel_lp64" || true)
if [ "$mkl_found" -eq 0 ]; then
    echo "❌ MKL 库未链接成功! ldd 输出:"
    ldd "$BIN" | grep mkl
    echo ""
    echo "修复: export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
    exit 1
fi
echo "  ✅ MKL 链接 OK"

# 2.2 实际启动 binary, 抓任何 runtime 库依赖错误 (GLIBCXX 缺失等)
# binary 没参数会打印 Usage 并退出非 0; 我们要的是没出现 'version not found' 之类
LIB_CHECK_OUT=$("$BIN" 2>&1 < /dev/null | head -5)
if echo "$LIB_CHECK_OUT" | grep -qE "GLIBCXX|libmkl|symbol lookup error|undefined symbol|version \`.*' not found|cannot open shared"; then
    echo "❌ binary 启动失败, 运行时库依赖有问题:"
    echo "$LIB_CHECK_OUT" | head -5 | sed 's/^/    /'
    echo ""
    echo "常见修复:"
    echo "  • libstdc++ 太老 (缺 GLIBCXX_3.4.30+):"
    echo "      conda install -y -c conda-forge 'libstdcxx-ng>=14'"
    echo "  • MKL 找不到: 确认 LD_LIBRARY_PATH 包含 \$CONDA_PREFIX/lib"
    echo "  • 节点系统库问题: 先 salloc + srun 进 db4ai 计算节点再跑"
    exit 1
fi
echo "  ✅ Runtime 库 (libstdc++/MKL) 加载 OK"

# 2.3 验证 conda env 的 libstdc++ 包含必要的 GLIBCXX 符号
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    if ! strings "$CONDA_PREFIX/lib/libstdc++.so.6" 2>/dev/null | grep -qE '^GLIBCXX_3\.4\.32$'; then
        echo "⚠️  警告: \$CONDA_PREFIX/lib/libstdc++.so.6 缺 GLIBCXX_3.4.32"
        echo "     binary 可能 fallback 到系统库或 base conda env. 如果后续 build 报错, 先升级:"
        echo "     conda install -y -c conda-forge 'libstdcxx-ng>=14'"
    else
        echo "  ✅ libstdc++ 含 GLIBCXX_3.4.32"
    fi
fi

# 2.4 验证 SLURM/节点身份 (避免误跑在登录节点)
HOST=$(hostname)
if [[ "$HOST" =~ login ]]; then
    echo ""
    echo "❌ 当前在登录节点 ($HOST), 不应在此跑长任务!"
    echo "   先 salloc 到 db4ai 节点:"
    echo "     salloc -p db4ai --qos=db4ai -N 1 -n 1 --cpus-per-task=16 --mem=128G --time=72:00:00"
    echo "     srun --jobid=<jobid> --pty /bin/bash -l"
    echo "   然后再跑此脚本."
    exit 1
fi
echo "  ✅ 跑在节点 $HOST (非登录节点)"

echo ""
echo "============================================================"
echo "[3/4] 清空旧 build_failed 标记"
echo "============================================================"
python3 analysis/clear_postfilter_failed.py

echo ""
echo "============================================================"
echo "[4/4] 启动 Post-filter v2 (synth_512d, synth_768d_hc, yahoo800k, dbpedia560k)"
echo "============================================================"
LOGFILE=analysis/run_postfilter_v2_$(date +%Y%m%d_%H%M%S).log
echo "  日志输出到: $LOGFILE"
echo "  开始时间: $(date)"
echo ""
echo "  Tip: 现在 Ctrl+A D 脱离 screen, 让它后台跑."
echo "       回来重连用: screen -r postfilter_v2"
echo ""
sleep 3

bash analysis/run_postfilter_v2.sh 2>&1 | tee "$LOGFILE"

echo ""
echo "============================================================"
echo "Post-filter v2 全部完成！结束时间: $(date)"
echo "============================================================"
