#!/bin/bash
# ---------------------------------------------------------------
# 目的: 重跑 UNG yfcc OR + Containment (V1 数据被覆盖了)
# 用法: 在 salloc + srun --pty 的 screen 会话里跑
#       cd ~/benchmarks/discrete/UNG-dev
#       bash run_ung_yfcc_or_and.sh 2>&1 | tee ung_yfcc_oa.log
# 数据落点: ~/benchmarks/discrete/UNG-dev/results_original/yfcc/summary.csv
#           (含 scenario 列, 跟旧的 results/yfcc/summary.csv 不冲突)
# 耗时: 编译 ~5min + overlap+containment 共用 index 6-8h
# ---------------------------------------------------------------
#
# 重要: 一次跑全 dataset (yfcc), 让 overlap 和 containment 共用 general index,
#       避免分两次跑各自重编 index 浪费几小时.
#       Equality progress.json 已有 60 个 completed, 会自动 skip.

set -e

# 用当前激活的 conda env 即可 (auto_ung_original.py 只用 stdlib, 任何 Python 3 都行)
# 如果没激活 env, 这里默认进 benchmark
source ~/miniconda3/etc/profile.d/conda.sh
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    conda activate benchmark
fi
echo "Using conda env: $CONDA_DEFAULT_ENV"

cd ~/benchmarks/discrete/UNG-dev

# === Step 1: 编译 UNG (如果未编译) ===
if [ ! -x build/apps/build_UNG_index ] || [ ! -x build/apps/search_UNG_index ]; then
    echo "=== Compiling UNG ==="
    cd codes
    mkdir -p build && cd build
    cmake .. > cmake.log 2>&1
    make -j16 build_UNG_index search_UNG_index 2>&1 | tail -20
    cd ../..
    if [ ! -d build ]; then
        ln -sfn codes/build build
    fi
fi

# === Step 2: 备份旧 progress JSON 让 overlap + containment 重跑 ===
# 不动 equality progress (已完成, 让脚本 skip)
TS=$(date +%Y%m%d_%H%M%S)
for sc in overlap containment; do
    if [ -f progress_yfcc_${sc}.json ]; then
        mv progress_yfcc_${sc}.json progress_yfcc_${sc}.json.bak.${TS}
        echo "Backed up progress_yfcc_${sc}.json -> .bak.${TS}"
    fi
done

# === Step 3: 一次跑 yfcc 全 scenario (equality 自动 skip, overlap+containment 共用 general index) ===
echo
echo "================== Running yfcc (all scenarios, sharing general index) =================="
date
python bash/auto_ung_original.py yfcc

# === Step 4: 汇报 ===
echo
echo "================== Done =================="
date
echo "Result summary:"
ls -lh results_original/yfcc/summary.csv 2>/dev/null || echo "WARNING: summary.csv 没生成!"
echo
echo "Rows by scenario:"
if [ -f results_original/yfcc/summary.csv ]; then
    awk -F',' 'NR>1{print $2}' results_original/yfcc/summary.csv | sort | uniq -c
fi
