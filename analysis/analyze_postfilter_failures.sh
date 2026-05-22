#!/bin/bash
# 诊断 Post-filter v2 在 4 个数据集上为什么 build_failed
# 用法: bash analysis/analyze_postfilter_failures.sh

set +e
cd ~/benchmarks/discrete

FAISS_DIR=~/benchmarks/discrete/faiss
DATASETS="synth_512d synth_768d_hc yahoo800k dbpedia560k"

# ============================================================
# [1] progress.json 失败任务样本
# ============================================================
echo "============================================================"
echo "[1] 各数据集 progress.json 失败信息汇总"
echo "============================================================"
python3 - <<'PYEOF'
import json, os
faiss_dir = os.path.expanduser("~/benchmarks/discrete/faiss")
for ds in ["synth_512d","synth_768d_hc","yahoo800k","dbpedia560k"]:
    pf = f"{faiss_dir}/progress_postfilter_{ds}.json"
    if not os.path.exists(pf): continue
    with open(pf) as f: d = json.load(f)
    done = d.get("completed", [])
    fail = d.get("failed", [])
    print(f"--- {ds:<18} ---")
    print(f"  completed: {len(done)}, failed: {len(fail)}")
    if fail:
        print(f"  失败任务样本 (前 5):")
        for x in fail[:5]:
            print(f"    {x}")
PYEOF

# ============================================================
# [2] build.log 失败模式分类
# ============================================================
echo ""
echo "============================================================"
echo "[2] 历史 build.log 的失败原因分析 (聚合所有数据集)"
echo "============================================================"
python3 - <<'PYEOF'
import os, glob, re
from collections import Counter
faiss_dir = os.path.expanduser("~/benchmarks/discrete/faiss")
all_counts = Counter()
total_logs = 0
sample_errors = []

for ds in ["synth_512d","synth_768d_hc","yahoo800k","dbpedia560k"]:
    idir = f"{faiss_dir}/data/index_files/hnsw/{ds}"
    if not os.path.isdir(idir):
        print(f"  {ds}: 索引目录不存在 ({idir})")
        continue
    logs = glob.glob(f"{idir}/*build.log")
    if not logs:
        print(f"  {ds}: 索引目录里没 build.log 文件")
        continue
    print(f"  {ds}: {len(logs)} 个 build.log")
    for lf in logs:
        total_logs += 1
        try:
            with open(lf) as f: text = f.read()
        except: continue
        if re.search(r"libmkl|cannot open shared|error while loading shared", text):
            all_counts["MKL/共享库找不到"] += 1
        elif re.search(r"Segmentation fault|SIGSEGV|signal 11", text):
            all_counts["Segfault"] += 1
            if len(sample_errors) < 3: sample_errors.append((lf, text[-300:]))
        elif re.search(r"bad_alloc|Cannot allocate memory|Killed", text):
            all_counts["OOM"] += 1
            if len(sample_errors) < 3: sample_errors.append((lf, text[-300:]))
        elif re.search(r"Aborted|Assertion.*failed|terminate called", text):
            all_counts["Abort/Assert"] += 1
            if len(sample_errors) < 3: sample_errors.append((lf, text[-300:]))
        elif re.search(r"error|Error|ERROR", text):
            all_counts["其他错误"] += 1
            if len(sample_errors) < 3: sample_errors.append((lf, text[-300:]))
        else:
            all_counts["未识别(可能 build 中断 / log 空)"] += 1

print()
print(f"  总分析 build.log: {total_logs}")
print(f"  失败模式分布:")
for label, n in all_counts.most_common():
    print(f"    {label}: {n}")

if sample_errors:
    print()
    print("  非 MKL 类样本 (用于人工判断):")
    for lf, snippet in sample_errors[:3]:
        print(f"    --- {os.path.basename(lf)} ---")
        for line in snippet.split('\n')[-6:]:
            if line.strip(): print(f"      {line}")

PYEOF

# ============================================================
# [3] 当前环境状态
# ============================================================
echo ""
echo "============================================================"
echo "[3] 当前环境状态"
echo "============================================================"
echo "  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-(none)}"
echo "  LD_LIBRARY_PATH 前 3 项:"
echo "${LD_LIBRARY_PATH:-(empty)}" | tr ':' '\n' | head -3 | sed 's/^/    /'

BIN=$FAISS_DIR/tutorial/cpp/build_HNSW_index_static
if [ -f "$BIN" ]; then
    echo ""
    echo "  build_HNSW_index_static 链接的关键库:"
    ldd "$BIN" 2>/dev/null | grep -E "mkl|not found" | head -10 | sed 's/^/    /'
    if ldd "$BIN" 2>/dev/null | grep -q "not found"; then
        echo "    ⚠️  存在 'not found' 库 — 当前 shell 环境无法运行 binary"
    else
        echo "    ✅ 所有库可加载 — 当前 shell 环境可运行 binary"
    fi
else
    echo "  ❌ binary 不存在: $BIN"
fi

# ============================================================
# [4] 综合判断 + 下一步建议
# ============================================================
echo ""
echo "============================================================"
echo "[4] 综合判断 + 下一步建议"
echo "============================================================"
python3 - <<'PYEOF'
import os, glob, re
from collections import Counter
faiss_dir = os.path.expanduser("~/benchmarks/discrete/faiss")
counts = Counter()
total = 0
for ds in ["synth_512d","synth_768d_hc","yahoo800k","dbpedia560k"]:
    idir = f"{faiss_dir}/data/index_files/hnsw/{ds}"
    if not os.path.isdir(idir): continue
    for lf in glob.glob(f"{idir}/*build.log"):
        total += 1
        try:
            with open(lf) as f: text = f.read()
        except: continue
        if re.search(r"libmkl|cannot open shared", text): counts["MKL"] += 1
        elif re.search(r"Segmentation fault|SIGSEGV", text): counts["Segfault"] += 1
        elif re.search(r"bad_alloc|Killed", text): counts["OOM"] += 1
        elif re.search(r"Aborted|Assertion", text): counts["Abort"] += 1
        elif re.search(r"error|Error", text): counts["其他"] += 1
        else: counts["空/未识别"] += 1

print(f"  分析 {total} 个 log, 模式: {dict(counts)}")
print()

if total == 0:
    print("  ⚠️  没找到任何 build.log → 历史索引目录被清空了, 无法分析旧失败.")
    print("     建议先单跑诊断: bash analysis/diagnose_postfilter_oom.sh yahoo800k")
elif counts.get("MKL", 0) >= total * 0.5:
    print("  🎯 主要失败原因是 MKL 库找不到 (与之前诊断一致).")
    print("     如果当前 shell 环境 ldd 显示 ✅ 所有库可加载, 可以直接清 failed 重跑:")
    print("       python3 analysis/clear_postfilter_failed.py")
    print("       screen -S postfilter_v2")
    print("       bash analysis/start_postfilter_v2.sh")
elif counts.get("Segfault", 0) > 0 or counts.get("OOM", 0) > 0 or counts.get("Abort", 0) > 0:
    print("  ⚠️  存在 Segfault/OOM/Abort 失败. 不只是 MKL 问题.")
    print("     建议先单跑诊断, 看具体 stderr:")
    print("       bash analysis/diagnose_postfilter_oom.sh yahoo800k")
elif counts.get("空/未识别", 0) >= total * 0.5:
    print("  ⚠️  大量 log 是空的 / 未识别 → build 进程可能没生成有效 log.")
    print("     可能原因: 旧索引被清理或 binary 启动失败.")
    print("     建议: 单跑诊断验证当前环境是否能跑")
    print("       bash analysis/diagnose_postfilter_oom.sh yahoo800k")
else:
    print("  ⚠️  失败原因分散. 建议单跑诊断:")
    print("     bash analysis/diagnose_postfilter_oom.sh yahoo800k")
PYEOF
