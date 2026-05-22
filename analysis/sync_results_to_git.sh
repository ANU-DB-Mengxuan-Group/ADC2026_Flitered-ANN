#!/bin/bash
# 把云端实验结果(progress.json / summary.csv 等)同步到 git
#
# 用法:
#   bash analysis/sync_results_to_git.sh           # dry-run: 只显示会 add/commit 啥
#   bash analysis/sync_results_to_git.sh --commit  # 真正 add + commit + push
#
# 处理范围:
#   ✅ ACORN/data/param_search_*/progress.json
#   ✅ ACORN/data/param_search_*/results/**/*summary.csv
#   ✅ faiss/progress_postfilter_*.json
#   ✅ faiss/results_postfilter/**/*summary.csv
#   ✅ analysis/*.csv (per-query 数据等)
#   ❌ 跳过大文件 (> 50MB), index 文件, *.bin 等

set +e
cd ~/benchmarks/discrete

ACTION=${1:-dry-run}

echo "============================================================"
echo "[1] 扫描可同步的实验结果文件"
echo "============================================================"

# 候选文件清单
TARGETS=()
add_target() {
    local f="$1"
    if [ -f "$f" ]; then
        local size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
        if [ "$size" -gt 52428800 ]; then  # 50 MB
            echo "  ⏭️  跳过(>50MB): $f ($((size/1024/1024))MB)"
        else
            TARGETS+=("$f")
            echo "  📄 $f ($(($size/1024)) KB)"
        fi
    fi
}

echo ""
echo "--- ACORN progress + summary ---"
for d in ACORN/data/param_search_*/; do
    add_target "${d}progress.json"
    for sf in $(find "${d}results/" -name "summary.csv" 2>/dev/null); do
        add_target "$sf"
    done
done

echo ""
echo "--- Post-filter progress + summary ---"
for f in faiss/progress_postfilter_*.json; do
    add_target "$f"
done
for sf in $(find faiss/results_postfilter/ -name "summary.csv" 2>/dev/null); do
    add_target "$sf"
done

echo ""
echo "--- analysis 目录里的 csv ---"
for f in analysis/*.csv analysis/per_label_lid/*.json analysis/*_metrics.json; do
    add_target "$f"
done

echo ""
echo "============================================================"
echo "[2] git status (相对仓库根)"
echo "============================================================"
echo "  共 ${#TARGETS[@]} 个候选文件 (≤50MB)"
echo ""
echo "  其中相对 git 状态的细分:"
new_count=0; mod_count=0; nochange=0
for f in "${TARGETS[@]}"; do
    status=$(git status -s "$f" 2>/dev/null)
    if [ -z "$status" ]; then
        nochange=$((nochange+1))
    elif [[ "$status" == "??"* ]]; then
        new_count=$((new_count+1))
    elif [[ "$status" == " M"* ]] || [[ "$status" == "M "* ]]; then
        mod_count=$((mod_count+1))
    fi
done
echo "    新文件 (??):    $new_count"
echo "    已改 (M):       $mod_count"
echo "    无变化:         $nochange"

echo ""
echo "============================================================"
if [ "$ACTION" != "--commit" ]; then
    echo "[3] dry-run 模式 - 没真正改 git"
    echo "============================================================"
    echo ""
    echo "如果要真正 add + commit + push, 跑:"
    echo "  bash analysis/sync_results_to_git.sh --commit"
    exit 0
fi

# 真正执行
echo "[3] 执行 add + commit + push"
echo "============================================================"

if [ $((new_count + mod_count)) -eq 0 ]; then
    echo "  ✅ 没有需要 commit 的改动, 退出."
    exit 0
fi

# 批量 add (用 xargs 防止参数过多)
printf '%s\n' "${TARGETS[@]}" | xargs git add -- 2>&1 | head -5

echo ""
echo "git status 暂存区:"
git status -s | head -20
echo "  (共 $(git status -s | wc -l) 个文件待 commit)"

echo ""
echo "Commit 中..."
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
git commit -m "Sync 云端实验结果: progress + summary ($TIMESTAMP)

ACORN v2: $(ls ACORN/data/param_search_*/progress.json 2>/dev/null | wc -l) 个数据集
Post-filter v2: $(ls faiss/progress_postfilter_*.json 2>/dev/null | wc -l) 个数据集

包含 progress.json + summary.csv 类小文件, 跳过 >50MB 的大文件.
" 2>&1 | tail -5

echo ""
echo "Push 中..."
git push origin master 2>&1 | tail -5

echo ""
echo "============================================================"
echo "完成. $(date)"
echo "============================================================"
