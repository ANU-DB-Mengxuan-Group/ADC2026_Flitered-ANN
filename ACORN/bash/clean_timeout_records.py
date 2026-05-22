#!/usr/bin/env python3
"""
清理 progress.json 中的 timeout 失败记录，保留成功记录

用法:
  python clean_timeout_records.py                     # 清理所有 V2 数据集
  python clean_timeout_records.py synth_192d          # 只清理指定数据集
  python clean_timeout_records.py --dry-run           # 预览模式，不实际修改
"""

import sys
import json
from pathlib import Path

# V2 数据集列表
V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]

# ACORN 数据目录 (使用 resolve() 确保绝对路径)
ACORN_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_timeout_records(dataset: str, dry_run: bool = False) -> dict:
    """清理单个数据集的 timeout 记录

    progress.json 格式:
    {
      "progress": {
        "[M, M_beta, gamma, null]": {"build_done": true/false, ...},      # build 记录
        "[M, M_beta, gamma, \"scenario\"]": {"search_done": true, ...},   # search 记录
      },
      "last_update": "...",
      "total_tasks": N,
      "completed_builds": N,
      "completed_searches": N
    }
    """
    progress_file = ACORN_DATA_DIR / f"param_search_{dataset}" / "progress.json"

    if not progress_file.exists():
        return {"status": "not_found", "file": str(progress_file)}

    with open(progress_file, 'r') as f:
        data = json.load(f)

    # 任务记录在 "progress" 键下
    progress = data.get("progress", {})

    # 统计
    total_builds = 0
    total_searches = 0
    timeout_keys = []

    for key, info in progress.items():
        if not isinstance(info, dict):
            continue

        # 判断是 build 还是 search（key 包含 null 表示 build）
        is_build = "null" in key

        if is_build:
            total_builds += 1
        else:
            total_searches += 1

        # 检查是否是 timeout 失败
        if info.get("failure_reason") == "timeout":
            timeout_keys.append(key)

    stats = {
        "file": str(progress_file),
        "before": {"builds": total_builds, "searches": total_searches, "total_keys": len(progress)},
        "timeout_records": timeout_keys,
        "removed_count": len(timeout_keys)
    }

    # 删除 timeout 记录
    if not dry_run and timeout_keys:
        for key in timeout_keys:
            del progress[key]

        # 写回整个 data（包含 progress 和元数据）
        with open(progress_file, 'w') as f:
            json.dump(data, f, indent=2)
        stats["status"] = "cleaned"
    elif dry_run:
        stats["status"] = "dry_run"
    else:
        stats["status"] = "no_change"

    stats["after"] = {"total_keys": len(progress) - (0 if dry_run else len(timeout_keys))}
    return stats


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        datasets = [d for d in args if d in V2_DATASETS]
        if not datasets:
            print(f"错误: 无效的数据集。可用: {V2_DATASETS}")
            sys.exit(1)
    else:
        datasets = V2_DATASETS

    print("=" * 60)
    print("清理 ACORN progress.json 中的 timeout 记录")
    print(f"模式: {'预览 (dry-run)' if dry_run else '实际修改'}")
    print(f"数据集: {datasets}")
    print("=" * 60)

    total_removed = 0

    for dataset in datasets:
        print(f"\n[{dataset}]")
        stats = clean_timeout_records(dataset, dry_run)

        if stats["status"] == "not_found":
            print(f"  ⚠️  文件不存在: {stats['file']}")
            continue

        print(f"  文件: {stats['file']}")
        print(f"  总记录数: {stats['before']['total_keys']} → {stats['after']['total_keys']}")
        print(f"  Timeout 记录: {stats['removed_count']}")

        if stats['timeout_records']:
            for key in stats['timeout_records']:
                print(f"    - {key}")

        if stats["status"] == "cleaned":
            print(f"  ✅ 已清理 {stats['removed_count']} 条 timeout 记录")
        elif stats["status"] == "dry_run":
            print(f"  🔍 预览模式，未实际修改")
        else:
            print(f"  ℹ️  无 timeout 记录需要清理")

        total_removed += stats["removed_count"]

    print("\n" + "=" * 60)
    print(f"总计发现: {total_removed} 条 timeout 记录")
    if dry_run and total_removed > 0:
        print("⚠️  这是预览模式。去掉 --dry-run 参数以实际执行清理。")
    print("=" * 60)


if __name__ == "__main__":
    main()
