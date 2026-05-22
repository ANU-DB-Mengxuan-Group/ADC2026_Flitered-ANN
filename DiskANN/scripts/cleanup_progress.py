#!/usr/bin/env python3
"""
清理 progress 文件中错误的记录

用法:
  python scripts/cleanup_progress.py arxiv 128 192    # 删除 R=128,192 的搜索记录
  python scripts/cleanup_progress.py arxiv 128        # 删除 R=128 的搜索记录
"""

import json
import sys
from pathlib import Path

DISKANN_DIR = "/home/remote/u7905817/benchmarks/discrete/DiskANN"

def main():
    if len(sys.argv) < 3:
        print("用法: python cleanup_progress.py <dataset> <R_value> [R_value2] ...")
        print("例如: python cleanup_progress.py arxiv 128 192")
        sys.exit(1)

    dataset = sys.argv[1]
    r_values = [int(x) for x in sys.argv[2:]]

    progress_file = f"{DISKANN_DIR}/data_fixed_eq/progress/{dataset}.json"

    if not Path(progress_file).exists():
        print(f"Progress file not found: {progress_file}")
        sys.exit(1)

    with open(progress_file) as f:
        p = json.load(f)

    original_count = len(p['completed'])

    # 删除指定 R 值的搜索记录，保留 build 记录
    prefixes_to_remove = [f"search_{r}_" for r in r_values]

    p['completed'] = [
        x for x in p['completed']
        if not any(x.startswith(prefix) for prefix in prefixes_to_remove)
    ]

    removed_count = original_count - len(p['completed'])

    with open(progress_file, 'w') as f:
        json.dump(p, f, indent=2)

    print(f"删除了 {removed_count} 条搜索记录")
    print(f"保留了 {len(p['completed'])} 条记录")
    print(f"文件已更新: {progress_file}")

if __name__ == '__main__':
    main()
