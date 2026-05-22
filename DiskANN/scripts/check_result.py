#!/usr/bin/env python3
"""
检查 DiskANN 搜索结果文件

用法:
  python scripts/check_result.py /tmp/test_nofilt_100_idx_uint32.bin
"""

import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("用法: python check_result.py <result_file>")
        sys.exit(1)

    result_file = sys.argv[1]

    with open(result_file, 'rb') as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.int32)
        ids = np.frombuffer(f.read(nq*k*4), dtype=np.uint32).reshape(nq, k)

    print(f"文件: {result_file}")
    print(f"Queries: {nq}, K: {k}")
    print(f"\n前10个query的结果:")
    for i in range(min(10, nq)):
        print(f"  Q{i}: {ids[i]}")

    # 统计
    all_zeros = np.sum(np.all(ids == 0, axis=1))
    has_zeros = np.sum(np.any(ids == 0, axis=1))
    unique_ids = len(np.unique(ids))

    print(f"\n统计:")
    print(f"  全为0的query数: {all_zeros}/{nq}")
    print(f"  包含0的query数: {has_zeros}/{nq}")
    print(f"  不同ID总数: {unique_ids}")

if __name__ == '__main__':
    main()
