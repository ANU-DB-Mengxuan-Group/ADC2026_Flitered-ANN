#!/usr/bin/env python3
"""
调试 GT 和搜索结果，验证 recall 计算是否正确

用法:
  python scripts/debug_gt_results.py arxiv
  python scripts/debug_gt_results.py arxiv index_R=128_L=100_Ls=100
"""

import sys
import os
import numpy as np
from pathlib import Path

DISKANN_DIR = "/home/remote/u7905817/benchmarks/discrete/DiskANN"

def load_gt(gt_file):
    """加载 DiskANN 格式的 GT 文件"""
    with open(gt_file, 'rb') as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.int32)
        ids = np.frombuffer(f.read(nq*k*4), dtype=np.uint32).reshape(nq, k)
        dists = np.frombuffer(f.read(nq*k*4), dtype=np.float32).reshape(nq, k)
    return nq, k, ids, dists

def load_result(result_file):
    """加载 DiskANN 搜索结果"""
    with open(result_file, 'rb') as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.int32)
        ids = np.frombuffer(f.read(nq*k*4), dtype=np.uint32).reshape(nq, k)
    return nq, k, ids

def compute_recall(gt_ids, result_ids):
    """手动计算 recall"""
    nq = gt_ids.shape[0]
    k = gt_ids.shape[1]

    recalls = []
    for i in range(nq):
        gt_set = set(gt_ids[i])
        result_set = set(result_ids[i])

        # 移除无效 ID (0xFFFFFFFF = 4294967295)
        gt_set.discard(4294967295)
        result_set.discard(4294967295)

        if len(gt_set) == 0:
            recalls.append(1.0)  # 没有有效 GT，视为完美
        else:
            correct = len(gt_set & result_set)
            recalls.append(correct / len(gt_set))

    return np.array(recalls)

def main():
    if len(sys.argv) < 2:
        print("用法: python debug_gt_results.py <dataset> [result_prefix]")
        sys.exit(1)

    dataset = sys.argv[1]
    output_base = f"{DISKANN_DIR}/data_fixed_eq"

    # 加载 GT
    gt_file = f"{output_base}/converted_labels/{dataset}/gt_diskann.bin"
    if not os.path.exists(gt_file):
        print(f"GT file not found: {gt_file}")
        sys.exit(1)

    nq, k, gt_ids, gt_dists = load_gt(gt_file)
    print(f"=" * 60)
    print(f"GT 文件: {gt_file}")
    print(f"Queries: {nq}, K: {k}")
    print(f"=" * 60)

    # 分析 GT
    print("\n【GT 分析】")
    invalid_counts = []
    for i in range(nq):
        invalid = np.sum(gt_ids[i] == 4294967295)
        invalid_counts.append(invalid)

    print(f"  有无效ID的query数: {np.sum(np.array(invalid_counts) > 0)}")
    print(f"  平均无效ID数: {np.mean(invalid_counts):.2f}")
    print(f"  最大无效ID数: {np.max(invalid_counts)}")

    print("\n【前5个query的GT】")
    for i in range(min(5, nq)):
        valid_count = np.sum(gt_ids[i] != 4294967295)
        print(f"  Q{i}: IDs={gt_ids[i][:5]}... (valid={valid_count})")
        print(f"       dists={gt_dists[i][:5]}")

    # 如果指定了结果文件，加载并比较
    if len(sys.argv) > 2:
        result_prefix = sys.argv[2]
    else:
        # 查找最新的结果文件
        result_dir = f"{output_base}/results/{dataset}"
        result_files = list(Path(result_dir).glob("*_idx_uint32.bin"))
        if not result_files:
            print(f"\n没有找到结果文件在 {result_dir}")
            return
        result_prefix = str(result_files[0]).replace("_idx_uint32.bin", "").replace(result_dir + "/", "")

    result_file = f"{output_base}/results/{dataset}/{result_prefix}_idx_uint32.bin"
    if not os.path.exists(result_file):
        # 尝试添加 _100 后缀
        result_file = f"{output_base}/results/{dataset}/{result_prefix}_100_idx_uint32.bin"

    if os.path.exists(result_file):
        print(f"\n{'=' * 60}")
        print(f"结果文件: {result_file}")
        print(f"{'=' * 60}")

        _, _, result_ids = load_result(result_file)

        print("\n【前5个query的搜索结果】")
        for i in range(min(5, nq)):
            print(f"  Q{i}: {result_ids[i]}")

        # 手动计算 recall
        recalls = compute_recall(gt_ids, result_ids)
        print(f"\n【手动计算的 Recall】")
        print(f"  Mean Recall@{k}: {np.mean(recalls):.4f}")
        print(f"  Min Recall: {np.min(recalls):.4f}")
        print(f"  Max Recall: {np.max(recalls):.4f}")
        print(f"  Recall=1.0 的query数: {np.sum(recalls == 1.0)}/{nq}")

        # 检查 GT 和 result 的匹配情况
        print(f"\n【逐query对比 (前10个)】")
        for i in range(min(10, nq)):
            gt_set = set(gt_ids[i]) - {4294967295}
            result_set = set(result_ids[i]) - {4294967295}
            correct = len(gt_set & result_set)
            print(f"  Q{i}: GT有效={len(gt_set)}, 结果={len(result_set)}, 正确={correct}, recall={correct/max(len(gt_set),1):.2f}")
    else:
        print(f"\n结果文件不存在: {result_file}")

if __name__ == '__main__':
    main()
