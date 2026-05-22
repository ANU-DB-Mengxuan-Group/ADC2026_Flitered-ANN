#!/usr/bin/env python3
"""
为UNG合成数据生成groundtruth

Fixed-Length Equality: 找到标签集合完全相同的最近邻
"""

import os
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_fvecs(filename):
    """加载fvecs格式的向量"""
    with open(filename, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        n = f.tell() // (4 + d * 4)
        f.seek(0)

        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]
            vectors[i] = np.fromfile(f, dtype=np.float32, count=dim)

    return vectors

def load_labels(filename):
    """加载UNG格式的标签文件（逗号分隔，1-based整数）

    转为frozenset用于比较。
    """
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label = frozenset(int(x) for x in line.split(','))
                labels.append(label)
    return labels

def compute_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k=10):
    """计算groundtruth，返回(ids, distances)"""
    # 按标签分组
    label_to_ids = defaultdict(list)
    for i, label in enumerate(base_labels):
        label_to_ids[label].append(i)

    print(f"  Found {len(label_to_ids)} unique label groups")

    gt_ids = []
    gt_dists = []
    for q_idx, q_label in enumerate(query_labels):
        matching_ids = label_to_ids.get(q_label, [])

        if len(matching_ids) == 0:
            gt_ids.append([-1] * k)
            gt_dists.append([float('inf')] * k)
            continue

        q_vec = query_vectors[q_idx]
        distances = []
        for base_id in matching_ids:
            dist = np.sum((base_vectors[base_id] - q_vec) ** 2)
            distances.append((dist, base_id))

        distances.sort()
        top_k_ids = [d[1] for d in distances[:k]]
        top_k_dists = [d[0] for d in distances[:k]]

        while len(top_k_ids) < k:
            top_k_ids.append(-1)
            top_k_dists.append(float('inf'))

        gt_ids.append(top_k_ids)
        gt_dists.append(top_k_dists)

    return gt_ids, gt_dists

def save_groundtruth_bin(gt, gt_distances, output_file):
    """保存groundtruth（UNG二进制格式）

    UNG格式: num_queries * K 个 pair<int32, float32>
    每个pair是 (neighbor_id, distance)，无header
    """
    import struct
    num_queries = len(gt)
    k = len(gt[0])

    with open(output_file, 'wb') as f:
        for i in range(num_queries):
            for j in range(k):
                # pair<int32, float32>
                f.write(struct.pack('if', gt[i][j], gt_distances[i][j]))
    print(f"  Saved GT (binary) to {output_file}: {num_queries} queries, K={k}")

def process_dataset(dataset, data_dir, label_dir, k=10):
    """处理一个数据集"""
    print(f"\nProcessing {dataset}...")

    base_vec_file = f"{data_dir}/{dataset}/{dataset}_base.fvecs"
    query_vec_file = f"{data_dir}/{dataset}/{dataset}_query_equal.fvecs"
    if not os.path.exists(query_vec_file):
        query_vec_file = f"{data_dir}/{dataset}/{dataset}_query.fvecs"

    base_label_file = f"{label_dir}/{dataset}/label_base.txt"
    query_label_file = f"{label_dir}/{dataset}/label_query.txt"

    for f in [base_vec_file, base_label_file, query_label_file]:
        if not os.path.exists(f):
            print(f"  File not found: {f}")
            return

    print(f"  Loading vectors...")
    print(f"    Using query file: {query_vec_file}")
    base_vectors = load_fvecs(base_vec_file)
    query_vectors = load_fvecs(query_vec_file)
    print(f"    Base: {base_vectors.shape}, Query: {query_vectors.shape}")

    base_labels = load_labels(base_label_file)
    query_labels = load_labels(query_label_file)
    query_vectors = query_vectors[:len(query_labels)]
    print(f"    Base labels: {len(base_labels)}, Query labels: {len(query_labels)}")

    print(f"  Computing groundtruth (k={k})...")
    gt_ids, gt_dists = compute_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k)

    save_groundtruth_bin(gt_ids, gt_dists, f"{label_dir}/{dataset}/gt.bin")

def main():
    if os.path.exists('/home/remote/u7905817'):
        data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
        label_dir = '/home/remote/u7905817/benchmarks/discrete/UNG-dev/synthetic_labels'
    else:
        data_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/data'
        label_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/UNG-dev/synthetic_labels'

    datasets = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']

    for dataset in datasets:
        process_dataset(dataset, data_dir, label_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
