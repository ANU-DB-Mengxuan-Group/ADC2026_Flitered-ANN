#!/usr/bin/env python3
"""
为UNG原始元数据实验准备数据

主要任务:
1. 转换label文件为UNG格式（1-based，逗号分隔）
2. 转换groundtruth从txt到UNG二进制格式
3. 检查/转换向量文件格式

使用:
  cd ~/benchmarks/discrete/UNG-dev
  python prepare_ung_original_data.py arxiv
  python prepare_ung_original_data.py yfcc
"""

import os
import sys
import struct
import numpy as np
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

def save_ung_bin(vectors, output_file):
    """保存为UNG二进制格式"""
    n, d = vectors.shape
    with open(output_file, 'wb') as f:
        f.write(struct.pack('II', n, d))
        vectors.astype(np.float32).tofile(f)
    print(f"  Saved {n} vectors (dim={d}) to {output_file}")

def convert_labels_to_ung(input_file, output_file):
    """转换标签文件为UNG格式 (1-based, 逗号分隔)

    ACORN/其他算法的标签可能是:
    - 逗号分隔，0-based
    - 空格分隔

    UNG需要: 逗号分隔，1-based
    """
    labels = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检测分隔符
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()

            # 转为整数 (假设原始是0-based，转为1-based)
            label_ids = [int(x) + 1 for x in parts if x.strip()]
            labels.append(label_ids)

    # 保存为UNG格式
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(','.join(str(x) for x in label) + '\n')

    print(f"  Converted {len(labels)} labels to {output_file}")
    return labels

def convert_gt_txt_to_bin(txt_file, bin_file, base_vectors, query_vectors, query_labels, base_labels, k=10):
    """转换groundtruth从txt到UNG二进制格式

    输入txt格式: 每行K个空格分隔的ID
    输出bin格式: num_queries * K 个 pair<int32, float32>

    需要计算距离，因为UNG格式需要(ID, distance)对
    """
    gt_ids = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = [int(x) for x in line.split()[:k]]
            while len(ids) < k:
                ids.append(-1)
            gt_ids.append(ids)

    print(f"  Loaded {len(gt_ids)} queries from {txt_file}")

    # 计算距离
    with open(bin_file, 'wb') as f:
        for q_idx, ids in enumerate(gt_ids):
            q_vec = query_vectors[q_idx]
            for base_id in ids:
                if base_id >= 0 and base_id < len(base_vectors):
                    dist = np.sum((base_vectors[base_id] - q_vec) ** 2)
                else:
                    dist = float('inf')
                f.write(struct.pack('if', base_id, dist))

    print(f"  Saved GT (binary) to {bin_file}: {len(gt_ids)} queries")

def load_ung_bin(filename):
    """加载UNG二进制格式的向量"""
    with open(filename, 'rb') as f:
        n, d = struct.unpack('II', f.read(8))
        vectors = np.fromfile(f, dtype=np.float32).reshape(n, d)
    return vectors

def process_dataset(dataset, data_dir):
    """处理一个数据集"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset}")
    print(f"{'='*60}")

    dataset_dir = f"{data_dir}/{dataset}"

    # 检查基础文件
    base_bin = f"{dataset_dir}/{dataset}_base.bin"
    base_fvecs = f"{dataset_dir}/{dataset}_base.fvecs"
    label_base = f"{dataset_dir}/label_base.txt"

    # 1. 转换base向量 (如果需要)
    if not os.path.exists(base_bin):
        if os.path.exists(base_fvecs):
            print(f"\nConverting base vectors to UNG format...")
            vectors = load_fvecs(base_fvecs)
            save_ung_bin(vectors, base_bin)
        else:
            print(f"Error: Neither {base_bin} nor {base_fvecs} found")
            return
    else:
        print(f"\nBase vectors already in UNG format: {base_bin}")

    # 加载base向量用于计算距离
    print(f"\nLoading base vectors...")
    base_vectors = load_ung_bin(base_bin)
    print(f"  Base: {base_vectors.shape}")

    # 加载base标签
    base_labels = []
    with open(label_base, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if ',' in line:
                    base_labels.append([int(x) for x in line.split(',')])
                else:
                    base_labels.append([int(x) for x in line.split()])
    print(f"  Base labels: {len(base_labels)}")

    # 处理每个场景
    scenarios = [
        ("and", "containment"),
        ("or", "overlap"),
        ("equal", "equality"),
    ]

    for suffix, scenario in scenarios:
        print(f"\n--- {scenario.upper()} (suffix: {suffix}) ---")

        query_bin = f"{dataset_dir}/{dataset}_query_{suffix}.bin"
        query_fvecs = f"{dataset_dir}/{dataset}_query_{suffix}.fvecs"
        query_label = f"{dataset_dir}/{dataset}_query_{suffix}.txt"
        gt_txt = f"{dataset_dir}/{dataset}_gt_{suffix}.txt"
        gt_bin = f"{dataset_dir}/{dataset}_gt_{suffix}.bin"

        # 检查query向量
        if not os.path.exists(query_bin):
            if os.path.exists(query_fvecs):
                print(f"  Converting query vectors...")
                vectors = load_fvecs(query_fvecs)
                save_ung_bin(vectors, query_bin)
            else:
                print(f"  Warning: Query file not found for {suffix}")
                continue
        else:
            print(f"  Query vectors OK: {query_bin}")

        # 检查query标签
        if not os.path.exists(query_label):
            print(f"  Warning: Query labels not found: {query_label}")
            continue
        else:
            print(f"  Query labels OK: {query_label}")

        # 转换groundtruth
        if os.path.exists(gt_txt):
            print(f"  Converting groundtruth to UNG binary format...")

            # 加载query向量和标签
            query_vectors = load_ung_bin(query_bin)
            query_labels = []
            with open(query_label, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if ',' in line:
                            query_labels.append([int(x) for x in line.split(',')])
                        else:
                            query_labels.append([int(x) for x in line.split()])

            convert_gt_txt_to_bin(gt_txt, gt_bin, base_vectors, query_vectors,
                                  query_labels, base_labels, k=10)
        elif os.path.exists(gt_bin):
            print(f"  Groundtruth binary OK: {gt_bin}")
        else:
            print(f"  Warning: No groundtruth found for {suffix}")

def main():
    if os.path.exists('/home/remote/u7905817'):
        data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
    else:
        data_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/data'

    # 所有6个数据集
    all_datasets = ['arxiv', 'tripclick', 'LAION1M', 'yfcc', 'ytb_audio', 'ytb_video']

    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = all_datasets

    for dataset in datasets:
        process_dataset(dataset, data_dir)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
