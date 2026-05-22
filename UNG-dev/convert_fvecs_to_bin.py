#!/usr/bin/env python3
"""
将fvecs格式转换为UNG二进制格式

UNG格式:
- 4字节 uint32: N (向量数量)
- 4字节 uint32: D (维度)
- N*D个 float32: 向量数据

fvecs格式:
- 每个向量: 4字节int32(维度) + D个float32(数据)
"""

import os
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
        # 写入header: N, D (uint32)
        f.write(struct.pack('II', n, d))
        # 写入向量数据
        vectors.astype(np.float32).tofile(f)
    print(f"  Saved {n} vectors (dim={d}) to {output_file}")

def convert_dataset(dataset, data_dir, output_dir):
    """转换一个数据集"""
    print(f"\nConverting {dataset}...")

    # 创建输出目录
    out_path = Path(output_dir) / dataset
    out_path.mkdir(parents=True, exist_ok=True)

    # 转换base向量
    base_fvecs = f"{data_dir}/{dataset}/{dataset}_base.fvecs"
    if os.path.exists(base_fvecs):
        print(f"  Loading base vectors...")
        base_vectors = load_fvecs(base_fvecs)
        save_ung_bin(base_vectors, out_path / f"{dataset}_base.bin")
    else:
        print(f"  Warning: {base_fvecs} not found")

    # 转换query向量 (优先使用_query_equal.fvecs)
    query_fvecs = f"{data_dir}/{dataset}/{dataset}_query_equal.fvecs"
    if not os.path.exists(query_fvecs):
        query_fvecs = f"{data_dir}/{dataset}/{dataset}_query.fvecs"

    if os.path.exists(query_fvecs):
        print(f"  Loading query vectors from {os.path.basename(query_fvecs)}...")
        query_vectors = load_fvecs(query_fvecs)
        save_ung_bin(query_vectors, out_path / f"{dataset}_query.bin")
    else:
        print(f"  Warning: query file not found")

def main():
    if os.path.exists('/home/remote/u7905817'):
        data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
        output_dir = '/home/remote/u7905817/benchmarks/discrete/UNG-dev/data_bin'
    else:
        data_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/data'
        output_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/UNG-dev/data_bin'

    datasets = ['arxiv', 'yfcc']

    for dataset in datasets:
        convert_dataset(dataset, data_dir, output_dir)

    print("\nDone! Binary files saved to:", output_dir)

if __name__ == '__main__':
    main()
