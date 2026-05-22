#!/usr/bin/env python3
"""
为NHQ合成数据生成groundtruth

Fixed-Length Equality: 找到标签完全相同的最近邻
"""

import os
import sys
import numpy as np
from collections import defaultdict

def load_fvecs(filename):
    """加载fvecs格式的向量"""
    with open(filename, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        f.seek(0, 2)
        n = f.tell() // (4 + d * 4)
        f.seek(0)
        
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]
            vectors[i] = np.fromfile(f, dtype=np.float32, count=dim)
    
    return vectors

def load_labels(filename):
    """加载标签文件"""
    labels = []
    with open(filename, 'r') as f:
        header = f.readline().strip().split()
        num_points = int(header[0])
        for line in f:
            label = tuple(line.strip().split())
            labels.append(label)
    return labels

def compute_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k=10):
    """计算groundtruth"""
    label_to_ids = defaultdict(list)
    for i, label in enumerate(base_labels):
        label_to_ids[label].append(i)
    
    print(f"  Found {len(label_to_ids)} unique label groups")
    
    gt = []
    for q_idx, q_label in enumerate(query_labels):
        matching_ids = label_to_ids.get(q_label, [])
        
        if len(matching_ids) == 0:
            gt.append([-1] * k)
            continue
        
        q_vec = query_vectors[q_idx]
        distances = []
        for base_id in matching_ids:
            dist = np.sum((base_vectors[base_id] - q_vec) ** 2)
            distances.append((dist, base_id))
        
        distances.sort()
        top_k = [d[1] for d in distances[:k]]
        
        while len(top_k) < k:
            top_k.append(-1)
        
        gt.append(top_k)
    
    return gt

def save_groundtruth(gt, output_file):
    """保存groundtruth"""
    with open(output_file, 'w') as f:
        for row in gt:
            f.write(' '.join(str(x) for x in row) + '\n')
    print(f"  Saved GT to {output_file}")

def find_query_file(data_dir, dataset):
    """查找query向量文件，优先使用_equal.fvecs"""
    candidates = [
        f"{data_dir}/{dataset}/{dataset}_query_equal.fvecs",
        f"{data_dir}/{dataset}/{dataset}_query.fvecs",
    ]
    for f in candidates:
        if os.path.exists(f):
            return f
    return None

def process_dataset(dataset, data_dir, label_dir, output_dir, k=10):
    """处理一个数据集"""
    print(f"\nProcessing {dataset}...")
    
    base_vec_file = f"{data_dir}/{dataset}/{dataset}_base.fvecs"
    query_vec_file = find_query_file(data_dir, dataset)
    base_label_file = f"{label_dir}/{dataset}/label_NHQ_base.txt"
    query_label_file = f"{label_dir}/{dataset}/label_NHQ_query.txt"
    
    if query_vec_file is None:
        print(f"  Query file not found for {dataset}")
        return
    
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
    gt = compute_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k)
    
    os.makedirs(f"{output_dir}/{dataset}", exist_ok=True)
    save_groundtruth(gt, f"{output_dir}/{dataset}/gt_NHQ.txt")

def main():
    data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
    label_dir = '/home/remote/u7905817/benchmarks/discrete/NHQ/synthetic_labels'
    output_dir = '/home/remote/u7905817/benchmarks/discrete/NHQ/synthetic_labels'
    
    datasets = ['arxiv', 'yfcc']
    
    for dataset in datasets:
        process_dataset(dataset, data_dir, label_dir, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
