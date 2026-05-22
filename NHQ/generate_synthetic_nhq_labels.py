#!/usr/bin/env python3
"""
为NHQ生成合成固定长度标签数据

根据论文：
- 固定长度: 4
- 每个位置: 3个可能值 (0, 1, 2)
- 均匀分布选择
"""

import os
import sys
import random
import numpy as np

def load_fvecs_count(filename):
    """获取fvecs文件中的向量数量"""
    with open(filename, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0, 2)
        n = f.tell() // (4 + d * 4)
    return n

def generate_synthetic_labels(num_points, num_attrs=4, num_values=3, seed=42):
    """生成合成标签"""
    random.seed(seed)
    labels = []
    for _ in range(num_points):
        label = [str(random.randint(0, num_values - 1)) for _ in range(num_attrs)]
        labels.append(label)
    return labels

def save_labels(labels, output_file, num_attrs, delimiter=' '):
    """保存标签到文件"""
    with open(output_file, 'w') as f:
        f.write(f"{len(labels)}{delimiter}{num_attrs}\n")
        for label in labels:
            f.write(delimiter.join(label) + '\n')
    print(f"  Saved {len(labels)} labels to {output_file}")

def generate_query_labels(base_labels, num_queries, seed=123):
    """从base标签中随机采样query标签"""
    random.seed(seed)
    return random.choices(base_labels, k=num_queries)

def find_query_file(data_dir, dataset):
    # 优先使用_equal.fvecs（Fixed-EQ实验用的query）
    candidates = [
        f"{data_dir}/{dataset}/{dataset}_query_equal.fvecs",
        f"{data_dir}/{dataset}/{dataset}_query.fvecs",
    ]
    for f in candidates:
        if os.path.exists(f):
            return f
    return None

def process_dataset(dataset_name, data_dir, output_dir, num_attrs=4, num_values=3):
    """处理一个数据集"""
    print(f"\nProcessing {dataset_name}...")
    
    base_file = f"{data_dir}/{dataset_name}/{dataset_name}_base.fvecs"
    query_file = find_query_file(data_dir, dataset_name)
    
    if not os.path.exists(base_file):
        print(f"  Base file not found: {base_file}")
        return
    if query_file is None:
        print(f"  Query file not found for {dataset_name}")
        return
    
    num_base = load_fvecs_count(base_file)
    num_queries = load_fvecs_count(query_file)
    
    print(f"  Base vectors: {num_base}, Query vectors: {num_queries}")
    
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 生成base标签
    print(f"  Generating base labels (attrs={num_attrs}, values={num_values})...")
    base_labels = generate_synthetic_labels(num_base, num_attrs, num_values, seed=42)
    save_labels(base_labels, os.path.join(dataset_dir, 'label_NHQ_base.txt'), num_attrs)
    
    # 生成query标签（从base采样，数量与query向量一致）
    print(f"  Generating {num_queries} query labels...")
    query_labels = generate_query_labels(base_labels, num_queries, seed=123)
    save_labels(query_labels, os.path.join(dataset_dir, 'label_NHQ_query.txt'), num_attrs)
    
    # 统计
    from collections import Counter
    label_counts = Counter(tuple(l) for l in base_labels)
    print(f"  Unique label combinations: {len(label_counts)} (expected ~{num_values**num_attrs})")

def main():
    data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
    output_dir = '/home/remote/u7905817/benchmarks/discrete/NHQ/synthetic_labels'
    
    datasets = ['arxiv', 'yfcc']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        process_dataset(dataset, data_dir, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
