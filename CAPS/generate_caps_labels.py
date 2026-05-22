#!/usr/bin/env python3
"""
为CAPS生成合成固定长度标签数据（逗号分隔格式）

CAPS期望的格式：逗号分隔
132687,4
2,0,0,2
1,0,0,0
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

def save_labels_comma(labels, output_file, num_attrs):
    """保存标签到文件（逗号分隔，CAPS格式，无header）"""
    with open(output_file, 'w') as f:
        # CAPS代码里跳过header的行被注释了，所以不生成header
        for label in labels:
            f.write(','.join(label) + '\n')
    print(f"  Saved {len(labels)} labels to {output_file}")

def generate_query_labels(base_labels, num_queries, seed=123):
    """从base标签中随机采样query标签"""
    random.seed(seed)
    return random.choices(base_labels, k=num_queries)

def process_dataset(dataset_name, data_dir, output_dir, num_attrs=4, num_values=3):
    """处理一个数据集"""
    print(f"\nProcessing {dataset_name}...")

    base_file = f"{data_dir}/{dataset_name}/{dataset_name}_base.fvecs"
    # 优先使用_equal.fvecs
    query_file = f"{data_dir}/{dataset_name}/{dataset_name}_query_equal.fvecs"
    if not os.path.exists(query_file):
        query_file = f"{data_dir}/{dataset_name}/{dataset_name}_query.fvecs"

    if not os.path.exists(base_file):
        print(f"  Base file not found: {base_file}")
        return
    if not os.path.exists(query_file):
        print(f"  Query file not found")
        return

    num_base = load_fvecs_count(base_file)
    num_queries = load_fvecs_count(query_file)

    print(f"  Base vectors: {num_base}, Query vectors: {num_queries}")
    print(f"  Using query file: {query_file}")

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 生成base标签（使用与NHQ相同的seed保持一致）
    print(f"  Generating base labels (attrs={num_attrs}, values={num_values})...")
    base_labels = generate_synthetic_labels(num_base, num_attrs, num_values, seed=42)
    save_labels_comma(base_labels, os.path.join(dataset_dir, 'label_CAPS_base.txt'), num_attrs)

    # 生成query标签
    print(f"  Generating {num_queries} query labels...")
    query_labels = generate_query_labels(base_labels, num_queries, seed=123)
    save_labels_comma(query_labels, os.path.join(dataset_dir, 'label_CAPS_query.txt'), num_attrs)

def main():
    # 检测运行环境
    if os.path.exists('/home/remote/u7905817'):
        # 服务器
        data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
        output_dir = '/home/remote/u7905817/benchmarks/discrete/CAPS/synthetic_labels'
    else:
        # 本地
        data_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/data'
        output_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/CAPS/synthetic_labels'

    datasets = ['arxiv', 'yfcc']

    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        process_dataset(dataset, data_dir, output_dir)

    print("\nDone! Now run generate_caps_groundtruth.py")

if __name__ == '__main__':
    main()
