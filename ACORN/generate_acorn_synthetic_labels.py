#!/usr/bin/env python3
"""
为ACORN生成Fixed-EQ合成标签（与NHQ/CAPS相同的4属性×3值）

ACORN标签格式：逗号分隔，无header（与CAPS相同）
"""

import os
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
    """生成合成标签

    ACORN使用set<int>存储标签，会去重。所以需要把位置相关的属性值转换为唯一标签ID：
    attr_i = v → label_id = i * num_values + v

    例如 4属性×3值:
    - 原始: 2,0,0,2 (attr0=2, attr1=0, attr2=0, attr3=2)
    - 转换: 2,3,6,11 (0*3+2, 1*3+0, 2*3+0, 3*3+2)

    这样每个属性组合都有唯一的标签集合。
    """
    random.seed(seed)
    labels = []
    for _ in range(num_points):
        attr_values = [random.randint(0, num_values - 1) for _ in range(num_attrs)]
        # 转换为唯一标签ID
        label_ids = [i * num_values + v for i, v in enumerate(attr_values)]
        labels.append([str(lid) for lid in label_ids])
    return labels

def save_labels_comma(labels, output_file):
    """保存标签（逗号分隔，无header）"""
    with open(output_file, 'w') as f:
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

    # 生成base标签（使用与NHQ/CAPS相同的seed保持一致）
    print(f"  Generating base labels (attrs={num_attrs}, values={num_values})...")
    base_labels = generate_synthetic_labels(num_base, num_attrs, num_values, seed=42)
    save_labels_comma(base_labels, os.path.join(dataset_dir, 'label_base_synthetic.txt'))

    # 生成query标签
    print(f"  Generating {num_queries} query labels...")
    query_labels = generate_query_labels(base_labels, num_queries, seed=123)
    save_labels_comma(query_labels, os.path.join(dataset_dir, 'label_query_synthetic.txt'))

def main():
    if os.path.exists('/home/remote/u7905817'):
        data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
        output_dir = '/home/remote/u7905817/benchmarks/discrete/ACORN/synthetic_labels'
    else:
        data_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/data'
        output_dir = '/Users/a1-6/workspace/discrete-filteredANN-benchmark/ACORN/synthetic_labels'

    datasets = ['arxiv', 'yfcc']

    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        process_dataset(dataset, data_dir, output_dir)

    print("\nDone! Now run generate_acorn_groundtruth.py")

if __name__ == '__main__':
    main()
