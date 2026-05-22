#!/usr/bin/env python3
"""
为NHQ创建等长标签数据

方案：将每个点的标签列表pad到最大长度，用 'PAD' 填充
这样NHQ可以正常处理，PAD与PAD匹配不会影响距离计算
"""

import os
import sys

def convert_labels_to_equal_length(input_file, output_file, max_len=None):
    """转换标签文件为等长格式"""
    # 读取所有标签
    all_labels = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels = line.split(',')
            all_labels.append(labels)
    
    num_points = len(all_labels)
    if max_len is None:
        max_len = max(len(labels) for labels in all_labels)
    
    print(f"  Points: {num_points}, Max labels: {max_len}")
    
    # 写入等长格式
    with open(output_file, 'w') as f:
        f.write(f"{num_points} {max_len}\n")
        for labels in all_labels:
            # pad到max_len长度，用PAD填充
            padded = labels + ['PAD'] * (max_len - len(labels))
            f.write(' '.join(padded) + '\n')
    
    return max_len

def process_dataset(dataset_name, data_dir, output_dir):
    """处理一个数据集"""
    print(f"\nProcessing {dataset_name}...")
    
    base_label_file = os.path.join(data_dir, dataset_name, 'label_base.txt')
    
    # 找到所有query label文件
    query_files = []
    for scenario in ['equal', 'and', 'or']:
        query_file = os.path.join(data_dir, dataset_name, f'{dataset_name}_query_{scenario}.txt')
        if os.path.exists(query_file):
            query_files.append((scenario, query_file))
    
    if not os.path.exists(base_label_file):
        print(f"  Base label file not found: {base_label_file}")
        return
    
    # 创建输出目录
    out_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)
    
    # 先处理base，获取max_len
    out_base = os.path.join(out_dataset_dir, 'label_NHQ_base.txt')
    print(f"  Converting base labels...")
    max_len = convert_labels_to_equal_length(base_label_file, out_base)
    
    # 处理query files，使用相同的max_len
    for scenario, query_file in query_files:
        out_query = os.path.join(out_dataset_dir, f'label_NHQ_query_{scenario}.txt')
        print(f"  Converting query labels ({scenario})...")
        convert_labels_to_equal_length(query_file, out_query, max_len)
    
    # 也创建一个通用的query文件（用equal）
    equal_query = os.path.join(data_dir, dataset_name, f'{dataset_name}_query_equal.txt')
    if os.path.exists(equal_query):
        out_query = os.path.join(out_dataset_dir, 'label_NHQ_query.txt')
        print(f"  Creating default query labels...")
        convert_labels_to_equal_length(equal_query, out_query, max_len)

def main():
    data_dir = '/home/remote/u7905817/benchmarks/datasets/discrete'
    output_dir = '/home/remote/u7905817/benchmarks/discrete/equal_length_experiment'
    
    datasets = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        process_dataset(dataset, data_dir, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
