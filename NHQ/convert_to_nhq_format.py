#!/usr/bin/env python3
"""
将变长标签数据转换为NHQ需要的固定列格式

原始格式 (label_base.txt):
7,19
9,25
5,23,27

NHQ格式 (label_NHQ_base.txt):
132687 76    # 点数 最大标签数
7 19 -1 -1 ... -1   # pad到76列
9 25 -1 -1 ... -1
5 23 27 -1 ... -1
"""

import sys
import os

def convert_labels(input_file, output_file):
    # 读取所有标签
    all_labels = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels = [int(x) for x in line.split(',')]
            all_labels.append(labels)
    
    num_points = len(all_labels)
    max_labels = max(len(labels) for labels in all_labels)
    
    print(f"Points: {num_points}, Max labels per point: {max_labels}")
    
    # 写入NHQ格式
    with open(output_file, 'w') as f:
        f.write(f"{num_points} {max_labels}\n")
        for labels in all_labels:
            # pad到max_labels长度
            padded = labels + [-1] * (max_labels - len(labels))
            f.write(' '.join(str(x) for x in padded) + '\n')
    
    print(f"Written to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)
    
    convert_labels(sys.argv[1], sys.argv[2])
