#!/usr/bin/env python3
"""将文本格式的gt转换为ivecs二进制格式"""

import numpy as np
import os

def txt_to_ivecs(txt_file, ivecs_file):
    """转换txt格式gt到ivecs格式"""
    gt = []
    with open(txt_file, 'r') as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            gt.append(row)
    
    gt = np.array(gt, dtype=np.int32)
    n, k = gt.shape
    
    with open(ivecs_file, 'wb') as f:
        for i in range(n):
            np.array([k], dtype=np.int32).tofile(f)
            gt[i].tofile(f)
    
    print(f"Converted {txt_file} -> {ivecs_file} ({n} queries, k={k})")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = ['arxiv', 'yfcc']
    
    for dataset in datasets:
        txt_file = f"{base_dir}/synthetic_labels/{dataset}/gt_NHQ.txt"
        ivecs_file = f"{base_dir}/synthetic_labels/{dataset}/gt_NHQ.ivecs"
        
        if os.path.exists(txt_file):
            txt_to_ivecs(txt_file, ivecs_file)
        else:
            print(f"Not found: {txt_file}")

if __name__ == '__main__':
    main()
