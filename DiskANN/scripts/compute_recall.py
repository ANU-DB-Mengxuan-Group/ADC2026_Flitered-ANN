#!/usr/bin/env python3
"""
Compute recall by comparing DiskANN search results with ground truth.
"""

import numpy as np
import struct
import argparse
import os

def read_diskann_bin(filename):
    """Read DiskANN bin format: header (npts, dim) + data"""
    with open(filename, 'rb') as f:
        npts, dim = struct.unpack('<ii', f.read(8))
        data = np.frombuffer(f.read(npts * dim * 4), dtype=np.uint32).reshape(npts, dim)
    return data

def read_ung_gt(filename, num_queries, K):
    """Read UNG format ground truth"""
    with open(filename, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
    return gt_data['idx'].reshape(num_queries, K)

def compute_recall(results, gt, K):
    """Compute recall@K"""
    num_queries = results.shape[0]
    total_recall = 0

    for i in range(num_queries):
        gt_set = set(gt[i][:K])
        # Handle -1 or invalid entries in ground truth
        gt_set.discard(0xFFFFFFFF)  # -1 as uint32
        if len(gt_set) == 0:
            continue

        result_set = set(results[i][:K])
        result_set.discard(0xFFFFFFFF)

        intersection = len(gt_set & result_set)
        total_recall += intersection / len(gt_set)

    return total_recall / num_queries

def main():
    parser = argparse.ArgumentParser(description='Compute recall for DiskANN results')
    parser.add_argument('--results_prefix', required=True, help='Results file prefix (e.g., /tmp/arxiv_diskann_results)')
    parser.add_argument('--gt_file', required=True, help='Ground truth file (UNG format)')
    parser.add_argument('--K', type=int, default=10, help='K for recall@K')
    parser.add_argument('--L_values', type=str, default='10,20,30,40,50,60,70,80,90,100',
                        help='Comma-separated L values')
    args = parser.parse_args()

    L_values = [int(x) for x in args.L_values.split(',')]

    # Read first result to get num_queries
    first_result = read_diskann_bin(f"{args.results_prefix}_{L_values[0]}_idx_uint32.bin")
    num_queries = first_result.shape[0]

    # Read ground truth
    gt = read_ung_gt(args.gt_file, num_queries, args.K)

    print(f"{'L':>6} {'Recall@' + str(args.K):>12}")
    print("-" * 20)

    for L in L_values:
        result_file = f"{args.results_prefix}_{L}_idx_uint32.bin"
        if not os.path.exists(result_file):
            print(f"{L:>6} {'N/A':>12}")
            continue

        results = read_diskann_bin(result_file)
        recall = compute_recall(results, gt, args.K)
        print(f"{L:>6} {recall:>12.4f}")

if __name__ == '__main__':
    main()
