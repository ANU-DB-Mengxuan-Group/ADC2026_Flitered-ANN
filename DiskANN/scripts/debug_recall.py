#!/usr/bin/env python3
"""
Debug script to compare search results with ground truth.
"""

import numpy as np
import struct
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='DiskANN result file (idx_uint32.bin)')
    parser.add_argument('--gt', required=True, help='Ground truth file (UNG format)')
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()

    # Read search results
    with open(args.results, 'rb') as f:
        n, d = struct.unpack('<ii', f.read(8))
        results = np.frombuffer(f.read(n * d * 4), dtype=np.uint32).reshape(n, d)

    print(f"Search results: {n} queries, {d} neighbors each")
    print(f"Results range: min={results.min()}, max={results.max()}")
    print(f"First 5 queries results:")
    print(results[:5])
    print()

    # Read ground truth (UNG format)
    with open(args.gt, 'rb') as f:
        gt_data = np.frombuffer(f.read(n * args.K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
    gt = gt_data['idx'].reshape(n, args.K)

    print(f"Ground truth: {n} queries, {args.K} neighbors each")
    print(f"GT range: min={gt.min()}, max={gt.max()}")
    print(f"First 5 queries GT:")
    print(gt[:5])
    print()

    # Compute recall for first few queries
    print("Per-query recall (first 10 queries):")
    total_recall = 0
    valid_queries = 0
    for i in range(min(10, n)):
        gt_set = set(gt[i])
        gt_set.discard(0xFFFFFFFF)  # Remove -1
        if len(gt_set) == 0:
            print(f"  Query {i}: GT empty, skipping")
            continue

        res_set = set(results[i])
        intersection = len(gt_set & res_set)
        recall = intersection / len(gt_set)
        print(f"  Query {i}: recall={recall:.2f}, intersection={intersection}, GT={gt_set}, Res={res_set}")
        total_recall += recall
        valid_queries += 1

    if valid_queries > 0:
        print(f"\nAverage recall (first {valid_queries} valid queries): {total_recall/valid_queries:.4f}")

    # Overall recall
    total = 0
    count = 0
    for i in range(n):
        gt_set = set(gt[i])
        gt_set.discard(0xFFFFFFFF)
        if len(gt_set) == 0:
            continue
        res_set = set(results[i])
        total += len(gt_set & res_set) / len(gt_set)
        count += 1

    print(f"\nOverall recall@{args.K}: {total/count:.4f} ({count} valid queries)")

if __name__ == '__main__':
    main()
