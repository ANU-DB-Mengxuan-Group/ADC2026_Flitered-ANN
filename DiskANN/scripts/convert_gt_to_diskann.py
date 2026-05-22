#!/usr/bin/env python3
"""
Convert UNG ground truth format to DiskANN format.

UNG format: num_queries * K pairs of (int32, float32) without header
DiskANN format: header (npts, dim as int32) + npts*dim ids + npts*dim distances
"""

import numpy as np
import struct
import sys
import os

def read_ung_gt(filename, num_queries, K):
    """Read UNG format ground truth"""
    with open(filename, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
        gt_data = gt_data.reshape((num_queries, K))

    indices = gt_data['idx'].copy()
    distances = gt_data['dist'].copy()

    return indices, distances

def get_fvecs_count(filename):
    """Get vector count from fvecs file"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return 0, 0
    dim = fv.view(np.int32)[0]
    n = fv.size // (dim + 1)
    return n, dim

def write_diskann_gt(filename, ids, distances):
    """Write DiskANN format ground truth"""
    num_queries, K = ids.shape

    with open(filename, 'wb') as f:
        # Header: npts, dim as int32
        f.write(struct.pack('ii', num_queries, K))
        # IDs as uint32
        ids.astype(np.uint32).tofile(f)
        # Distances as float32
        distances.astype(np.float32).tofile(f)

    print(f"Written DiskANN GT to {filename}: {num_queries} queries, K={K}")

def main():
    if len(sys.argv) < 5:
        print("Usage: convert_gt_to_diskann.py <ung_gt.bin> <query.fvecs> <K> <output.bin>")
        print("  or: convert_gt_to_diskann.py <ung_gt.bin> <num_queries> <K> <output.bin>")
        sys.exit(1)

    ung_gt_file = sys.argv[1]
    K = int(sys.argv[3])
    output_file = sys.argv[4]

    # Try to get num_queries from fvecs file or direct number
    try:
        num_queries = int(sys.argv[2])
    except ValueError:
        query_file = sys.argv[2]
        num_queries, _ = get_fvecs_count(query_file)

    print(f"Converting {ung_gt_file} ({num_queries} queries, K={K}) to DiskANN format...")

    ids, dists = read_ung_gt(ung_gt_file, num_queries, K)
    write_diskann_gt(output_file, ids, dists)

    print("Done!")

if __name__ == '__main__':
    main()
