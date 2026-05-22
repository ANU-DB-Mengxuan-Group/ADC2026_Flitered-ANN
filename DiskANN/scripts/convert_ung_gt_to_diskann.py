#!/usr/bin/env python3
"""
Convert UNG ground truth format to DiskANN format.

UNG format: num_queries * K pairs of (int32, float32) without header
DiskANN format: header (npts, dim as int32) + npts*dim ids + npts*dim distances
"""

import numpy as np
import struct
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert UNG GT to DiskANN format')
    parser.add_argument('--ung_gt', required=True, help='Input UNG ground truth file')
    parser.add_argument('--query_bin', required=True, help='Query bin file (to get num_queries)')
    parser.add_argument('--output', required=True, help='Output DiskANN format GT file')
    parser.add_argument('--K', type=int, default=10, help='Number of neighbors (default: 10)')
    args = parser.parse_args()

    # Get num_queries from query bin file header
    with open(args.query_bin, 'rb') as f:
        num_queries = struct.unpack('<i', f.read(4))[0]

    print(f'Queries: {num_queries}, K: {args.K}')

    # Read UNG format: num_queries * K pairs of (uint32 idx, float32 dist)
    with open(args.ung_gt, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * args.K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
    gt_data = gt_data.reshape(num_queries, args.K)

    # Extract and make contiguous
    ids = np.ascontiguousarray(gt_data['idx']).astype(np.uint32)
    dists = np.ascontiguousarray(gt_data['dist']).astype(np.float32)

    # Write DiskANN format
    with open(args.output, 'wb') as f:
        # Header: npts, dim (both int32, little-endian)
        f.write(struct.pack('<ii', num_queries, args.K))
        # IDs as uint32 array
        ids.tofile(f)
        # Distances as float32 array
        dists.tofile(f)

    file_size = os.path.getsize(args.output)
    expected_size = 8 + num_queries * args.K * 4 * 2
    print(f'Written to {args.output}')
    print(f'File size: {file_size} bytes (expected: {expected_size})')

if __name__ == '__main__':
    main()
