#!/usr/bin/env python3
"""
Check DiskANN label mapping and compare with query labels.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_prefix', required=True, help='DiskANN index prefix')
    parser.add_argument('--query_labels', required=True, help='Query labels file')
    args = parser.parse_args()

    # Check label mapping file
    map_file = f"{args.index_prefix}_labels_map.txt"
    if os.path.exists(map_file):
        print(f"=== Label mapping ({map_file}) ===")
        with open(map_file) as f:
            lines = f.readlines()
        print(f"Total mappings: {len(lines)}")
        print("First 20 mappings:")
        for line in lines[:20]:
            print(f"  {line.strip()}")
        print()

    # Check labels file
    labels_file = f"{args.index_prefix}_labels.txt"
    if os.path.exists(labels_file):
        print(f"=== Index labels ({labels_file}) ===")
        with open(labels_file) as f:
            lines = f.readlines()
        print(f"Total label entries: {len(lines)}")
        print("First 10 entries:")
        for i, line in enumerate(lines[:10]):
            print(f"  Point {i}: {line.strip()}")
        print()

    # Check formatted labels
    formatted_file = f"{args.index_prefix}_label_formatted.txt"
    if os.path.exists(formatted_file):
        print(f"=== Formatted labels ({formatted_file}) ===")
        with open(formatted_file) as f:
            lines = f.readlines()
        print(f"Total formatted entries: {len(lines)}")
        print("First 10 entries:")
        for i, line in enumerate(lines[:10]):
            print(f"  Point {i}: {line.strip()}")
        print()

    # Check query labels
    print(f"=== Query labels ({args.query_labels}) ===")
    with open(args.query_labels) as f:
        lines = f.readlines()
    print(f"Total queries: {len(lines)}")
    print("First 10 queries:")
    for i, line in enumerate(lines[:10]):
        print(f"  Query {i}: {line.strip()}")

if __name__ == '__main__':
    main()
