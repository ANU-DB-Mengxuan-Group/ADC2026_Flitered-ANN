#!/usr/bin/env python3
"""
Build label mapping by comparing original labels with DiskANN index labels.
"""

import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_labels', required=True, help='Original label_base.txt')
    parser.add_argument('--index_labels', required=True, help='DiskANN index labels file')
    parser.add_argument('--query_labels', required=True, help='Query labels file')
    parser.add_argument('--output', required=True, help='Output converted query labels')
    args = parser.parse_args()

    # Load both label files
    original = []
    with open(args.original_labels) as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(set(int(x) for x in line.split(',')))

    index = []
    with open(args.index_labels) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(set(int(x) for x in line.split(',')))

    print(f"Loaded {len(original)} original labels, {len(index)} index labels")

    # Build mapping by finding co-occurrences
    # For each original label, find which internal labels always appear with it
    orig_to_internal = {}
    internal_to_orig = {}

    # Track co-occurrences
    orig_label_to_points = defaultdict(set)
    internal_label_to_points = defaultdict(set)

    for i, (orig_set, idx_set) in enumerate(zip(original, index)):
        for label in orig_set:
            orig_label_to_points[label].add(i)
        for label in idx_set:
            internal_label_to_points[label].add(i)

    # Find mappings: original label X maps to internal Y if they appear in exactly the same points
    for orig_label, orig_points in orig_label_to_points.items():
        for internal_label, internal_points in internal_label_to_points.items():
            if orig_points == internal_points:
                if orig_label not in orig_to_internal:
                    orig_to_internal[orig_label] = internal_label
                    internal_to_orig[internal_label] = orig_label
                break

    print(f"Found {len(orig_to_internal)} label mappings")

    # Show some mappings
    print("\nSample mappings (original -> internal):")
    for orig in sorted(orig_to_internal.keys())[:20]:
        print(f"  {orig} -> {orig_to_internal[orig]}")

    # Convert query labels
    query_labels = []
    with open(args.query_labels) as f:
        for line in f:
            line = line.strip()
            if line:
                query_labels.append([int(x) for x in line.split(',')])

    # Convert
    converted = []
    unmapped = set()
    for q_labels in query_labels:
        new_labels = []
        for label in q_labels:
            if label in orig_to_internal:
                new_labels.append(orig_to_internal[label])
            else:
                unmapped.add(label)
                new_labels.append(label)
        converted.append(new_labels)

    if unmapped:
        print(f"\nWarning: {len(unmapped)} query labels not mapped: {sorted(unmapped)[:20]}")

    # Write output
    with open(args.output, 'w') as f:
        for labels in converted:
            f.write(','.join(map(str, labels)) + '\n')

    print(f"\nWritten converted labels to {args.output}")

    # Show first 10
    print("\nFirst 10 conversions:")
    for i in range(min(10, len(query_labels))):
        print(f"  {query_labels[i]} -> {converted[i]}")

if __name__ == '__main__':
    main()
