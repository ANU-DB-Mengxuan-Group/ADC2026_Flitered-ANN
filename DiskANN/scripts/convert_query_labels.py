#!/usr/bin/env python3
"""
Convert query labels from original label IDs to DiskANN internal IDs.
"""

import argparse

def load_label_map(map_file):
    """Load DiskANN label mapping: internal_id -> original_label"""
    internal_to_original = {}
    original_to_internal = {}

    with open(map_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Format: internal_id original_label
                internal_id = int(parts[0])
                original_label = int(parts[1])
                internal_to_original[internal_id] = original_label
                original_to_internal[original_label] = internal_id

    return internal_to_original, original_to_internal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', required=True, help='DiskANN labels_map.txt file')
    parser.add_argument('--query_labels', required=True, help='Original query labels file')
    parser.add_argument('--output', required=True, help='Output converted labels file')
    args = parser.parse_args()

    # Load label mapping
    internal_to_original, original_to_internal = load_label_map(args.map_file)
    print(f"Loaded {len(original_to_internal)} label mappings")

    # Show some mappings
    print("Sample mappings (original -> internal):")
    for i, (orig, internal) in enumerate(sorted(original_to_internal.items())[:10]):
        print(f"  {orig} -> {internal}")

    # Convert query labels
    converted_lines = []
    unmapped_labels = set()

    with open(args.query_labels) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                converted_lines.append("")
                continue

            original_labels = [int(x) for x in line.split(',')]
            internal_labels = []

            for label in original_labels:
                if label in original_to_internal:
                    internal_labels.append(original_to_internal[label])
                else:
                    unmapped_labels.add(label)
                    internal_labels.append(label)  # Keep original if not mapped

            converted_lines.append(','.join(map(str, internal_labels)))

    if unmapped_labels:
        print(f"Warning: {len(unmapped_labels)} labels not found in mapping: {sorted(unmapped_labels)[:20]}")

    # Write output
    with open(args.output, 'w') as f:
        for line in converted_lines:
            f.write(line + '\n')

    print(f"Converted {len(converted_lines)} queries to {args.output}")

    # Show first few converted
    print("\nFirst 10 converted queries:")
    with open(args.query_labels) as orig, open(args.output) as conv:
        orig_lines = orig.readlines()
        conv_lines = conv.readlines()
        for i in range(min(10, len(orig_lines))):
            print(f"  {orig_lines[i].strip()} -> {conv_lines[i].strip()}")

if __name__ == '__main__':
    main()
