#!/usr/bin/env python3
"""
Analyze selectivity of containment queries.
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_labels', required=True, help='Base labels file')
    parser.add_argument('--query_labels', required=True, help='Query labels file')
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to analyze')
    args = parser.parse_args()

    # Load base labels
    base_labels = []
    with open(args.base_labels) as f:
        for line in f:
            line = line.strip()
            if line:
                labels = set(int(x) for x in line.split(','))
                base_labels.append(labels)

    print(f"Loaded {len(base_labels)} base points")

    # Load query labels
    query_labels = []
    with open(args.query_labels) as f:
        for line in f:
            line = line.strip()
            if line:
                labels = set(int(x) for x in line.split(','))
                query_labels.append(labels)

    print(f"Loaded {len(query_labels)} queries")
    print()

    # Analyze each query
    print(f"{'Query':<8} {'Labels':<20} {'#Match':<10} {'Selectivity':<12} {'First 5 matches'}")
    print("-" * 80)

    total_matches = 0
    for i in range(min(args.num_queries, len(query_labels))):
        q_labels = query_labels[i]

        # Find matching base points (containment: query_labels ⊆ base_labels)
        matches = [j for j, base in enumerate(base_labels) if q_labels.issubset(base)]

        selectivity = len(matches) / len(base_labels) * 100
        total_matches += len(matches)

        labels_str = ','.join(map(str, sorted(q_labels)))
        matches_str = ','.join(map(str, matches[:5]))

        print(f"{i:<8} {labels_str:<20} {len(matches):<10} {selectivity:<12.4f}% {matches_str}")

    print()
    print(f"Average matches per query: {total_matches / min(args.num_queries, len(query_labels)):.1f}")

    # Check overlap with ground truth
    print()
    print("Checking if ground truth matches are valid...")

if __name__ == '__main__':
    main()
