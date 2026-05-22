"""
Generate synthetic labels for real vector datasets.

Takes real vectors (e.g., arxiv-for-fanns) and generates:
- Multi-label assignments with controlled cardinality
- Query labels for AND/OR/Equality scenarios
- Ground truth (brute-force k-NN among matching vectors)

Output format matches the benchmark convention:
- label_base.txt: one line per vector, comma-separated label IDs
- {name}_query_{scenario}.txt: one line per query, comma-separated query labels
- {name}_query_{scenario}.fvecs: query vectors
- {name}_gt_{scenario}.txt: one line per query, space-separated GT IDs (top-k)
- {name}_base.fvecs: symlink or copy of source vectors

Usage:
    python generate_synthetic_labels.py \
        --source_vectors /path/to/vectors.fvecs \
        --output_dir /path/to/output/ \
        --name synth200 \
        --label_cardinality 200 \
        --num_queries 1000 \
        --k 10
"""

import numpy as np
import struct
import os
import argparse
from collections import Counter


def read_fvecs(fname, max_n=None):
    """Read fvecs file format."""
    vectors = []
    with open(fname, 'rb') as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            d = struct.unpack('i', buf)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            if len(vec) < d:
                break
            vectors.append(vec)
            if max_n and len(vectors) >= max_n:
                break
    return np.array(vectors)


def write_fvecs(fname, vecs):
    """Write fvecs file format."""
    with open(fname, 'wb') as f:
        for vec in vecs:
            d = len(vec)
            f.write(struct.pack('i', d))
            f.write(vec.astype(np.float32).tobytes())


def generate_labels(n, label_cardinality, labels_per_vector=(1, 5), seed=42):
    """
    Generate multi-label assignments.

    Each vector gets between labels_per_vector[0] and labels_per_vector[1] labels,
    drawn from [0, label_cardinality).

    Uses Zipf-like distribution for label frequency (more realistic than uniform).
    """
    rng = np.random.RandomState(seed)

    # Zipf-like label probabilities (some labels are more common)
    ranks = np.arange(1, label_cardinality + 1)
    probs = 1.0 / (ranks ** 0.8)  # mild Zipf (exponent 0.8)
    probs /= probs.sum()

    labels = []
    for i in range(n):
        num_labels = rng.randint(labels_per_vector[0], labels_per_vector[1] + 1)
        chosen = rng.choice(label_cardinality, size=num_labels, replace=False, p=probs)
        labels.append(sorted(chosen.tolist()))

    return labels


def generate_queries_and_gt(vectors, labels, num_queries, k, seed=42):
    """
    Generate queries for AND/OR/Equality scenarios with brute-force ground truth.

    Returns dict with keys: 'and', 'or', 'equal'
    Each value is dict with: 'query_vecs', 'query_labels', 'gt_ids'
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)

    # Adaptive noise: 10% of median vector norm
    norms = np.linalg.norm(vectors[:min(n, 10000)], axis=1)
    noise_scale = float(np.median(norms)) * 0.1
    print(f"  Query noise scale: {noise_scale:.4f} (10% of median norm {np.median(norms):.4f})")

    # Build label-to-vectors index
    label_to_vecs = {}
    for i, lbls in enumerate(labels):
        for l in lbls:
            if l not in label_to_vecs:
                label_to_vecs[l] = []
            label_to_vecs[l].append(i)

    all_labels = sorted(label_to_vecs.keys())

    results = {}

    for scenario in ['and', 'or', 'equal']:
        query_vecs = []
        query_labels_list = []
        gt_ids = []

        attempts = 0
        while len(query_vecs) < num_queries and attempts < num_queries * 10:
            attempts += 1

            # Pick a random vector as query seed
            qi = rng.randint(n)
            q_vec = vectors[qi] + rng.randn(vectors.shape[1]).astype(np.float32) * noise_scale
            vec_labels = labels[qi]

            if scenario == 'equal':
                # Query labels = exact label set of the seed vector
                q_labels = vec_labels
                # Match: vectors with exactly this label set (exclude seed)
                candidates = []
                for i in range(n):
                    if i != qi and sorted(labels[i]) == sorted(q_labels):
                        candidates.append(i)

            elif scenario == 'and':
                # Query labels = 2 labels from the seed vector
                if len(vec_labels) < 2:
                    continue
                q_labels = sorted(rng.choice(vec_labels, size=2, replace=False).tolist())
                # Match: vectors containing ALL query labels (exclude seed)
                sets = [set(label_to_vecs.get(l, [])) for l in q_labels]
                candidates = list(sets[0].intersection(*sets[1:]) - {qi})

            elif scenario == 'or':
                # Query labels = 2-3 random labels (not necessarily from seed)
                num_q = rng.randint(2, 4)
                q_labels = sorted(rng.choice(all_labels, size=num_q, replace=False).tolist())
                # Match: vectors containing ANY query label (exclude seed)
                candidate_set = set()
                for l in q_labels:
                    candidate_set.update(label_to_vecs.get(l, []))
                candidate_set.discard(qi)
                candidates = list(candidate_set)

            if len(candidates) < k:
                continue

            # Brute-force k-NN among candidates
            cand_vecs = vectors[candidates]
            dists = np.linalg.norm(cand_vecs - q_vec, axis=1)
            topk_idx = np.argsort(dists)[:k]
            topk_ids = [candidates[j] for j in topk_idx]

            query_vecs.append(q_vec)
            query_labels_list.append(q_labels)
            gt_ids.append(topk_ids)

        print(f"  {scenario}: generated {len(query_vecs)} queries (attempts: {attempts})")

        results[scenario] = {
            'query_vecs': np.array(query_vecs),
            'query_labels': query_labels_list,
            'gt_ids': gt_ids,
        }

    return results


def save_dataset(output_dir, name, vectors, labels, query_results, k):
    """Save dataset in benchmark-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    # Base vectors
    base_fvecs = os.path.join(output_dir, f'{name}_base.fvecs')
    if not os.path.exists(base_fvecs):
        print(f"Writing {base_fvecs}...")
        write_fvecs(base_fvecs, vectors)

    # Base labels
    label_file = os.path.join(output_dir, 'label_base.txt')
    print(f"Writing {label_file}...")
    with open(label_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

    # Unique labels (for some methods)
    unique_labels_file = os.path.join(output_dir, 'unique_labels_base.txt')
    with open(unique_labels_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

    # Per-scenario files
    for scenario, data in query_results.items():
        # Query vectors
        qvec_file = os.path.join(output_dir, f'{name}_query_{scenario}.fvecs')
        print(f"Writing {qvec_file}...")
        write_fvecs(qvec_file, data['query_vecs'])

        # Query labels
        qlabel_file = os.path.join(output_dir, f'{name}_query_{scenario}.txt')
        with open(qlabel_file, 'w') as f:
            for lbls in data['query_labels']:
                f.write(','.join(str(l) for l in lbls) + '\n')

        # Ground truth (space-separated IDs)
        gt_file = os.path.join(output_dir, f'{name}_gt_{scenario}.txt')
        with open(gt_file, 'w') as f:
            for ids in data['gt_ids']:
                f.write(' '.join(str(i) for i in ids) + '\n')

        # Ground truth binary (for DiskANN etc.)
        # Format: nq (uint32), k (uint32), then nq*k uint32 IDs
        gt_bin_file = os.path.join(output_dir, f'{name}_gt_{scenario}.bin')
        nq = len(data['gt_ids'])
        with open(gt_bin_file, 'wb') as f:
            f.write(struct.pack('II', nq, k))
            for ids in data['gt_ids']:
                for id_ in ids:
                    f.write(struct.pack('I', id_))

    # Dataset info
    info_file = os.path.join(output_dir, 'dataset_info.txt')
    all_label_ids = set()
    for lbls in labels:
        all_label_ids.update(lbls)

    avg_labels = np.mean([len(l) for l in labels])

    with open(info_file, 'w') as f:
        f.write(f"name: {name}\n")
        f.write(f"n_vectors: {len(vectors)}\n")
        f.write(f"dim: {vectors.shape[1]}\n")
        f.write(f"label_cardinality: {len(all_label_ids)}\n")
        f.write(f"avg_labels_per_vector: {avg_labels:.2f}\n")
        f.write(f"n_queries_per_scenario: {len(query_results['and']['gt_ids'])}\n")
        f.write(f"k: {k}\n")
        for scenario in ['and', 'or', 'equal']:
            f.write(f"n_queries_{scenario}: {len(query_results[scenario]['gt_ids'])}\n")

    print(f"\nDataset saved to {output_dir}")
    print(f"  Vectors: {len(vectors)} x {vectors.shape[1]}")
    print(f"  Label cardinality: {len(all_label_ids)}")
    print(f"  Avg labels/vector: {avg_labels:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic labels for real vectors')
    parser.add_argument('--source_vectors', required=True, help='Path to source vectors (.fvecs or .npy)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--name', required=True, help='Dataset name prefix')
    parser.add_argument('--label_cardinality', type=int, default=200, help='Number of unique labels')
    parser.add_argument('--min_labels', type=int, default=1, help='Min labels per vector')
    parser.add_argument('--max_labels', type=int, default=5, help='Max labels per vector')
    parser.add_argument('--num_queries', type=int, default=1000, help='Queries per scenario')
    parser.add_argument('--k', type=int, default=10, help='Top-k for ground truth')
    parser.add_argument('--max_vectors', type=int, default=None, help='Limit number of vectors')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load vectors
    print(f"Loading vectors from {args.source_vectors}...")
    if args.source_vectors.endswith('.npy'):
        vectors = np.load(args.source_vectors)
    else:
        vectors = read_fvecs(args.source_vectors, max_n=args.max_vectors)

    if args.max_vectors and len(vectors) > args.max_vectors:
        vectors = vectors[:args.max_vectors]

    print(f"  Loaded {len(vectors)} vectors, dim={vectors.shape[1]}")

    # Generate labels
    print(f"Generating labels (cardinality={args.label_cardinality})...")
    labels = generate_labels(
        len(vectors), args.label_cardinality,
        labels_per_vector=(args.min_labels, args.max_labels),
        seed=args.seed
    )

    # Generate queries and ground truth
    print(f"Generating queries and ground truth (k={args.k})...")
    query_results = generate_queries_and_gt(
        vectors, labels, args.num_queries, args.k, seed=args.seed + 1
    )

    # Save
    save_dataset(args.output_dir, args.name, vectors, labels, query_results, args.k)


if __name__ == '__main__':
    main()
