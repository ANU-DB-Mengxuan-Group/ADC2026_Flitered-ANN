#!/usr/bin/env python3
"""
Convert H&M e-commerce dataset for filtered ANN benchmarking.

Source: Qdrant/hm_ecommerce_products
- 105,100 products with 2048d embeddings
- garment_group_name: 21 unique categories (real semantic labels)

Supports two input modes:
  1. Local files: --vectors_npy + --payloads_jsonl (from existing download)
  2. HuggingFace download: automatic if local files not specified

Multi-label assignment:
  Each item keeps its true garment_group label. Additional labels assigned
  for nearby group centroids (soft assignment), enabling AND/OR/Equal queries.

Usage:
    # From existing local files:
    python analysis/download_hm_dataset.py \
        --vectors_npy ~/benchmarks/datasets/discrete/hnm/vectors.npy \
        --payloads_jsonl ~/benchmarks/datasets/discrete/hnm/payloads.jsonl \
        --output_dir ~/benchmarks/datasets/discrete/hm21 \
        --name hm21

    # Or download from HuggingFace:
    python analysis/download_hm_dataset.py \
        --output_dir ~/benchmarks/datasets/discrete/hm21 \
        --name hm21
"""

import argparse
import json
import struct
import os
import numpy as np
from collections import Counter, defaultdict


def write_fvecs(fname, vecs):
    """Write fvecs file format."""
    with open(fname, 'wb') as f:
        for vec in vecs:
            d = len(vec)
            f.write(struct.pack('i', d))
            f.write(vec.astype(np.float32).tobytes())


def load_local_hm_data(vectors_npy, payloads_jsonl):
    """Load H&M data from local npy + jsonl files."""
    print(f"Loading vectors from {vectors_npy}...")
    embeddings = np.load(vectors_npy).astype(np.float32)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading payloads from {payloads_jsonl}...")
    garment_groups = []
    with open(payloads_jsonl) as f:
        for line in f:
            payload = json.loads(line)
            garment_groups.append(payload['garment_group_name'])

    assert len(garment_groups) == len(embeddings), \
        f"Mismatch: {len(embeddings)} vectors vs {len(garment_groups)} payloads"

    unique_groups = sorted(set(garment_groups))
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    primary_labels = [group_to_id[g] for g in garment_groups]

    print(f"  {len(unique_groups)} garment groups:")
    counts = Counter(garment_groups)
    for g in unique_groups:
        print(f"    {group_to_id[g]:2d}: {g} ({counts[g]})")

    return embeddings, primary_labels, unique_groups, group_to_id


def download_hm_data():
    """Download H&M dataset from HuggingFace. Returns embeddings + primary labels."""
    from datasets import load_dataset

    print("Downloading H&M dataset from HuggingFace...")
    ds = load_dataset("Qdrant/hm_ecommerce_products", split="train")
    print(f"  {len(ds)} items")

    print("Extracting embeddings...")
    embeddings = np.array(ds['dense_embedding'], dtype=np.float32)
    print(f"  Shape: {embeddings.shape}")

    garment_groups = ds['garment_group_name']
    unique_groups = sorted(set(garment_groups))
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    primary_labels = [group_to_id[g] for g in garment_groups]

    print(f"  {len(unique_groups)} garment groups:")
    counts = Counter(garment_groups)
    for g in unique_groups:
        print(f"    {group_to_id[g]:2d}: {g} ({counts[g]})")

    return embeddings, primary_labels, unique_groups, group_to_id


def assign_multi_labels(vectors, primary_labels, n_groups,
                        max_extra=2, threshold=1.5, seed=42):
    """
    Assign multiple labels via centroid-distance soft assignment.

    Each item keeps its true garment_group label. Extra labels are added for
    group centroids within threshold * distance_to_own_centroid. This models
    items near category boundaries belonging to multiple groups.
    """
    n = len(vectors)

    # Per-group centroids
    centroids = np.zeros((n_groups, vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(n_groups)
    for i in range(n):
        centroids[primary_labels[i]] += vectors[i]
        counts[primary_labels[i]] += 1
    for g in range(n_groups):
        if counts[g] > 0:
            centroids[g] /= counts[g]

    print(f"  Soft label assignment (threshold={threshold}, max_extra={max_extra})...")

    labels = []
    extra_count = 0
    batch_size = 5000

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = vectors[start:end]

        # (batch, n_groups) distances to all centroids
        dists = np.linalg.norm(batch[:, None, :] - centroids[None, :, :], axis=2)

        for j in range(end - start):
            i = start + j
            own_dist = dists[j, primary_labels[i]]

            item_labels = [primary_labels[i]]
            # Nearest other centroids
            other = [(g, dists[j, g]) for g in range(n_groups)
                     if g != primary_labels[i]]
            other.sort(key=lambda x: x[1])

            for g, d in other[:max_extra]:
                if own_dist > 0 and d < threshold * own_dist:
                    item_labels.append(g)
                    extra_count += 1

            labels.append(sorted(item_labels))

    avg_labels = np.mean([len(l) for l in labels])
    print(f"  Avg labels/item: {avg_labels:.2f}")
    print(f"  Extra labels assigned: {extra_count} ({extra_count / n:.1%})")
    dist_counts = Counter(len(l) for l in labels)
    for k in sorted(dist_counts):
        print(f"    {k} label(s): {dist_counts[k]} ({dist_counts[k] / n:.1%})")

    return labels


def generate_queries_and_gt(vectors, labels, num_queries, k, seed=42):
    """Generate queries for AND/OR/Equal with brute-force ground truth."""
    rng = np.random.RandomState(seed)
    n = len(vectors)

    label_to_vecs = defaultdict(list)
    for i, lbls in enumerate(labels):
        for l in lbls:
            label_to_vecs[l].append(i)
    all_labels = sorted(label_to_vecs.keys())

    results = {}

    for scenario in ['and', 'or', 'equal']:
        query_vecs, query_labels_list, gt_ids = [], [], []
        attempts = 0

        while len(query_vecs) < num_queries and attempts < num_queries * 20:
            attempts += 1
            qi = rng.randint(n)
            q_vec = vectors[qi] + rng.randn(vectors.shape[1]).astype(np.float32) * 0.01
            vec_labels = labels[qi]

            if scenario == 'equal':
                q_labels = vec_labels
                candidates = [i for i in range(n)
                              if sorted(labels[i]) == sorted(q_labels)]

            elif scenario == 'and':
                if len(vec_labels) < 2:
                    continue
                q_labels = sorted(rng.choice(
                    vec_labels, size=2, replace=False).tolist())
                sets = [set(label_to_vecs.get(l, [])) for l in q_labels]
                candidates = list(sets[0].intersection(*sets[1:]))

            elif scenario == 'or':
                num_q = rng.randint(2, min(4, len(all_labels) + 1))
                q_labels = sorted(rng.choice(
                    all_labels, size=num_q, replace=False).tolist())
                candidate_set = set()
                for l in q_labels:
                    candidate_set.update(label_to_vecs.get(l, []))
                candidates = list(candidate_set)

            if len(candidates) < k:
                continue

            cand_vecs = vectors[candidates]
            dists = np.linalg.norm(cand_vecs - q_vec, axis=1)
            topk_idx = np.argsort(dists)[:k]
            topk_ids = [candidates[j] for j in topk_idx]

            query_vecs.append(q_vec)
            query_labels_list.append(q_labels)
            gt_ids.append(topk_ids)

        print(f"  {scenario}: {len(query_vecs)} queries (attempts: {attempts})")
        results[scenario] = {
            'query_vecs': (np.array(query_vecs) if query_vecs
                           else np.zeros((0, vectors.shape[1]))),
            'query_labels': query_labels_list,
            'gt_ids': gt_ids,
        }

    return results


def save_dataset(output_dir, name, vectors, labels, query_results, k):
    """Save in benchmark-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    # Base vectors (fvecs)
    base_fvecs = os.path.join(output_dir, f'{name}_base.fvecs')
    print(f"Writing {base_fvecs}...")
    write_fvecs(base_fvecs, vectors)

    # Base labels (0-based, comma-separated)
    label_file = os.path.join(output_dir, 'label_base.txt')
    with open(label_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

    unique_labels_file = os.path.join(output_dir, 'unique_labels_base.txt')
    with open(unique_labels_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

    # Per-scenario
    for scenario, data in query_results.items():
        if not data['gt_ids']:
            print(f"  SKIP {scenario}: no queries generated")
            continue

        qvec_file = os.path.join(output_dir, f'{name}_query_{scenario}.fvecs')
        write_fvecs(qvec_file, data['query_vecs'])

        qlabel_file = os.path.join(output_dir, f'{name}_query_{scenario}.txt')
        with open(qlabel_file, 'w') as f:
            for lbls in data['query_labels']:
                f.write(','.join(str(l) for l in lbls) + '\n')

        gt_file = os.path.join(output_dir, f'{name}_gt_{scenario}.txt')
        with open(gt_file, 'w') as f:
            for ids in data['gt_ids']:
                f.write(' '.join(str(i) for i in ids) + '\n')

        gt_bin_file = os.path.join(output_dir, f'{name}_gt_{scenario}.bin')
        nq = len(data['gt_ids'])
        with open(gt_bin_file, 'wb') as f:
            f.write(struct.pack('II', nq, k))
            for ids in data['gt_ids']:
                for id_ in ids:
                    f.write(struct.pack('I', id_))

    # Dataset info
    all_label_ids = set()
    for lbls in labels:
        all_label_ids.update(lbls)
    avg_labels = np.mean([len(l) for l in labels])

    info_file = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"name: {name}\n")
        f.write(f"source: Qdrant/hm_ecommerce_products (HuggingFace)\n")
        f.write(f"n_vectors: {len(vectors)}\n")
        f.write(f"dim: {vectors.shape[1]}\n")
        f.write(f"label_cardinality: {len(all_label_ids)}\n")
        f.write(f"avg_labels_per_vector: {avg_labels:.2f}\n")
        f.write(f"k: {k}\n")
        for sc in ['and', 'or', 'equal']:
            nq = len(query_results[sc]['gt_ids'])
            f.write(f"n_queries_{sc}: {nq}\n")

    print(f"\nSaved to {output_dir}")
    print(f"  Vectors: {len(vectors)} x {vectors.shape[1]}")
    print(f"  Label cardinality: {len(all_label_ids)}")
    print(f"  Avg labels/vector: {avg_labels:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert H&M dataset for filtered ANN benchmarking')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--name', default='hm21')
    parser.add_argument('--vectors_npy', default=None,
                        help='Path to vectors.npy (skip HuggingFace download)')
    parser.add_argument('--payloads_jsonl', default=None,
                        help='Path to payloads.jsonl (skip HuggingFace download)')
    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Centroid distance threshold for soft label assignment')
    parser.add_argument('--max_extra', type=int, default=2,
                        help='Max extra labels per item')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.vectors_npy and args.payloads_jsonl:
        embeddings, primary_labels, unique_groups, group_to_id = \
            load_local_hm_data(args.vectors_npy, args.payloads_jsonl)
    else:
        embeddings, primary_labels, unique_groups, group_to_id = download_hm_data()

    print("\nAssigning multi-labels...")
    labels = assign_multi_labels(
        embeddings, primary_labels, len(unique_groups),
        max_extra=args.max_extra, threshold=args.threshold, seed=args.seed)

    print("\nGenerating queries and ground truth...")
    query_results = generate_queries_and_gt(
        embeddings, labels, args.num_queries, args.k, seed=args.seed + 1)

    print("\nSaving dataset...")
    save_dataset(args.output_dir, args.name, embeddings, labels,
                 query_results, args.k)

    # Save label mapping for reference
    mapping_file = os.path.join(args.output_dir, 'label_mapping.txt')
    with open(mapping_file, 'w') as f:
        for group, id_ in sorted(group_to_id.items(), key=lambda x: x[1]):
            count = sum(1 for l in primary_labels if l == id_)
            f.write(f"{id_}\t{group}\t{count}\n")
    print(f"  Label mapping: {mapping_file}")


if __name__ == '__main__':
    main()
