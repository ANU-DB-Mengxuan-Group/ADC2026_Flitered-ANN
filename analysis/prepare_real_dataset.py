#!/usr/bin/env python3
"""
Prepare real-label text classification datasets for filtered ANN benchmarking.

Downloads a HuggingFace text classification dataset, generates embeddings
using sentence-transformers, assigns multi-labels via centroid-distance
soft assignment, and generates queries + brute-force ground truth.

Supported datasets:
  - dbpedia14: DBpedia ontology, 14 categories, 560K items
  - agnews:    AG News, 4 categories, 120K items

Usage:
    # DBpedia-14 (100K subset, 768d)
    python analysis/prepare_real_dataset.py \
        --dataset dbpedia14 \
        --output_dir ~/benchmarks/datasets/discrete/dbpedia14 \
        --name dbpedia14 \
        --max_items 100000

    # AG News (full 120K, 768d)
    python analysis/prepare_real_dataset.py \
        --dataset agnews \
        --output_dir ~/benchmarks/datasets/discrete/agnews4 \
        --name agnews4

Requirements:
    pip install datasets sentence-transformers
"""

import argparse
import json
import struct
import os
import numpy as np
from collections import Counter, defaultdict


# ── Dataset configs ────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "dbpedia14": {
        "hf_name": "fancyzhx/dbpedia_14",
        "split": "train",
        "text_col": "content",
        "label_col": "label",
        "label_names": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building",
            "NaturalPlace", "Village", "Animal", "Plant",
            "Album", "Film", "WrittenWork",
        ],
    },
    "agnews": {
        "hf_name": "fancyzhx/ag_news",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "label_names": ["World", "Sports", "Business", "SciTech"],
    },
    "yahoo": {
        "hf_name": "community-datasets/yahoo_answers_topics",
        "split": "train",
        "text_col": "best_answer",
        "label_col": "topic",
        "label_names": [
            "Society&Culture", "Science&Mathematics", "Health",
            "Education&Reference", "Computers&Internet", "Sports",
            "Business&Finance", "Entertainment&Music",
            "Family&Relationships", "Politics&Government",
        ],
    },
}


# ── I/O utilities ─────────────────────────────────────────────────────

def write_fvecs(fname, vecs):
    with open(fname, 'wb') as f:
        for vec in vecs:
            d = len(vec)
            f.write(struct.pack('i', d))
            f.write(vec.astype(np.float32).tobytes())


def assign_multi_labels(vectors, primary_labels, n_groups,
                        max_extra=2, threshold=1.5):
    """Soft label assignment via centroid distance (same as hm21)."""
    n = len(vectors)

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
        dists = np.linalg.norm(batch[:, None, :] - centroids[None, :, :], axis=2)

        for j in range(end - start):
            i = start + j
            own_dist = dists[j, primary_labels[i]]
            item_labels = [primary_labels[i]]

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

    # Adaptive noise: 10% of median vector norm
    norms = np.linalg.norm(vectors[:min(n, 10000)], axis=1)
    noise_scale = float(np.median(norms)) * 0.1
    print(f"  Query noise scale: {noise_scale:.4f} (10% of median norm {np.median(norms):.4f})")

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
            q_vec = vectors[qi] + rng.randn(vectors.shape[1]).astype(np.float32) * noise_scale
            vec_labels = labels[qi]

            if scenario == 'equal':
                q_labels = vec_labels
                candidates = [i for i in range(n)
                              if i != qi and sorted(labels[i]) == sorted(q_labels)]

            elif scenario == 'and':
                if len(vec_labels) < 2:
                    continue
                q_labels = sorted(rng.choice(
                    vec_labels, size=2, replace=False).tolist())
                sets = [set(label_to_vecs.get(l, [])) for l in q_labels]
                candidates = list(sets[0].intersection(*sets[1:]) - {qi})

            elif scenario == 'or':
                num_q = rng.randint(2, min(4, len(all_labels) + 1))
                q_labels = sorted(rng.choice(
                    all_labels, size=num_q, replace=False).tolist())
                candidate_set = set()
                for l in q_labels:
                    candidate_set.update(label_to_vecs.get(l, []))
                candidate_set.discard(qi)
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


def save_dataset(output_dir, name, vectors, labels, query_results, k,
                 label_names, source):
    """Save in benchmark-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    base_fvecs = os.path.join(output_dir, f'{name}_base.fvecs')
    print(f"Writing {base_fvecs}...")
    write_fvecs(base_fvecs, vectors)

    label_file = os.path.join(output_dir, 'label_base.txt')
    with open(label_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

    unique_labels_file = os.path.join(output_dir, 'unique_labels_base.txt')
    with open(unique_labels_file, 'w') as f:
        for lbls in labels:
            f.write(','.join(str(l) for l in lbls) + '\n')

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

        # GT text format
        gt_file = os.path.join(output_dir, f'{name}_gt_{scenario}.txt')
        with open(gt_file, 'w') as f:
            for ids in data['gt_ids']:
                f.write(' '.join(str(i) for i in ids) + '\n')

        # GT binary format
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
        f.write(f"source: {source}\n")
        f.write(f"n_vectors: {len(vectors)}\n")
        f.write(f"dim: {vectors.shape[1]}\n")
        f.write(f"label_cardinality: {len(all_label_ids)}\n")
        f.write(f"avg_labels_per_vector: {avg_labels:.2f}\n")
        f.write(f"k: {k}\n")
        for sc in ['and', 'or', 'equal']:
            nq = len(query_results[sc]['gt_ids'])
            f.write(f"n_queries_{sc}: {nq}\n")

    # Label mapping
    mapping_file = os.path.join(output_dir, 'label_mapping.txt')
    with open(mapping_file, 'w') as f:
        for i, name_ in enumerate(label_names):
            f.write(f"{i}\t{name_}\n")

    print(f"\nSaved to {output_dir}")
    print(f"  Vectors: {len(vectors)} x {vectors.shape[1]}")
    print(f"  Label cardinality: {len(all_label_ids)}")
    print(f"  Avg labels/vector: {avg_labels:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare real-label datasets for filtered ANN benchmarking')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to prepare')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--name', default=None, help='Dataset name (default: same as --dataset)')
    parser.add_argument('--model', default='all-mpnet-base-v2',
                        help='Sentence-transformers model for embedding (default: all-mpnet-base-v2, 768d)')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Max items to use (subsample if dataset is larger)')
    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Centroid distance threshold for soft label assignment')
    parser.add_argument('--max_extra', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for embedding generation')
    parser.add_argument('--save-embeddings', type=str, default=None,
                        help='Save embeddings+labels to .npz and exit (Phase 1, local)')
    parser.add_argument('--load-embeddings', type=str, default=None,
                        help='Load pre-computed embeddings from .npz (Phase 2, cluster)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.name is None:
        args.name = args.dataset

    cfg = DATASET_CONFIGS[args.dataset]

    # ── Phase 2 shortcut: load pre-computed embeddings ──────────────
    if args.load_embeddings:
        print("=" * 60)
        print(f"Loading pre-computed embeddings from {args.load_embeddings}")
        print("=" * 60)
        data = np.load(args.load_embeddings, allow_pickle=True)
        embeddings = data['embeddings'].astype(np.float32)
        primary_labels = data['labels'].tolist()
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Labels: {len(primary_labels)} items, {len(set(primary_labels))} classes")
    else:
        # ── Phase 1: Download + embed ──────────────────────────────────
        print("=" * 60)
        print(f"Step 1: Downloading {cfg['hf_name']}...")
        print("=" * 60)
        from datasets import load_dataset
        ds = load_dataset(cfg['hf_name'], split=cfg['split'])
        print(f"  {len(ds)} items, columns: {ds.column_names}")

        texts = ds[cfg['text_col']]
        raw_labels = ds[cfg['label_col']]

        # Subsample if needed
        rng = np.random.RandomState(args.seed)
        if args.max_items and len(texts) > args.max_items:
            print(f"\n  Subsampling {args.max_items} from {len(texts)}...")
            n_per_class = args.max_items // len(cfg['label_names'])
            indices = []
            for c in range(len(cfg['label_names'])):
                class_idx = [i for i, l in enumerate(raw_labels) if l == c]
                chosen = rng.choice(class_idx, size=min(n_per_class, len(class_idx)),
                                    replace=False)
                indices.extend(chosen.tolist())
            # Fill remaining
            remaining = args.max_items - len(indices)
            if remaining > 0:
                all_idx = set(range(len(texts))) - set(indices)
                extra = rng.choice(list(all_idx), size=remaining, replace=False)
                indices.extend(extra.tolist())
            rng.shuffle(indices)
            indices = indices[:args.max_items]

            texts = [texts[i] for i in indices]
            raw_labels = [raw_labels[i] for i in indices]
            print(f"  Subsampled to {len(texts)} items")

        primary_labels = list(raw_labels)

        print(f"\n  Label distribution:")
        counts = Counter(primary_labels)
        for i, name in enumerate(cfg['label_names']):
            print(f"    {i:2d}: {name} ({counts.get(i, 0)})")

        # Step 2: Generate embeddings
        print("\n" + "=" * 60)
        print(f"Step 2: Generating embeddings with {args.model}...")
        print("=" * 60)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        max_len = 512
        texts_truncated = [t[:max_len * 4] for t in texts]

        embeddings = model.encode(
            texts_truncated,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        print(f"  Embeddings shape: {embeddings.shape}")

        # Phase 1 exit: save embeddings and stop
        if args.save_embeddings:
            np.savez_compressed(args.save_embeddings,
                                embeddings=embeddings,
                                labels=np.array(primary_labels))
            print(f"\n  Saved to {args.save_embeddings}")
            print(f"  Transfer to cluster, then run with --load-embeddings")
            return

    # Step 3: Multi-label assignment
    print("\n" + "=" * 60)
    print("Step 3: Multi-label assignment...")
    print("=" * 60)
    labels = assign_multi_labels(
        embeddings, primary_labels, len(cfg['label_names']),
        max_extra=args.max_extra, threshold=args.threshold)

    # Step 4: Generate queries + GT
    print("\n" + "=" * 60)
    print("Step 4: Generating queries and ground truth...")
    print("=" * 60)
    query_results = generate_queries_and_gt(
        embeddings, labels, args.num_queries, args.k, seed=args.seed + 1)

    # Step 5: Save
    print("\n" + "=" * 60)
    print("Step 5: Saving dataset...")
    print("=" * 60)
    save_dataset(args.output_dir, args.name, embeddings, labels,
                 query_results, args.k, cfg['label_names'],
                 source=cfg['hf_name'])


if __name__ == '__main__':
    main()
