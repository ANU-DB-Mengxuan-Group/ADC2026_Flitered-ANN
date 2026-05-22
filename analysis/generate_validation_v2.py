#!/usr/bin/env python3
"""
Generate large-scale validation datasets for routing rule evaluation.

Creates synthetic-label datasets at dimensions matching the training set range
(192d, 512d, 768d) with 800K vectors each.

Datasets:
  1. synth_192d: 800K, 192d, LID≈25, card=200  (matches yfcc dimension)
  2. synth_512d: 800K, 512d, LID≈35, card=30   (matches LAION dimension+card)
  3. synth_768d_hc: 800K, 768d, LID≈25, card=1000 (high cardinality)
  4. synth_10k: 800K, 192d, LID≈23, card=10000 (high cardinality)
  5. synth_4k: 800K, 768d, LID≈25, card=4000   (matches arxiv/ytb range)
  6. synth_100k: 800K, 192d, LID≈23, card=100000 (approaches yfcc's 181K)

For real-label datasets (yahoo800k, dbpedia560k), use prepare_real_dataset.py.

Usage (on cluster):
    cd ~/benchmarks/discrete
    python analysis/generate_validation_v2.py --dataset synth_192d
    python analysis/generate_validation_v2.py --dataset synth_512d
    python analysis/generate_validation_v2.py --dataset synth_768d_hc
    python analysis/generate_validation_v2.py --dataset all

Requirements: numpy only (no GPU needed)
"""

import argparse
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from generate_synthetic_labels import (
    generate_labels, generate_queries_and_gt, save_dataset, write_fvecs
)
from generate_lid_datasets import generate_controlled_lid_vectors


DATASET_CONFIGS = {
    "synth_192d": {
        "n_vectors": 800_000,
        "ambient_dim": 192,
        "intrinsic_dim": 25,      # LID ≈ 25 (similar to yfcc=23)
        "label_cardinality": 200,
        "min_labels": 1,
        "max_labels": 5,
        "seed": 2001,
    },
    "synth_512d": {
        "n_vectors": 800_000,
        "ambient_dim": 512,
        "intrinsic_dim": 35,      # LID ≈ 35 (similar to LAION=36.3)
        "label_cardinality": 30,
        "min_labels": 1,
        "max_labels": 3,
        "seed": 2002,
    },
    "synth_768d_hc": {
        "n_vectors": 800_000,
        "ambient_dim": 768,
        "intrinsic_dim": 25,      # LID ≈ 25 (similar to arxiv=25.5)
        "label_cardinality": 1000,
        "min_labels": 1,
        "max_labels": 5,
        "seed": 2003,
    },
    "synth_10k": {
        "n_vectors": 800_000,
        "ambient_dim": 192,       # same as yfcc (192d)
        "intrinsic_dim": 23,      # LID ≈ 23 (same as yfcc=23)
        "label_cardinality": 10000,  # high cardinality
        "min_labels": 1,
        "max_labels": 8,
        "seed": 2004,
    },
    "synth_4k": {
        "n_vectors": 800_000,
        "ambient_dim": 768,       # same as arxiv (768d)
        "intrinsic_dim": 25,      # LID ≈ 25 (same as arxiv=25.5)
        "label_cardinality": 4000,   # matches arxiv(4231), ytb(3862)
        "min_labels": 1,
        "max_labels": 5,
        "seed": 2005,
    },
    "synth_100k": {
        "n_vectors": 800_000,
        "ambient_dim": 192,       # same as yfcc (192d)
        "intrinsic_dim": 23,      # LID ≈ 23 (same as yfcc=23)
        "label_cardinality": 100000, # approaches yfcc's 181K
        "min_labels": 1,
        "max_labels": 2,          # sparse like yfcc
        "seed": 2006,
    },
}

NUM_QUERIES = 1000
K = 10


def generate_dataset(name, cfg, output_root, num_queries=NUM_QUERIES, k=K):
    output_dir = os.path.join(output_root, name)

    gt_check = os.path.join(output_dir, f"{name}_gt_and.txt")
    if os.path.exists(gt_check):
        print(f"\n=== SKIP {name} (already exists at {output_dir}) ===")
        return

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  N={cfg['n_vectors']}, D={cfg['ambient_dim']}, "
          f"LID≈{cfg['intrinsic_dim']}, card={cfg['label_cardinality']}")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. Generate vectors with controlled LID
    print(f"\n[1/4] Generating {cfg['n_vectors']} vectors "
          f"(intrinsic_dim={cfg['intrinsic_dim']} -> ambient_dim={cfg['ambient_dim']})...")
    vectors = generate_controlled_lid_vectors(
        cfg['n_vectors'], cfg['intrinsic_dim'], cfg['ambient_dim'],
        seed=cfg['seed']
    )
    print(f"  Shape: {vectors.shape}, dtype: {vectors.dtype}")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"  Norm range: [{norms.min():.1f}, {norms.max():.1f}], "
          f"median: {np.median(norms):.1f}")

    # 2. Generate labels
    print(f"\n[2/4] Generating labels (cardinality={cfg['label_cardinality']})...")
    labels = generate_labels(
        cfg['n_vectors'], cfg['label_cardinality'],
        labels_per_vector=(cfg['min_labels'], cfg['max_labels']),
        seed=cfg['seed'] + 1
    )
    avg_labels = np.mean([len(l) for l in labels])
    print(f"  Avg labels/vector: {avg_labels:.2f}")

    # 3. Generate queries and GT
    print(f"\n[3/4] Generating queries and ground truth (k={k})...")
    query_results = generate_queries_and_gt(
        vectors, labels, num_queries, k, seed=cfg['seed'] + 2
    )

    # 4. Save
    print(f"\n[4/4] Saving to {output_dir}...")
    save_dataset(output_dir, name, vectors, labels, query_results, k)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate large-scale validation datasets')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='Which dataset to generate (or "all")')
    parser.add_argument('--output_root',
                        default=os.path.expanduser("~/benchmarks/datasets/discrete"),
                        help='Root output directory')
    parser.add_argument('--num_queries', type=int, default=NUM_QUERIES)
    parser.add_argument('--k', type=int, default=K)
    args = parser.parse_args()

    num_queries = args.num_queries
    k = args.k

    if args.dataset == 'all':
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    for name in datasets:
        generate_dataset(name, DATASET_CONFIGS[name], args.output_root,
                         num_queries=num_queries, k=k)

    print(f"\n{'='*60}")
    print("All datasets generated!")
    print(f"Location: {args.output_root}/")
    for name in datasets:
        cfg = DATASET_CONFIGS[name]
        print(f"  {name}: {cfg['n_vectors']}×{cfg['ambient_dim']}d, "
              f"LID≈{cfg['intrinsic_dim']}, card={cfg['label_cardinality']}")


if __name__ == "__main__":
    main()
