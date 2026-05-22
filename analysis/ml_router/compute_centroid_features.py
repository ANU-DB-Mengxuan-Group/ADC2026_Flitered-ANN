#!/usr/bin/env python3
"""
Compute label centroid features for ML routing.

For each dataset, computes per-label centroids from base vectors,
then for each query computes distances from query vector to its
filter labels' centroids. Outputs a CSV that can be joined with
ml_training_data.csv or ml_validation_data.csv.

Usage:
  python analysis/compute_centroid_features.py \
      --data-root ~/benchmarks/datasets/discrete \
      --datasets arxiv yfcc LAION1M tripclick ytb_audio ytb_video \
      --output analysis/centroid_features_train.csv

  python analysis/compute_centroid_features.py \
      --data-root ~/benchmarks/datasets/discrete \
      --datasets synth200 arxiv_fanns_real lid50 lid80 lid100 lid120 lid150 \
                 synth5 synth30 synth100 hm21 \
      --output analysis/centroid_features_val.csv
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


SCENARIOS = ["and", "or", "equal"]


def read_fvecs(path: Path) -> np.ndarray:
    """Read .fvecs format file → (N, dim) float32 array."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 4:
        raise ValueError(f"File too small: {path}")
    dim = struct.unpack("<i", buf[:4])[0]
    row_bytes = 4 + dim * 4
    n = len(buf) // row_bytes
    arr = np.frombuffer(buf, dtype=np.float32).reshape(n, dim + 1)
    return arr[:, 1:].copy()


def load_labels_txt(path: Path) -> List[List[int]]:
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                labels.append([])
            else:
                labels.append([int(x) for x in line.split(",")])
    return labels


def compute_centroids(base: np.ndarray, labels: List[List[int]]) -> Dict[int, np.ndarray]:
    """Compute centroid for each label from base vectors."""
    label_vecs = defaultdict(list)
    for i, lab_list in enumerate(labels):
        for l in lab_list:
            label_vecs[l].append(i)

    centroids = {}
    for l, indices in label_vecs.items():
        idx = np.array(indices)
        centroids[l] = base[idx].mean(axis=0)

    return centroids


def compute_query_centroid_distances(
    query_vec: np.ndarray,
    query_labels: List[int],
    centroids: Dict[int, np.ndarray],
) -> dict:
    """Compute distance features from a query vector to its filter labels' centroids."""
    dists = []
    for l in query_labels:
        if l in centroids:
            d = np.linalg.norm(query_vec - centroids[l])
            dists.append(d)

    if not dists:
        return {
            "min_centroid_dist": 0.0,
            "max_centroid_dist": 0.0,
            "mean_centroid_dist": 0.0,
        }

    return {
        "min_centroid_dist": float(min(dists)),
        "max_centroid_dist": float(max(dists)),
        "mean_centroid_dist": float(np.mean(dists)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute label centroid distance features for ML routing")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fieldnames = [
        "query_id", "dataset", "scenario",
        "min_centroid_dist", "max_centroid_dist", "mean_centroid_dist",
    ]

    all_rows = []

    for ds in args.datasets:
        ds_dir = args.data_root / ds
        base_path = ds_dir / f"{ds}_base.fvecs"

        if not base_path.is_file():
            print(f"  WARNING: {base_path} not found, skipping {ds}")
            continue

        # Load base vectors
        print(f"\n{'='*60}")
        print(f"[{ds}] Loading base vectors ...")
        t0 = time.time()
        base = read_fvecs(base_path)
        print(f"  {base.shape[0]:,} vectors, dim={base.shape[1]}, {time.time()-t0:.1f}s")

        # Load base labels
        label_path = ds_dir / "label_base.txt"
        if not label_path.is_file():
            print(f"  WARNING: {label_path} not found, skipping {ds}")
            continue
        base_labels = load_labels_txt(label_path)

        # Compute centroids
        print(f"  Computing centroids ...")
        t0 = time.time()
        centroids = compute_centroids(base, base_labels)
        print(f"  {len(centroids)} label centroids computed, {time.time()-t0:.1f}s")

        # Process each scenario
        for scenario in SCENARIOS:
            # Load query vectors
            q_vec_path = ds_dir / f"{ds}_query_{scenario}.fvecs"
            if not q_vec_path.is_file():
                # Try alternative naming
                q_vec_path = ds_dir / f"query_{scenario}.fvecs"
            if not q_vec_path.is_file():
                print(f"  WARNING: query vectors not found for {ds}/{scenario}")
                continue

            # Load query labels
            q_label_path = ds_dir / f"label_query_{scenario}.txt"
            if not q_label_path.is_file():
                q_label_path = ds_dir / f"{ds}_query_{scenario}.txt"
            if not q_label_path.is_file():
                print(f"  WARNING: query labels not found for {ds}/{scenario}")
                continue

            query_vecs = read_fvecs(q_vec_path)
            query_labels = load_labels_txt(q_label_path)

            n_queries = min(len(query_vecs), len(query_labels))
            print(f"  {ds}/{scenario}: {n_queries} queries")

            for qid in range(n_queries):
                if not query_labels[qid]:
                    continue
                feats = compute_query_centroid_distances(
                    query_vecs[qid], query_labels[qid], centroids
                )
                row = {"query_id": qid, "dataset": ds, "scenario": scenario}
                row.update(feats)
                all_rows.append(row)

        # Free memory
        del base
        del centroids

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Written {len(all_rows)} rows to {args.output}")
    print(f"Datasets: {len(args.datasets)}, Scenarios: {len(SCENARIOS)}")


if __name__ == "__main__":
    main()
