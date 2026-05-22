#!/usr/bin/env python3
"""
Compute per-label Local Intrinsic Dimensionality (LID) for filtered ANN datasets.

For each label l with sufficient vectors, computes the LID of the sub-manifold
formed by vectors sharing that label.  The key output is the *correlation ratio*:

    correlation_ratio = mean(per_label_LID) / global_LID

  ≈ 1.0  → labels are spatially random (no clustering benefit for UNG)
  << 1.0 → same-label vectors cluster in space (UNG partitioning is effective)

Usage:
  python tools/per_label_lid.py \\
      --data-root ~/benchmarks/datasets/discrete \\
      --dataset arxiv \\
      --output analysis/per_label_lid/arxiv_per_label_lid.json

Reuses I/O and LID helpers from dataset_metrics.py in the same directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# --- Import helpers from sibling module ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_metrics import read_fvecs, load_labels, compute_lid_batch  # noqa: E402


def build_label_index(labels: List[List[int]]) -> Dict[int, List[int]]:
    """Map each individual label value to the list of vector indices carrying it."""
    label_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, lab_list in enumerate(labels):
        for l in lab_list:
            label_to_idx[l].append(i)
    return label_to_idx


def compute_per_label_lid(
    base: np.ndarray,
    label_to_idx: Dict[int, List[int]],
    k_default: int = 100,
    min_size: int = 200,
    max_labels: int = 500,
    query_sample: int = 500,
    max_subset: int = 50000,
    seed: int = 42,
) -> List[dict]:
    """Compute LID for the sub-manifold of each qualifying label.

    Parameters
    ----------
    base : (N, D) float32 array
    label_to_idx : mapping label -> list of vector indices
    k_default : k for KNN in LID estimator
    min_size : minimum subset size to include a label
    max_labels : cap on number of labels to process
    query_sample : number of query points sampled from each subset
    max_subset : if |S_l| > this, subsample the base for KNN
    seed : random seed
    """
    rng = np.random.default_rng(seed)

    # Filter labels by min_size, then take largest max_labels
    qualifying = {l: idx for l, idx in label_to_idx.items() if len(idx) >= min_size}
    sorted_labels = sorted(qualifying.keys(), key=lambda l: len(qualifying[l]), reverse=True)
    if len(sorted_labels) > max_labels:
        sorted_labels = sorted_labels[:max_labels]

    print(f"  Labels total: {len(label_to_idx)}, "
          f"qualifying (>= {min_size}): {len(qualifying)}, "
          f"processing: {len(sorted_labels)}")

    results = []
    for i, l in enumerate(sorted_labels):
        idx = np.array(qualifying[l])
        subset = base[idx]  # (|S_l|, D)

        # Subsample large subsets for KNN base
        if len(subset) > max_subset:
            sub_idx = rng.choice(len(subset), max_subset, replace=False)
            knn_base = subset[sub_idx]
        else:
            knn_base = subset

        # Adaptive k
        k = min(k_default, len(knn_base) // 5)
        if k < 5:
            continue  # too few points for reliable LID

        # Sample query points from the subset
        n_q = min(query_sample, len(subset))
        q_idx = rng.choice(len(subset), n_q, replace=False)
        queries = subset[q_idx]

        lids = compute_lid_batch(queries, knn_base, k=k, batch_size=256)
        valid = lids[~np.isnan(lids)]
        if len(valid) == 0:
            continue

        rec = {
            "label": int(l),
            "subset_size": len(idx),
            "k_used": k,
            "mean_lid": float(valid.mean()),
            "std_lid": float(valid.std()),
            "n_queries": int(len(valid)),
        }
        results.append(rec)

        if (i + 1) % 50 == 0 or (i + 1) == len(sorted_labels):
            print(f"  [{i+1}/{len(sorted_labels)}] label={l}, "
                  f"n={len(idx)}, k={k}, LID={valid.mean():.2f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-label LID and correlation ratio")
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to datasets/discrete directory")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g., arxiv, yfcc)")
    parser.add_argument("--k", type=int, default=100,
                        help="Default k for LID KNN (default: 100)")
    parser.add_argument("--min-size", type=int, default=200,
                        help="Minimum subset size per label (default: 200)")
    parser.add_argument("--max-labels", type=int, default=500,
                        help="Max number of labels to process (default: 500)")
    parser.add_argument("--query-sample", type=int, default=500,
                        help="Query points per label subset (default: 500)")
    parser.add_argument("--max-subset", type=int, default=50000,
                        help="Subsample large subsets for KNN (default: 50000)")
    parser.add_argument("--global-lid", type=float, default=None,
                        help="Pre-computed global LID (if omitted, computed from data)")
    parser.add_argument("--global-lid-queries", type=int, default=1000,
                        help="Query points for global LID computation (default: 1000)")
    parser.add_argument("--global-lid-base-sample", type=int, default=200000,
                        help="Base sample for global LID computation (default: 200000)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON (default: analysis/per_label_lid/{dataset}_per_label_lid.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.data_root / args.dataset
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} not found", file=sys.stderr)
        sys.exit(1)

    # --- Load data ---
    base_path = dataset_dir / f"{args.dataset}_base.fvecs"
    label_path = dataset_dir / "label_base.txt"
    for p in (base_path, label_path):
        if not p.is_file():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {args.dataset} base vectors from {base_path} ...")
    t0 = time.time()
    base = read_fvecs(base_path)
    print(f"  Loaded {base.shape[0]:,} vectors, dim={base.shape[1]}, "
          f"took {time.time()-t0:.1f}s")

    print(f"Loading labels from {label_path} ...")
    labels = load_labels(label_path)
    assert len(labels) == base.shape[0], \
        f"Vector count {base.shape[0]} != label count {len(labels)}"

    label_to_idx = build_label_index(labels)

    # --- Global LID ---
    if args.global_lid is not None:
        global_lid = args.global_lid
        print(f"\nUsing provided global LID: {global_lid:.2f}")
    else:
        print(f"\nComputing global LID (k={args.k}) ...")
        rng = np.random.default_rng(args.seed)

        # Sample base for KNN
        if base.shape[0] > args.global_lid_base_sample:
            b_idx = rng.choice(base.shape[0], args.global_lid_base_sample, replace=False)
            base_for_knn = base[b_idx]
        else:
            base_for_knn = base

        # Sample queries
        n_gq = min(args.global_lid_queries, base.shape[0])
        q_idx = rng.choice(base.shape[0], n_gq, replace=False)
        global_queries = base[q_idx]

        t0 = time.time()
        global_lids = compute_lid_batch(global_queries, base_for_knn, k=args.k)
        valid_global = global_lids[~np.isnan(global_lids)]
        global_lid = float(valid_global.mean())
        print(f"  Global LID: {global_lid:.2f} "
              f"(median {float(np.median(valid_global)):.2f}, "
              f"std {float(valid_global.std()):.2f}), "
              f"took {time.time()-t0:.1f}s")

    # --- Per-label LID ---
    print(f"\n--- Per-label LID ---")
    t0 = time.time()
    per_label_results = compute_per_label_lid(
        base, label_to_idx,
        k_default=args.k,
        min_size=args.min_size,
        max_labels=args.max_labels,
        query_sample=args.query_sample,
        max_subset=args.max_subset,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Computed {len(per_label_results)} labels in {elapsed:.1f}s")

    if not per_label_results:
        print("Warning: no labels qualified. Try lowering --min-size.", file=sys.stderr)
        sys.exit(1)

    # --- Aggregate ---
    lids_arr = np.array([r["mean_lid"] for r in per_label_results])
    sizes_arr = np.array([r["subset_size"] for r in per_label_results], dtype=np.float64)
    weights = sizes_arr / sizes_arr.sum()

    weighted_mean = float(np.sum(lids_arr * weights))
    unweighted_mean = float(lids_arr.mean())
    median_lid = float(np.median(lids_arr))
    std_lid = float(lids_arr.std())
    correlation_ratio = weighted_mean / global_lid if global_lid > 0 else float("nan")

    # --- Output ---
    output = {
        "dataset": args.dataset,
        "global_lid": global_lid,
        "min_subset_size": args.min_size,
        "k_default": args.k,
        "n_labels_total": len(label_to_idx),
        "n_labels_computed": len(per_label_results),
        "per_label": per_label_results,
        "summary": {
            "mean_per_label_lid": weighted_mean,
            "median_per_label_lid": median_lid,
            "std_per_label_lid": std_lid,
            "correlation_ratio": correlation_ratio,
            "unweighted_mean_per_label_lid": unweighted_mean,
        },
    }

    output_path = args.output
    if output_path is None:
        output_path = Path("analysis/per_label_lid") / f"{args.dataset}_per_label_lid.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.dataset}")
    print(f"{'='*60}")
    print(f"  Global LID:          {global_lid:.2f}")
    print(f"  Per-label LID (wt):  {weighted_mean:.2f}")
    print(f"  Per-label LID (med): {median_lid:.2f}")
    print(f"  Correlation ratio:   {correlation_ratio:.4f}")
    print(f"  Labels computed:     {len(per_label_results)} / {len(label_to_idx)}")
    print(f"  Time:                {elapsed:.1f}s")
    print(f"  Output:              {output_path}")


if __name__ == "__main__":
    main()
