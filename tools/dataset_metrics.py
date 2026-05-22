#!/usr/bin/env python3
"""
Compute dataset difficulty metrics for Filtered ANN experiments.

Metrics:
  1. LID  (Local Intrinsic Dimensionality) – MLE estimator, per-query average
  2. RC   (Relative Contrast) – D_kmax / D_1 ratio, per-query average
  3. Label statistics – cardinality, entropy, mean group size
  4. Distribution Factor – Sliced Wasserstein distance between per-label
     filtered subsets and the full dataset

Usage:
  python tools/dataset_metrics.py \
      --data-root /home/remote/u7905817/benchmarks/datasets/discrete \
      --dataset arxiv

  # Run all 6 datasets:
  for ds in arxiv yfcc LAION1M tripclick ytb_audio ytb_video; do
      python tools/dataset_metrics.py --data-root /path/to/discrete --dataset $ds
  done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# I/O helpers (reused from dataset_sperability.py)
# ---------------------------------------------------------------------------

def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = raw.view(np.int32)[0]
    if dim <= 0:
        raise ValueError(f"{path} invalid dimension: {dim}")
    vectors = raw.reshape(-1, dim + 1)[:, 1:]
    return vectors


def load_labels(path: Path) -> List[List[int]]:
    labels: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                labels.append([])
                continue
            labels.append([int(tok) for tok in line.split(",") if tok])
    return labels


# ---------------------------------------------------------------------------
# Metric 1: LID (Local Intrinsic Dimensionality)
#
# MLE estimator (Levina & Bickel 2005, Amsaleg et al. 2015):
#   For query x with k nearest neighbors at distances r_1 <= ... <= r_k:
#   LID(x) = -( (1/(k-1)) * sum_{i=1}^{k-1} log(r_i / r_k) )^{-1}
# ---------------------------------------------------------------------------

def compute_lid_batch(queries: np.ndarray, base: np.ndarray,
                      k: int = 100, batch_size: int = 256) -> np.ndarray:
    """Compute LID for each query point using brute-force KNN."""
    n_queries = queries.shape[0]
    lids = np.zeros(n_queries, dtype=np.float64)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = queries[start:end]  # (B, D)

        # Compute distances: (B, N)
        # Use chunked computation to avoid memory blow-up
        dists = _batched_l2_distances(batch, base)

        # Partial sort: get k smallest distances per query
        # np.partition is O(N) per row, much faster than full sort
        kth_indices = np.argpartition(dists, k, axis=1)[:, :k]  # (B, k)
        kth_dists = np.take_along_axis(dists, kth_indices, axis=1)  # (B, k)
        kth_dists.sort(axis=1)  # sort only k elements: (B, k)

        # MLE LID estimator
        # r_k = kth_dists[:, -1], r_i = kth_dists[:, :-1]
        r_k = kth_dists[:, -1:]  # (B, 1)
        r_i = kth_dists[:, 1:-1]  # (B, k-2), skip r_0=0 (self or very close)

        # Avoid log(0): clamp small values
        r_i = np.maximum(r_i, 1e-10)
        r_k = np.maximum(r_k, 1e-10)

        log_ratios = np.log(r_i / r_k)  # (B, k-2), all <= 0
        mean_log = log_ratios.mean(axis=1)  # (B,)

        # LID = -1 / mean(log(r_i/r_k))
        # mean_log is negative, so LID is positive
        valid = mean_log < -1e-10
        lid_batch = np.full(end - start, np.nan)
        lid_batch[valid] = -1.0 / mean_log[valid]

        lids[start:end] = lid_batch

        if (start // batch_size) % 10 == 0:
            print(f"  LID: {end}/{n_queries} queries processed", flush=True)

    return lids


def _batched_l2_distances(queries: np.ndarray, base: np.ndarray,
                          chunk_size: int = 50000) -> np.ndarray:
    """Compute L2 distances between queries and base, chunked to save memory."""
    n_q = queries.shape[0]
    n_b = base.shape[0]
    dists = np.empty((n_q, n_b), dtype=np.float32)

    # ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q·b
    q_sq = (queries ** 2).sum(axis=1, keepdims=True)  # (B, 1)

    for c_start in range(0, n_b, chunk_size):
        c_end = min(c_start + chunk_size, n_b)
        b_chunk = base[c_start:c_end]  # (C, D)
        b_sq = (b_chunk ** 2).sum(axis=1, keepdims=False)  # (C,)
        dot = queries @ b_chunk.T  # (B, C)
        dists[:, c_start:c_end] = np.sqrt(
            np.maximum(q_sq + b_sq[np.newaxis, :] - 2 * dot, 0)
        )

    return dists


# ---------------------------------------------------------------------------
# Metric 2: RC (Relative Contrast)
#
# RC(q) = D_max / D_min = dist(q, kth-NN) / dist(q, 1st-NN)
# High RC = easy to distinguish neighbors; Low RC = distance concentration
# ---------------------------------------------------------------------------

def compute_rc_batch(queries: np.ndarray, base: np.ndarray,
                     k: int = 100, batch_size: int = 256) -> np.ndarray:
    """Compute Relative Contrast for each query."""
    n_queries = queries.shape[0]
    rcs = np.zeros(n_queries, dtype=np.float64)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = queries[start:end]

        dists = _batched_l2_distances(batch, base)
        kth_indices = np.argpartition(dists, k, axis=1)[:, :k]
        kth_dists = np.take_along_axis(dists, kth_indices, axis=1)
        kth_dists.sort(axis=1)

        d_min = kth_dists[:, 1]  # skip index 0 (could be self)
        d_max = kth_dists[:, -1]

        d_min = np.maximum(d_min, 1e-10)
        rcs[start:end] = d_max / d_min

        if (start // batch_size) % 10 == 0:
            print(f"  RC: {end}/{n_queries} queries processed", flush=True)

    return rcs


# ---------------------------------------------------------------------------
# Metric 3: Label Statistics
# ---------------------------------------------------------------------------

def compute_label_stats(labels: List[List[int]]) -> dict:
    """Compute label cardinality, entropy, and group size stats."""
    # Unique individual label values
    all_labels = []
    for lab_list in labels:
        all_labels.extend(lab_list)
    label_counts = Counter(all_labels)

    cardinality = len(label_counts)

    # Entropy of label frequency distribution
    freqs = np.array(list(label_counts.values()), dtype=np.float64)
    probs = freqs / freqs.sum()
    entropy = -np.sum(probs * np.log2(np.maximum(probs, 1e-15)))

    # Label combination stats (for Equality scenario)
    combo_counts = Counter(tuple(sorted(lab)) for lab in labels if lab)
    combo_sizes = np.array(list(combo_counts.values()))

    return {
        "label_cardinality": cardinality,
        "label_entropy": float(entropy),
        "num_unique_combos": len(combo_counts),
        "mean_combo_size": float(combo_sizes.mean()),
        "median_combo_size": float(np.median(combo_sizes)),
        "min_combo_size": int(combo_sizes.min()),
        "max_combo_size": int(combo_sizes.max()),
    }


# ---------------------------------------------------------------------------
# Metric 4: Distribution Factor (Sliced Wasserstein Distance)
#
# For each label l, compute the Sliced Wasserstein Distance between
# S_l = {vectors with label l} and the full dataset.
#
# Sliced Wasserstein: project both distributions onto random 1D directions,
# compute 1D Wasserstein for each, and average.
# ---------------------------------------------------------------------------

def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray,
                                 n_projections: int = 50,
                                 seed: int = 42) -> float:
    """Compute Sliced Wasserstein Distance between two point clouds.

    Projects onto random 1D directions and computes the average
    1D Wasserstein (earth mover's) distance.
    """
    rng = np.random.default_rng(seed)
    dim = X.shape[1]

    # Random unit vectors for projection
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total_wd = 0.0
    for d in directions:
        # Project onto 1D
        proj_x = X @ d  # (n_x,)
        proj_y = Y @ d  # (n_y,)

        # 1D Wasserstein = integral of |CDF_X - CDF_Y|
        # Efficient: sort both, then compare quantiles
        proj_x_sorted = np.sort(proj_x)
        proj_y_sorted = np.sort(proj_y)

        # Resample to same size for quantile comparison
        n_samples = min(len(proj_x_sorted), len(proj_y_sorted), 10000)
        quantiles = np.linspace(0, 1, n_samples, endpoint=False)
        qx = np.quantile(proj_x_sorted, quantiles)
        qy = np.quantile(proj_y_sorted, quantiles)

        total_wd += float(np.mean(np.abs(qx - qy)))

    return total_wd / n_projections


# --- DF: 单 label 计算 (供 multiprocessing.Pool 用) ---
def _compute_df_single_label(args):
    """Worker function: compute SWD for a single label."""
    l, idx, base, base_sample, base_sample_size, seed_l, n_total = args
    subset = base[idx]
    if len(subset) > base_sample_size:
        rng = np.random.default_rng(seed_l)
        sub_idx = rng.choice(len(subset), base_sample_size, replace=False)
        subset = subset[sub_idx]
    if len(subset) < 10:
        return None
    swd = sliced_wasserstein_distance(subset, base_sample, seed=seed_l)
    return {
        "label": int(l),
        "subset_size": len(idx),
        "selectivity": len(idx) / n_total,
        "swd": swd,
    }


def compute_distribution_factor(
    base: np.ndarray,
    labels: List[List[int]],
    max_labels: int = 50,
    base_sample_size: int = 50000,
    seed: int = 42,
    n_workers: int = None,
) -> dict:
    """Compute distribution factor for each label value.

    For each label l, measures how different the spatial distribution of
    vectors with label l is from the overall dataset distribution.

    并行化: per-label SWD 计算用 multiprocessing.Pool 并行 (n_workers 默认 = 全部核).
    """
    import multiprocessing as mp
    if n_workers is None:
        # 优先看 SLURM cgroup CPU 限制 (避免在共享节点上 over-subscribe)
        try:
            n_workers = len(os.sched_getaffinity(0))  # 进程实际可用的 CPU 数
        except (AttributeError, OSError):
            n_workers = mp.cpu_count()

    rng = np.random.default_rng(seed)

    # Sample from base for efficiency
    if base.shape[0] > base_sample_size:
        sample_idx = rng.choice(base.shape[0], base_sample_size, replace=False)
        base_sample = base[sample_idx]
    else:
        base_sample = base

    # Build label -> vector indices mapping
    label_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, lab_list in enumerate(labels):
        for l in lab_list:
            label_to_idx[l].append(i)

    # Sort labels by frequency, take top max_labels
    sorted_labels = sorted(label_to_idx.keys(),
                           key=lambda l: len(label_to_idx[l]), reverse=True)
    if len(sorted_labels) > max_labels:
        sorted_labels = sorted_labels[:max_labels]

    n_total = base.shape[0]
    print(f"  DF: 用 {min(n_workers, len(sorted_labels))} 进程并行 {len(sorted_labels)} labels", flush=True)

    # 并行任务参数
    tasks = []
    for i, l in enumerate(sorted_labels):
        # 不同 label 用不同 seed (避免重复 sampling pattern)
        tasks.append((l, label_to_idx[l], base, base_sample,
                      base_sample_size, seed + i, n_total))

    results = []
    if n_workers > 1 and len(sorted_labels) > 1:
        with mp.Pool(processes=n_workers) as pool:
            for r in pool.imap_unordered(_compute_df_single_label, tasks, chunksize=1):
                if r is None:
                    continue
                results.append(r)
                # 进度打印
                print(f"  DF: label={r['label']}, n={r['subset_size']}, SWD={r['swd']:.6f}", flush=True)
    else:
        # 单进程 fallback
        for t in tasks:
            r = _compute_df_single_label(t)
            if r is None:
                continue
            results.append(r)
            print(f"  DF: label={r['label']}, n={r['subset_size']}, SWD={r['swd']:.6f}", flush=True)

    # 按 label 排序保持输出稳定
    results.sort(key=lambda x: x["label"])

    swd_values = [r["swd"] for r in results]
    return {
        "per_label": results,
        "mean_swd": float(np.mean(swd_values)) if swd_values else 0.0,
        "std_swd": float(np.std(swd_values)) if swd_values else 0.0,
        "max_swd": float(np.max(swd_values)) if swd_values else 0.0,
        "min_swd": float(np.min(swd_values)) if swd_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset difficulty metrics (LID, RC, Label Stats, Distribution Factor)")
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to datasets/discrete directory")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g., arxiv, yfcc)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of neighbors for LID/RC (default: 100)")
    parser.add_argument("--query-sample", type=int, default=1000,
                        help="Number of query points to sample for LID/RC (default: 1000)")
    parser.add_argument("--base-sample", type=int, default=200000,
                        help="Max base vectors for LID/RC KNN (default: 200000). "
                             "Larger datasets are subsampled for speed.")
    parser.add_argument("--skip-lid-rc", action="store_true",
                        help="Skip LID/RC computation (slow for large datasets)")
    parser.add_argument("--skip-df", action="store_true",
                        help="Skip distribution factor computation")
    parser.add_argument("--df-workers", type=int, default=None,
                        help="DF 并行进程数 (默认 = mp.cpu_count())")
    parser.add_argument("--df-max-labels", type=int, default=None,
                        help="DF 最多处理多少 label (默认 50; 大数据集如 synth_768d_hc 1000 labels 可考虑减小)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file (default: {data-root}/{dataset}_metrics.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.data_root / args.dataset
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (args.data_root / f"{args.dataset}_metrics.json")

    # --- Load data ---
    base_path = dataset_dir / f"{args.dataset}_base.fvecs"
    if not base_path.is_file():
        print(f"Error: {base_path} not found", file=sys.stderr)
        sys.exit(1)

    label_path = dataset_dir / "label_base.txt"
    if not label_path.is_file():
        print(f"Error: {label_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.dataset} base vectors from {base_path} ...")
    t0 = time.time()
    base = read_fvecs(base_path)
    print(f"  Loaded {base.shape[0]} vectors, dim={base.shape[1]}, "
          f"took {time.time()-t0:.1f}s")

    print(f"Loading labels from {label_path} ...")
    labels = load_labels(label_path)
    assert len(labels) == base.shape[0], \
        f"Vector count {base.shape[0]} != label count {len(labels)}"

    results = {
        "dataset": args.dataset,
        "n_vectors": base.shape[0],
        "dimension": base.shape[1],
    }

    # --- Metric 3: Label Statistics (fast, always compute) ---
    print("\n--- Label Statistics ---")
    label_stats = compute_label_stats(labels)
    results["label_stats"] = label_stats
    for k, v in label_stats.items():
        print(f"  {k}: {v}")

    # --- Metric 1 & 2: LID and RC ---
    if not args.skip_lid_rc:
        # Sample query points from base (or use query file if available)
        query_path = dataset_dir / f"{args.dataset}_query.fvecs"
        if query_path.is_file():
            print(f"\nLoading query vectors from {query_path} ...")
            queries = read_fvecs(query_path)
            print(f"  Loaded {queries.shape[0]} query vectors")
        else:
            print(f"\nNo query file found, sampling from base vectors")
            rng = np.random.default_rng(args.seed)
            q_idx = rng.choice(base.shape[0], min(args.query_sample, base.shape[0]),
                               replace=False)
            queries = base[q_idx]

        # Subsample queries if too many
        if queries.shape[0] > args.query_sample:
            rng = np.random.default_rng(args.seed)
            q_idx = rng.choice(queries.shape[0], args.query_sample, replace=False)
            queries = queries[q_idx]

        # Subsample base for KNN to avoid O(query * N) blow-up on large datasets
        if base.shape[0] > args.base_sample:
            rng = np.random.default_rng(args.seed)
            b_idx = rng.choice(base.shape[0], args.base_sample, replace=False)
            base_for_knn = base[b_idx]
            print(f"  Base subsampled: {base.shape[0]:,} -> {args.base_sample:,} for KNN")
        else:
            base_for_knn = base

        print(f"\n--- LID (k={args.k}, {queries.shape[0]} queries, "
              f"base={base_for_knn.shape[0]:,}) ---")
        t0 = time.time()
        lids = compute_lid_batch(queries, base_for_knn, k=args.k)
        valid_lids = lids[~np.isnan(lids)]
        lid_time = time.time() - t0
        print(f"  Mean LID: {valid_lids.mean():.2f}")
        print(f"  Median LID: {np.median(valid_lids):.2f}")
        print(f"  Std LID: {valid_lids.std():.2f}")
        print(f"  Time: {lid_time:.1f}s")
        results["lid"] = {
            "mean": float(valid_lids.mean()),
            "median": float(np.median(valid_lids)),
            "std": float(valid_lids.std()),
            "min": float(valid_lids.min()),
            "max": float(valid_lids.max()),
            "k": args.k,
            "n_queries": len(valid_lids),
        }

        print(f"\n--- RC (k={args.k}, {queries.shape[0]} queries, "
              f"base={base_for_knn.shape[0]:,}) ---")
        t0 = time.time()
        rcs = compute_rc_batch(queries, base_for_knn, k=args.k)
        valid_rcs = rcs[rcs > 0]
        # RC is prone to outliers (D_min near 0), use median and trimmed mean
        rc_p5, rc_p95 = np.percentile(valid_rcs, [5, 95])
        trimmed = valid_rcs[(valid_rcs >= rc_p5) & (valid_rcs <= rc_p95)]
        rc_time = time.time() - t0
        print(f"  Median RC: {np.median(valid_rcs):.4f}")
        print(f"  Trimmed Mean RC (5-95%): {trimmed.mean():.4f}")
        print(f"  Time: {rc_time:.1f}s")
        results["rc"] = {
            "median": float(np.median(valid_rcs)),
            "trimmed_mean": float(trimmed.mean()),
            "p5": float(rc_p5),
            "p95": float(rc_p95),
            "k": args.k,
            "n_queries": len(valid_rcs),
        }
    else:
        print("\n--- LID/RC skipped ---")

    # --- Metric 4: Distribution Factor ---
    if not args.skip_df:
        print(f"\n--- Distribution Factor (Sliced Wasserstein) ---")
        t0 = time.time()
        df_kwargs = {"seed": args.seed, "n_workers": args.df_workers}
        if args.df_max_labels is not None:
            df_kwargs["max_labels"] = args.df_max_labels
        df_results = compute_distribution_factor(base, labels, **df_kwargs)
        df_time = time.time() - t0
        print(f"  Mean SWD: {df_results['mean_swd']:.6f}")
        print(f"  Std SWD: {df_results['std_swd']:.6f}")
        print(f"  Range: [{df_results['min_swd']:.6f}, {df_results['max_swd']:.6f}]")
        print(f"  Time: {df_time:.1f}s")
        results["distribution_factor"] = df_results
    else:
        print("\n--- Distribution Factor skipped ---")

    # --- Save ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # --- Summary table row ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.dataset}")
    print(f"{'='*60}")
    print(f"  N={results['n_vectors']:,}  dim={results['dimension']}")
    print(f"  Labels: {label_stats['label_cardinality']} unique values, "
          f"entropy={label_stats['label_entropy']:.2f} bits, "
          f"{label_stats['num_unique_combos']} unique combos")
    if "lid" in results:
        print(f"  LID: {results['lid']['mean']:.2f} ± {results['lid']['std']:.2f}")
    if "rc" in results:
        print(f"  RC:  median={results['rc']['median']:.4f}, "
              f"trimmed_mean={results['rc']['trimmed_mean']:.4f}")
    if "distribution_factor" in results:
        df = results["distribution_factor"]
        print(f"  DF (SWD): {df['mean_swd']:.6f} ± {df['std_swd']:.6f}")


if __name__ == "__main__":
    main()
