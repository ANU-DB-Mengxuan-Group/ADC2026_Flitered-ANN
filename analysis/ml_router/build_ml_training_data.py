#!/usr/bin/env python3
"""
Build ML training dataset for routing experiment (single script, runs on cluster).

Does everything in one shot:
  1. Reads query label files → computes per-query features (selectivity, etc.)
  2. Reads *_metrics.json → loads dataset-level features (LID, RC, etc.)
  3. Reads perquery_recall CSV → determines best method per query
  4. Merges all into one training CSV

Output: each row = one query with all features + best_method label.

Usage:
  python analysis/build_ml_training_data.py \
      --data-root ~/benchmarks/datasets/discrete \
      --output analysis/ml_training_data.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


DATASETS = ["arxiv", "yfcc", "LAION1M", "tripclick", "ytb_audio", "ytb_video"]
SCENARIOS = ["and", "or", "equal"]

# Calibration curve from LID sweep (random labels baseline)
CALIB_A = 0.1006
CALIB_B = -0.0589


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_labels_txt(path: Path) -> List[List[int]]:
    """Load comma-separated label file."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                labels.append([])
            else:
                labels.append([int(x) for x in line.split(",")])
    return labels


def build_label_to_indices(labels: List[List[int]]) -> Dict[int, Set[int]]:
    """Inverted index: label → set of vector indices."""
    idx = defaultdict(set)
    for i, lab_list in enumerate(labels):
        for l in lab_list:
            idx[l].add(i)
    return idx


# ── Step 1: Per-query features ───────────────────────────────────────────────


def extract_perquery_features(
    data_root: Path,
    datasets: List[str],
    scenarios: List[str],
) -> Dict[tuple, dict]:
    """
    Compute per-query features from query/base label files.
    Returns dict keyed by (query_id, dataset, scenario).
    """
    result = {}

    for ds in datasets:
        ds_dir = data_root / ds
        label_base_path = ds_dir / "label_base.txt"
        if not label_base_path.is_file():
            print(f"  Warning: {label_base_path} not found, skipping {ds}")
            continue

        print(f"\n  [{ds}] Loading base labels ...")
        t0 = time.time()
        base_labels = load_labels_txt(label_base_path)
        n_base = len(base_labels)
        print(f"    {n_base:,} vectors, {time.time()-t0:.1f}s")

        # Build inverted index + label frequencies
        label_to_idx = build_label_to_indices(base_labels)
        label_freq = {l: len(idxs) / n_base for l, idxs in label_to_idx.items()}

        # Precompute label sets (for equality matching)
        base_label_sets = [set(ls) for ls in base_labels]

        # avg labels per vector
        avg_labels_per_vec = sum(len(ls) for ls in base_labels) / n_base

        for scenario in scenarios:
            # Find query label file
            q_path = ds_dir / f"label_query_{scenario}.txt"
            if not q_path.is_file():
                q_path = ds_dir / f"{ds}_query_{scenario}.txt"
            if not q_path.is_file():
                print(f"    Warning: query labels not found for {ds}/{scenario}")
                continue

            query_labels = load_labels_txt(q_path)
            print(f"    {ds}/{scenario}: {len(query_labels)} queries")

            for qid, q_labels in enumerate(query_labels):
                if not q_labels:
                    continue

                # Label frequency stats
                freqs = [label_freq.get(l, 0.0) for l in q_labels]

                # Selectivity
                if scenario == "and":
                    matching = label_to_idx.get(q_labels[0], set()).copy()
                    for l in q_labels[1:]:
                        matching &= label_to_idx.get(l, set())
                    sel = len(matching) / n_base
                elif scenario == "or":
                    matching = set()
                    for l in q_labels:
                        matching |= label_to_idx.get(l, set())
                    sel = len(matching) / n_base
                elif scenario == "equal":
                    q_set = set(q_labels)
                    rarest = min(q_labels, key=lambda l: len(label_to_idx.get(l, set())))
                    count = sum(1 for i in label_to_idx.get(rarest, set())
                                if base_label_sets[i] == q_set)
                    sel = count / n_base
                else:
                    sel = 0.0

                # Cooccurrence (fraction with ALL query labels)
                cooc_set = label_to_idx.get(q_labels[0], set()).copy()
                for l in q_labels[1:]:
                    cooc_set &= label_to_idx.get(l, set())
                cooc = len(cooc_set) / n_base

                result[(qid, ds, scenario)] = {
                    "query_label_count": len(q_labels),
                    "selectivity": sel,
                    "min_label_freq": min(freqs),
                    "max_label_freq": max(freqs),
                    "mean_label_freq": sum(freqs) / len(freqs),
                    "label_cooccurrence": cooc,
                    "avg_labels_per_vec": avg_labels_per_vec,
                }

    return result


# ── Step 2: Dataset-level features ───────────────────────────────────────────


def load_dataset_features(analysis_dir: Path) -> Dict[str, dict]:
    """Load dataset-level features from metrics JSON + per-label LID."""
    features = {}
    for ds in DATASETS:
        metrics_path = analysis_dir / f"{ds}_metrics.json"
        perlabel_path = analysis_dir / "per_label_lid" / f"{ds}_per_label_lid.json"

        if not metrics_path.is_file():
            print(f"  Warning: {metrics_path} not found")
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        # Per-label LID
        corr_ratio = None
        norm_ratio = None
        if perlabel_path.is_file():
            with open(perlabel_path) as f:
                p = json.load(f)
            corr_ratio = p["summary"]["correlation_ratio"]
            per_label = p["per_label"]
            if per_label:
                glid = p["global_lid"]
                sizes = np.array([r["subset_size"] for r in per_label], dtype=float)
                lids = np.array([r["mean_lid"] for r in per_label])
                raw = lids / glid
                exp = CALIB_A * np.log(sizes) + CALIB_B
                valid = exp > 0
                nr = np.where(valid, raw / exp, np.nan)
                weights = sizes / sizes.sum()
                norm_ratio = float(np.nansum(nr * weights))

        ls = m["label_stats"]
        features[ds] = {
            "N": m["n_vectors"],
            "dim": m["dimension"],
            "LID_mean": m["lid"]["mean"],
            "LID_median": m["lid"]["median"],
            "LID_std": m["lid"]["std"],
            "RC_median": m["rc"]["median"],
            "RC_trimmed_mean": m["rc"]["trimmed_mean"],
            "RC_p95": m["rc"]["p95"],
            "label_cardinality": ls["label_cardinality"],
            "label_entropy": ls["label_entropy"],
            "num_unique_combos": ls["num_unique_combos"],
            "DF_mean_swd": m["distribution_factor"]["mean_swd"],
            "correlation_ratio": corr_ratio,
            "normalized_ratio": norm_ratio,
        }

    return features


# ── Step 3: Per-query recall → best method ───────────────────────────────────


def load_perquery_recall(recall_path: Path, threshold: float = 0.7) -> Dict:
    """Load per-query recall, apply threshold, determine best method."""
    raw = {}
    with open(recall_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["query_id"]), row["dataset"], row["scenario"])
            method = row["method"]
            recall = float(row["recall_at_10"])
            if key not in raw:
                raw[key] = {}
            raw[key][method] = recall

    # Per-cell average recall for threshold filtering
    cell_recalls = defaultdict(list)
    for (qid, ds, sc), mr in raw.items():
        for method, recall in mr.items():
            cell_recalls[(ds, sc, method)].append(recall)
    cell_avg = {k: np.mean(v) for k, v in cell_recalls.items()}

    result = {}
    for (qid, ds, sc), method_recalls in raw.items():
        filtered = {m: r for m, r in method_recalls.items()
                    if cell_avg.get((ds, sc, m), 0) >= threshold}
        if not filtered:
            filtered = method_recalls
        best = max(filtered, key=filtered.get)
        result[(qid, ds, sc)] = {
            "best_method": best,
            "best_recall": filtered[best],
            "n_candidates": len(filtered),
        }

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build ML training data for routing (all-in-one)")
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to datasets/discrete directory")
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"),
                        help="Directory with metrics JSONs and recall CSVs")
    parser.add_argument("--recall-csv", type=Path,
                        default=Path("analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv"))
    parser.add_argument("--recall-threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/ml_training_data.csv"))
    args = parser.parse_args()

    # Step 1: Per-query features
    print("=" * 60)
    print("Step 1: Extracting per-query features")
    print("=" * 60)
    pq_features = extract_perquery_features(
        args.data_root, DATASETS, SCENARIOS)
    print(f"\n  Total: {len(pq_features)} query features extracted")

    # Step 2: Dataset-level features
    print("\n" + "=" * 60)
    print("Step 2: Loading dataset-level features")
    print("=" * 60)
    ds_features = load_dataset_features(args.analysis_dir)
    print(f"  {len(ds_features)} datasets loaded")

    # Step 3: Per-query recall
    print("\n" + "=" * 60)
    print("Step 3: Loading per-query recall")
    print("=" * 60)
    recall_data = load_perquery_recall(args.recall_csv, args.recall_threshold)
    print(f"  {len(recall_data)} query records loaded")

    # Step 4: Merge
    print("\n" + "=" * 60)
    print("Step 4: Merging into training data")
    print("=" * 60)

    fieldnames = [
        "query_id", "dataset", "scenario",
        # Per-query features
        "query_label_count", "selectivity",
        "min_label_freq", "max_label_freq", "mean_label_freq",
        "label_cooccurrence", "avg_labels_per_vec",
        # Dataset-level features
        "N", "dim",
        "LID_mean", "LID_median", "LID_std",
        "RC_median", "RC_trimmed_mean", "RC_p95",
        "label_cardinality", "label_entropy", "num_unique_combos",
        "DF_mean_swd",
        "correlation_ratio", "normalized_ratio",
        # Target
        "best_method", "best_recall", "n_candidates",
    ]

    rows = []
    missing_pq = 0
    for (qid, ds, sc), recall_info in sorted(recall_data.items()):
        if ds not in ds_features:
            continue
        if (qid, ds, sc) not in pq_features:
            missing_pq += 1
            continue

        row = {"query_id": qid, "dataset": ds, "scenario": sc}
        row.update(pq_features[(qid, ds, sc)])
        row.update(ds_features[ds])
        row.update(recall_info)
        rows.append(row)

    if missing_pq > 0:
        print(f"  Warning: {missing_pq} queries missing per-query features")

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written {len(rows)} rows to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    mc = Counter(r["best_method"] for r in rows)
    for method, count in mc.most_common():
        print(f"  {method}: {count} ({100*count/len(rows):.1f}%)")
    print(f"\n  Datasets: {dict(Counter(r['dataset'] for r in rows))}")
    print(f"  Scenarios: {dict(Counter(r['scenario'] for r in rows))}")
    print(f"  Features: {len(fieldnames) - 3 - 3} input + 3 target + 3 identifiers")


if __name__ == "__main__":
    main()
