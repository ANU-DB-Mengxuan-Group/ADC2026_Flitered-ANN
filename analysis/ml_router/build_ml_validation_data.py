#!/usr/bin/env python3
"""
Build ML validation dataset (same format as ml_training_data.csv).

Adapted from build_ml_training_data.py for validation datasets.
Produces one row per query with per-query features + dataset features + best method.

Usage:
  python analysis/build_ml_validation_data.py \
      --data-root ~/benchmarks/datasets/discrete \
      --analysis-dir analysis \
      --recall-csv analysis/ml_router/perquery_recall/perquery_recall_validation_3methods.csv \
      --output analysis/ml_validation_data.csv
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


VAL_DATASETS = [
    "synth200", "arxiv_fanns_real",
    "lid50", "lid80", "lid100", "lid120", "lid150",
    "synth5", "synth30", "synth100", "hm21",
]
SCENARIOS = ["and", "or", "equal"]
CANDIDATE_METHODS = ["UNG", "Post-filter", "SIEVE"]
RECALL_THRESHOLD = 0.7

# Calibration curve from LID sweep (random labels baseline)
CALIB_A = 0.1006
CALIB_B = -0.0589


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


def build_label_to_indices(labels: List[List[int]]) -> Dict[int, Set[int]]:
    idx = defaultdict(set)
    for i, lab_list in enumerate(labels):
        for l in lab_list:
            idx[l].add(i)
    return idx


def extract_perquery_features(data_root: Path, datasets: List[str]) -> Dict[tuple, dict]:
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

        label_to_idx = build_label_to_indices(base_labels)
        label_freq = {l: len(idxs) / n_base for l, idxs in label_to_idx.items()}
        base_label_sets = [set(ls) for ls in base_labels]
        avg_labels_per_vec = sum(len(ls) for ls in base_labels) / n_base

        for scenario in SCENARIOS:
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

                freqs = [label_freq.get(l, 0.0) for l in q_labels]

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


def load_dataset_features(analysis_dir: Path, datasets: List[str]) -> Dict[str, dict]:
    features = {}
    for ds in datasets:
        metrics_path = analysis_dir / f"{ds}_metrics.json"
        perlabel_path = analysis_dir / "per_label_lid" / f"{ds}_per_label_lid.json"

        if not metrics_path.is_file():
            print(f"  Warning: {metrics_path} not found, skipping {ds}")
            continue

        with open(metrics_path) as f:
            m = json.load(f)

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


def load_perquery_recall(recall_path: Path, threshold: float = 0.7) -> Dict:
    raw = {}
    with open(recall_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in CANDIDATE_METHODS:
                continue
            key = (int(row["query_id"]), row["dataset"], row["scenario"])
            if key not in raw:
                raw[key] = {}
            raw[key][method] = float(row["recall_at_10"])

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
        # Add per-method recall
        for m in CANDIDATE_METHODS:
            result[(qid, ds, sc)][f"recall_{m}"] = method_recalls.get(m, 0.0)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build ML validation data (same format as training data)")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--recall-csv", type=Path, required=True)
    parser.add_argument("--recall-threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/ml_validation_data.csv"))
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="覆盖 VAL_DATASETS, 用于 V2 数据集等. "
                             "例: --datasets synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else VAL_DATASETS

    # Step 1: Per-query features
    print("=" * 60)
    print(f"Step 1: Extracting per-query features ({len(datasets)} datasets)")
    print("=" * 60)
    pq_features = extract_perquery_features(args.data_root, datasets)
    print(f"\n  Total: {len(pq_features)} query features extracted")

    # Step 2: Dataset-level features
    print("\n" + "=" * 60)
    print("Step 2: Loading dataset-level features")
    print("=" * 60)
    ds_features = load_dataset_features(args.analysis_dir, datasets)
    print(f"  {len(ds_features)} datasets loaded")
    missing = [ds for ds in datasets if ds not in ds_features]
    if missing:
        print(f"  WARNING: Missing metrics for: {missing}")
        print(f"  Run 'python tools/dataset_metrics.py' for these datasets first!")

    # Step 3: Per-query recall
    print("\n" + "=" * 60)
    print("Step 3: Loading per-query recall")
    print("=" * 60)
    recall_data = load_perquery_recall(args.recall_csv, args.recall_threshold)
    print(f"  {len(recall_data)} query records loaded")

    # Step 4: Merge
    print("\n" + "=" * 60)
    print("Step 4: Merging into validation data")
    print("=" * 60)

    fieldnames = [
        "query_id", "dataset", "scenario",
        "query_label_count", "selectivity",
        "min_label_freq", "max_label_freq", "mean_label_freq",
        "label_cooccurrence", "avg_labels_per_vec",
        "N", "dim",
        "LID_mean", "LID_median", "LID_std",
        "RC_median", "RC_trimmed_mean", "RC_p95",
        "label_cardinality", "label_entropy", "num_unique_combos",
        "DF_mean_swd",
        "correlation_ratio", "normalized_ratio",
        "best_method", "best_recall", "n_candidates",
        "recall_UNG", "recall_Post-filter", "recall_SIEVE",
    ]

    rows = []
    missing_pq = 0
    missing_ds = 0
    for (qid, ds, sc), recall_info in sorted(recall_data.items()):
        if ds not in ds_features:
            missing_ds += 1
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
    if missing_ds > 0:
        print(f"  Warning: {missing_ds} queries missing dataset features (no metrics JSON)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written {len(rows)} rows to {args.output}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    mc = Counter(r["best_method"] for r in rows)
    for method, count in mc.most_common():
        print(f"  {method}: {count} ({100*count/len(rows):.1f}%)")
    print(f"\n  Datasets: {dict(Counter(r['dataset'] for r in rows))}")
    print(f"  Scenarios: {dict(Counter(r['scenario'] for r in rows))}")


if __name__ == "__main__":
    main()
