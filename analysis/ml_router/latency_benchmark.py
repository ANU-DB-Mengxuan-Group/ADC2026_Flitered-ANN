#!/usr/bin/env python3
"""Latency benchmark for online filtered-ANN routing.

Measures per-query latency of the complete routing pipeline (including
on-the-fly selectivity computation) and compares to underlying ANN search
latency. Replaces the partial 8.8 us number cited in earlier drafts.

Steps timed (per query, n_reps each):
  1. selectivity_bitmap         -- compute_selectivity_bitmap()
  2. feature_vector_build       -- np.array construction
  3. mlp_forward                -- 3 per-method MLP-Reg predictions
  4. argmax                     -- pick best method
  5. config_lookup              -- offline (method, config) table query

Total routing latency = sum of step medians.
ANN search latency  = 1 / median QPS at recall >= 0.9 (per dataset, scenario).

Output:
  --out-csv  : long-format latency breakdown CSV (one row per
               dataset x scenario x step x stat).
  --out-summary : per-(dataset, scenario) total routing latency vs ANN search.

Usage on cluster:
  cd ~/benchmarks/discrete
  python analysis/online_routing/latency_benchmark.py \
      --bitmap-dir analysis/ml_router/bitmaps \
      --train-csv  analysis/ml_training_data.csv \
      --metrics-dir analysis \
      --n-sample-queries 100 --n-reps 100
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from route_online import (
    compute_selectivity_packed,
    compute_selectivity_roaring,
    convert_packed_to_roaring,
    build_feature_vector_minimal,
    scale_manual,
    extract_mlp_weights,
    predict_recalls_manual,
    pick_method,
    lookup_config,
)


# ============================================================
# Constants
# ============================================================

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc",
               "yahoo800k", "dbpedia560k"]
SCENARIOS = ["and", "or", "equal"]
CANDIDATE_METHODS = ["UNG", "Post-filter", "SIEVE", "ACORN", "FilteredVamana"]
RECALL_THRESHOLD = 0.7


# ============================================================
# Loading
# ============================================================

def load_bitmap(bitmap_dir: Path, dataset: str) -> dict:
    with open(bitmap_dir / f"{dataset}.pkl", "rb") as f:
        return pickle.load(f)


def load_v2_queries(data_root: Path, dataset: str, scenario: str) -> list:
    """Read query label file. Returns list of list[int]."""
    fname = f"{dataset}_query_{scenario}.txt"
    qfile = data_root / dataset / fname
    queries = []
    with open(qfile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append([int(x) for x in line.split(",") if x])
    return queries


def load_lid_mean(metrics_dir: Path, dataset: str) -> float:
    with open(metrics_dir / f"{dataset}_metrics.json") as f:
        d = json.load(f)
    # tolerate either nested or flat
    return float(d.get("LID_mean") or d["lid"]["mean"])


# ============================================================
# Model training (V1 -> MLP-Reg minimal x3)
# ============================================================

def merge_recall_into_features(train_csv: Path, recall_csv: Path) -> pd.DataFrame:
    """Load V1 training features + merge per-method recall columns.

    train_csv has features but only best_method/best_recall.
    recall_csv has per-query per-method recall (long format).
    We pivot recall_csv and merge to produce wide-format recall_{method} columns.
    """
    feat_df = pd.read_csv(train_csv)
    rec_df = pd.read_csv(recall_csv)
    rec_df = rec_df[rec_df["method"].isin(CANDIDATE_METHODS)]
    # pivot to wide: one column per method
    pivot = rec_df.pivot_table(
        index=["query_id", "dataset", "scenario"],
        columns="method", values="recall_at_10",
    ).reset_index()
    pivot.columns = [
        f"recall_{c}" if c in CANDIDATE_METHODS else c
        for c in pivot.columns
    ]
    merged = feat_df.merge(pivot, on=["query_id", "dataset", "scenario"], how="inner")
    return merged


def train_mlp_reg_minimal(train_csv: Path, recall_csv: Path):
    """Train 3 per-method MLP-Reg regressors on V1 with minimal features.

    Returns: (regressors_dict, scaler).
    """
    df = merge_recall_into_features(train_csv, recall_csv)
    feat_cols = ["selectivity", "LID_mean"]
    X = df[feat_cols].values.astype(np.float32)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    regressors = {}
    for m in CANDIDATE_METHODS:
        col = f"recall_{m}"
        if col not in df.columns:
            print(f"[warn] {col} missing after merge; skipping {m}")
            continue
        y = df[col].values.astype(np.float32)
        mask = ~np.isnan(y)
        if mask.sum() < len(y):
            print(f"  [note] dropping {len(y) - mask.sum()} NaN rows for {m}")
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
        reg.fit(Xs[mask], y[mask])
        regressors[m] = reg
        print(f"  trained MLP-Reg[{m}] on {mask.sum()} samples")
    return regressors, scaler


# ============================================================
# Config table (offline) for QPS lookup
# ============================================================

def build_config_table(summary_paths: dict) -> dict:
    """Build offline config table.

    Args:
        summary_paths: {method -> summary.csv path glob list}

    Returns:
        {(dataset, scenario, method) -> [(config_str, recall, qps), ...]}
    """
    table = defaultdict(list)
    for method, paths in summary_paths.items():
        for path in paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            # find dataset name
            for _, row in df.iterrows():
                ds = row.get("dataset", path.parent.name)
                sc = row["scenario"]
                # Tolerate column name variants
                recall = row.get("recall@10") or row.get("recall_at_10") or row.get("recall")
                qps = row.get("qps") or row.get("QPS")
                if pd.isna(recall) or pd.isna(qps) or float(qps) <= 0:
                    continue
                # build config string from non-meta columns
                cfg = "_".join(
                    f"{k}={row[k]}" for k in df.columns
                    if k not in {"dataset", "scenario", "recall@10",
                                 "recall_at_10", "recall", "qps", "QPS",
                                 "build_time_s", "index_size_mb", "status",
                                 "timestamp"}
                    and not pd.isna(row[k])
                )
                table[(ds, sc, method)].append((cfg, float(recall), float(qps)))
    return dict(table)


def find_ann_search_latency(config_table: dict, dataset: str, scenario: str,
                            target_recall: float = 0.9) -> float | None:
    """Median ANN search latency (ms/query) among (method, config) with recall >= target."""
    qpss = []
    for m in CANDIDATE_METHODS:
        for (cfg, r, q) in config_table.get((dataset, scenario, m), []):
            if r >= target_recall:
                qpss.append(q)
    if not qpss:
        return None
    median_qps = float(np.median(qpss))
    return 1000.0 / median_qps  # ms/query


# ============================================================
# Latency timing
# ============================================================

def time_step(fn, n_reps=100):
    """Run fn() n_reps times, return list of per-call latencies (us)."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return times


def benchmark_query(q_labels, scenario, dataset, packed_data, lid_mean,
                    manual_models, scaler_mean, scaler_scale,
                    config_table, n_reps=100, T=0.9,
                    selectivity_impl="packed", roaring_data=None) -> dict:
    """Time each routing step for one query. Returns dict step -> [us].

    Times BOTH Plain (argmax + 1 lookup) and Strict (gate + N lookups + global max)
    paths so the per-step breakdown can be compared.

    selectivity_impl: "packed" (default) or "roaring". For "roaring", pass the
    converted roaring_data; the step is recorded as `selectivity_roaring`.
    """
    from route_online import (
        pick_methods_strict,
        lookup_config_strict,
    )
    if selectivity_impl == "roaring":
        assert roaring_data is not None, "roaring_data must be provided when selectivity_impl='roaring'"
        sel_fn = lambda: compute_selectivity_roaring(q_labels, scenario, roaring_data)
        sel_step_name = "selectivity_roaring"
    else:
        sel_fn = lambda: compute_selectivity_packed(q_labels, scenario, packed_data)
        sel_step_name = "selectivity_packed"

    sel = sel_fn()
    x = build_feature_vector_minimal(sel, lid_mean)
    x_scaled = scale_manual(x, scaler_mean, scaler_scale)
    preds = predict_recalls_manual(x_scaled, manual_models)
    m = pick_method(preds)
    _ = lookup_config(m, dataset, scenario, config_table)
    passing = pick_methods_strict(preds, T)
    _ = lookup_config_strict(passing, dataset, scenario, config_table, T)

    times = {}
    # Shared steps
    times[sel_step_name] = time_step(sel_fn, n_reps)
    times["feature_build_and_scale"] = time_step(
        lambda: scale_manual(build_feature_vector_minimal(sel, lid_mean), scaler_mean, scaler_scale),
        n_reps)
    times["mlp_forward_manual"] = time_step(
        lambda: predict_recalls_manual(x_scaled, manual_models), n_reps)
    # Plain-specific steps
    times["plain_argmax"] = time_step(
        lambda: pick_method(preds), n_reps)
    times["plain_config_lookup"] = time_step(
        lambda: lookup_config(m, dataset, scenario, config_table), n_reps)
    # Strict-specific steps
    times["strict_gate"] = time_step(
        lambda: pick_methods_strict(preds, T), n_reps)
    times["strict_lookup_global_qps"] = time_step(
        lambda: lookup_config_strict(passing, dataset, scenario, config_table, T), n_reps)
    return times


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path.home() / "benchmarks/datasets/discrete")
    p.add_argument("--bitmap-dir", type=Path,
                   default=Path("analysis/ml_router/bitmaps"))
    p.add_argument("--train-csv", type=Path,
                   default=Path("analysis/ml_training_data.csv"))
    p.add_argument("--recall-csv", type=Path,
                   default=Path("analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv"))
    p.add_argument("--metrics-dir", type=Path, default=Path("analysis"))
    p.add_argument("--ung-summary-glob", type=str,
                   default="UNG-dev/results*/{dataset}/summary.csv")
    p.add_argument("--pf-summary-glob", type=str,
                   default="faiss/results_postfilter/{dataset}/summary.csv")
    p.add_argument("--sieve-summary-glob", type=str,
                   default="SIEVE/results/{dataset}/summary.csv")
    p.add_argument("--acorn-summary-glob", type=str,
                   default="ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv")
    p.add_argument("--fv-summary-glob", type=str,
                   default="DiskANN/data_original/results/{dataset}/summary.csv")
    p.add_argument("--n-sample-queries", type=int, default=100)
    p.add_argument("--n-reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path,
                   default=Path("analysis/ml_router/latency_breakdown.csv"))
    p.add_argument("--out-summary", type=Path,
                   default=Path("analysis/online_routing/latency_summary.csv"))
    p.add_argument("--out-per-query", type=Path,
                   default=Path("analysis/online_routing/latency_per_query.csv"),
                   help="per-query plain/strict routing latency (one row per V2 query)")
    p.add_argument("--selectivity-impl", choices=["packed", "roaring"],
                   default="packed",
                   help="bitmap implementation for selectivity computation")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print(f"STEP 1: Train MLP-Reg minimal x{len(CANDIDATE_METHODS)} on V1")
    print("=" * 60)
    regressors, scaler = train_mlp_reg_minimal(args.train_csv, args.recall_csv)

    # Extract manual MLP weights (bypasses sklearn predict overhead)
    manual_models = {m: extract_mlp_weights(reg) for m, reg in regressors.items()}
    # Cache scaler params for manual scaling (bypass sklearn .transform overhead)
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    print(f"  extracted manual weights for {len(manual_models)} models")

    print()
    print("=" * 60)
    print("STEP 2: Build config table from method summary.csv files")
    print("=" * 60)
    repo = Path.cwd()  # assume run from ~/benchmarks/discrete
    summary_paths = {
        "UNG": [repo / args.ung_summary_glob.format(dataset=ds) for ds in V2_DATASETS],
        "Post-filter": [repo / args.pf_summary_glob.format(dataset=ds) for ds in V2_DATASETS],
        "SIEVE": [repo / args.sieve_summary_glob.format(dataset=ds) for ds in V2_DATASETS],
        "ACORN": [repo / args.acorn_summary_glob.format(dataset=ds) for ds in V2_DATASETS],
        "FilteredVamana": [repo / args.fv_summary_glob.format(dataset=ds) for ds in V2_DATASETS],
    }
    config_table = build_config_table(summary_paths)
    print(f"  loaded {sum(len(v) for v in config_table.values())} (method, cfg) entries "
          f"across {len(config_table)} cells")

    print()
    print("=" * 60)
    print("STEP 3: Latency benchmark")
    print("=" * 60)

    rows = []
    summary_rows = []
    per_query_rows = []
    for ds in V2_DATASETS:
        bitmap_path = args.bitmap_dir / f"{ds}.pkl"
        if not bitmap_path.exists():
            print(f"[skip] {ds}: bitmap missing")
            continue
        print(f"\nDataset: {ds}")
        bitmap_data = load_bitmap(args.bitmap_dir, ds)
        roaring_data = (convert_packed_to_roaring(bitmap_data)
                        if args.selectivity_impl == "roaring" else None)
        lid_mean = load_lid_mean(args.metrics_dir, ds)

        for sc in SCENARIOS:
            try:
                queries = load_v2_queries(args.data_root, ds, sc)
            except FileNotFoundError:
                print(f"  [skip] {ds}/{sc}: query file missing")
                continue
            if not queries:
                continue

            # sample N queries
            n_sample = min(args.n_sample_queries, len(queries))
            idxs = rng.choice(len(queries), size=n_sample, replace=False)

            sel_step_name = ("selectivity_roaring"
                             if args.selectivity_impl == "roaring"
                             else "selectivity_packed")
            SHARED_STEPS = [sel_step_name, "feature_build_and_scale",
                            "mlp_forward_manual"]
            PLAIN_STEPS = ["plain_argmax", "plain_config_lookup"]
            STRICT_STEPS = ["strict_gate", "strict_lookup_global_qps"]

            per_step_all = defaultdict(list)
            for qi in idxs:
                q = queries[int(qi)]
                step_times = benchmark_query(
                    q, sc, ds, bitmap_data, lid_mean,
                    manual_models, scaler_mean, scaler_scale,
                    config_table, n_reps=args.n_reps,
                    selectivity_impl=args.selectivity_impl,
                    roaring_data=roaring_data,
                )
                # Per-query medians (over n_reps) -> plain/strict routing total
                per_query_step_med = {step: float(np.median(ts))
                                       for step, ts in step_times.items()}
                pq_plain = sum(per_query_step_med.get(s, 0.0)
                               for s in SHARED_STEPS + PLAIN_STEPS)
                pq_strict = sum(per_query_step_med.get(s, 0.0)
                                for s in SHARED_STEPS + STRICT_STEPS)
                per_query_rows.append({
                    "dataset": ds, "scenario": sc, "query_id": int(qi),
                    "plain_routing_us": pq_plain,
                    "strict_routing_us": pq_strict,
                })
                for step, ts in step_times.items():
                    per_step_all[step].extend(ts)

            # Aggregate per step (using SHARED/PLAIN/STRICT_STEPS defined above)
            step_medians = {}
            for step, ts in per_step_all.items():
                arr = np.asarray(ts)
                med = float(np.median(arr))
                p95 = float(np.percentile(arr, 95))
                p99 = float(np.percentile(arr, 99))
                step_medians[step] = med
                rows.append({
                    "dataset": ds, "scenario": sc, "step": step,
                    "median_us": med, "p95_us": p95, "p99_us": p99,
                    "n_samples": len(arr),
                })
                print(f"  {sc:8s} {step:28s} median={med:7.2f}us  p95={p95:7.2f}us  p99={p99:7.2f}us")

            plain_total = sum(step_medians.get(s, 0.0) for s in SHARED_STEPS + PLAIN_STEPS)
            strict_total = sum(step_medians.get(s, 0.0) for s in SHARED_STEPS + STRICT_STEPS)

            ann_latency_ms = find_ann_search_latency(config_table, ds, sc)
            summary_rows.append({
                "dataset": ds,
                "scenario": sc,
                "plain_total_us": plain_total,
                "strict_total_us": strict_total,
                "strict_overhead_us": strict_total - plain_total,
                "ann_search_ms": ann_latency_ms,
                "plain_pct_of_total": (plain_total / (ann_latency_ms * 1000) * 100)
                                       if ann_latency_ms else None,
                "strict_pct_of_total": (strict_total / (ann_latency_ms * 1000) * 100)
                                        if ann_latency_ms else None,
            })
            ann_str = f"{ann_latency_ms:.2f} ms" if ann_latency_ms else "N/A"
            print(f"  {sc:8s} PLAIN total  = {plain_total:.2f} us  |  STRICT total = {strict_total:.2f} us  "
                  f"|  +{strict_total - plain_total:.2f} us overhead  |  ANN search = {ann_str}")

    # Write outputs
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote breakdown to {args.out_csv}")

    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote summary to {args.out_summary}")

    if per_query_rows:
        args.out_per_query.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_per_query, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
            w.writeheader()
            w.writerows(per_query_rows)
        print(f"Wrote per-query latency to {args.out_per_query}")


if __name__ == "__main__":
    main()
