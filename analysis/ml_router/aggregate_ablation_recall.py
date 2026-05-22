#!/usr/bin/env python3
"""Aggregate plain argmax recall + inference latency for ablation variants.

For each ablation variant (feature: full/core/minimal, layers: 1L/2L/3L/4L),
reads predictions CSV, picks the method with the highest predicted recall per
query (or uses the MLP-Reg_pred column if present), then averages
actual_recall_<method> across queries.

Outputs analysis/ml_router/results/v2_5methods/ablation_metrics.csv with
columns: ablation, variant, n_features, n_layers, recall, gap_to_oracle,
inference_us, n_queries.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "analysis/ml_router/results/v2_5methods"

METHODS = ["UNG", "Post-filter", "SIEVE", "ACORN", "FilteredVamana"]


def derive_pred(pred_df: pd.DataFrame) -> pd.Series:
    """Pick a method per row: prefer MLP-Reg_pred col, else argmax over
    MLP-Reg_predrecall_<method>."""
    if "MLP-Reg_pred" in pred_df.columns:
        return pred_df["MLP-Reg_pred"]
    cols = [f"MLP-Reg_predrecall_{m}" for m in METHODS if f"MLP-Reg_predrecall_{m}" in pred_df.columns]
    methods_in_csv = [c.replace("MLP-Reg_predrecall_", "") for c in cols]
    arr = pred_df[cols].values
    best = arr.argmax(axis=1)
    return pd.Series([methods_in_csv[i] for i in best], index=pred_df.index)


def aggregate(pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv)
    df["pred"] = derive_pred(df)
    recalls = []
    for _, row in df.iterrows():
        pred = row["pred"]
        col = f"actual_recall_{pred}"
        if col not in df.columns:
            continue
        recalls.append(row[col])
    if not recalls:
        return {"n_queries": 0, "recall": None}
    return {"n_queries": len(recalls), "recall": float(np.mean(recalls))}


def main():
    rows = []

    FEATURE_VARIANTS = [
        ("full",    22, RESULTS / "full_train_val_predictions_reg.csv",
                        RESULTS / "full_train_val_results_reg.json"),
        ("core",     6, RESULTS / "full_train_val_predictions_core_reg.csv",
                        RESULTS / "full_train_val_results_core_reg.json"),
        ("minimal",  3, RESULTS / "full_train_val_predictions_minimal_reg.csv",
                        RESULTS / "full_train_val_results_minimal_reg.json"),
    ]
    oracle_recall = None
    print("=== Feature ablation ===")
    for tag, nf, pred_csv, sum_json in FEATURE_VARIANTS:
        with open(sum_json) as f:
            s = json.load(f)
        oracle_recall = s["oracle_recall"]
        inf = s["MLP-Reg"]["time_us_per_query"]
        agg = aggregate(pred_csv)
        gap = oracle_recall - agg["recall"]
        print(f"  {tag:8s} nf={nf:3d}  recall={agg['recall']:.4f}  gap={gap:+.4f}  inf_us={inf:.2f}  (n={agg['n_queries']})")
        rows.append(dict(ablation="features", variant=tag, n_features=nf, n_layers=2,
                         recall=agg["recall"], gap_to_oracle=gap, inference_us=inf,
                         n_queries=agg["n_queries"], oracle_recall=oracle_recall))

    print("\n=== Layers ablation ===")
    layers_sum_json = RESULTS / "layers_ablation_summary.json"
    if not layers_sum_json.exists():
        print(f"  WARN: layers summary missing")
    else:
        with open(layers_sum_json) as f:
            ls = json.load(f)
        oracle_recall = ls["oracle_recall"]
        for tag, info in ls["variants"].items():
            pred_csv = RESULTS / f"layers_ablation_predictions_{tag}.csv"
            inf = info["time_us_per_query"]
            nlayers = info["n_hidden_layers"]
            agg = aggregate(pred_csv)
            gap = oracle_recall - agg["recall"]
            print(f"  {tag:8s} nL={nlayers}  recall={agg['recall']:.4f}  gap={gap:+.4f}  inf_us={inf:.2f}  (n={agg['n_queries']})")
            rows.append(dict(ablation="layers", variant=tag, n_features=3, n_layers=nlayers,
                             recall=agg["recall"], gap_to_oracle=gap, inference_us=inf,
                             n_queries=agg["n_queries"], oracle_recall=oracle_recall))

    out = RESULTS / "ablation_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(f"Oracle recall: {oracle_recall:.4f}")


if __name__ == "__main__":
    main()
