#!/usr/bin/env python3
"""Multi-seed feature ablation for MLP-Reg.

Runs MLP-Reg training across all FEATURE_SETS variants and multiple seeds,
recording validation set recall for each (feature_set, seed) combination.

Output: analysis/ml_router/results/v2_5methods/feature_ablation_multiseed.csv
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from train_ml_router_v2 import (
    CANDIDATE_METHODS, FEATURE_SETS, build_training_data, prepare_features,
    compute_routing_recall,
)

OUT_CSV = REPO / "analysis/ml_router/results/v2_5methods/feature_ablation_multiseed.csv"

SIZES = [
    ("minimal", 3),
    ("size4",   4),
    ("size5",   5),
    ("core",    6),
    ("size8",   8),
    ("size12", 12),
    ("size18", 18),
    ("full",   22),
]
SEEDS = [42, 7, 13, 21, 99]


def train_one(X_train, X_val, train_df, val_df, seed):
    """Train per-method MLP-Reg with given seed, return validation recall."""
    method_preds = {}
    for m in CANDIDATE_METHODS:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_te = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=seed, early_stopping=True, validation_fraction=0.15,
        )
        model.fit(X_tr, train_df[f"recall_{m}"].values)
        method_preds[m] = model.predict(X_te)
    arr = np.column_stack([method_preds[m] for m in CANDIDATE_METHODS])
    best_idx = arr.argmax(axis=1)
    pred_labels = np.array([CANDIDATE_METHODS[i] for i in best_idx])
    return compute_routing_recall(pred_labels, val_df, CANDIDATE_METHODS)


def main():
    print("Loading training and validation data ...")
    train_df = build_training_data(
        REPO / "analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv",
        REPO / "analysis/ml_training_data.csv",
        CANDIDATE_METHODS,
    )
    val_df = build_training_data(
        REPO / "analysis/ml_router/perquery_recall/perquery_recall_validation_5methods.csv",
        REPO / "analysis/ml_v2_data.csv",
        CANDIDATE_METHODS,
    )
    print(f"train: {len(train_df)} rows, val: {len(val_df)} rows")

    rows = []
    for set_name, n_feat in SIZES:
        X_train = prepare_features(train_df, feature_set=set_name)
        X_val   = prepare_features(val_df,   feature_set=set_name)
        # Align columns (scenario one-hot may differ if some sets are missing)
        for c in set(X_train.columns) - set(X_val.columns):
            X_val[c] = 0
        for c in set(X_val.columns) - set(X_train.columns):
            X_train[c] = 0
        X_val = X_val[X_train.columns]

        print(f"\n=== Feature set {set_name} (n={n_feat}, input dim={X_train.shape[1]}) ===")
        for seed in SEEDS:
            r = train_one(X_train, X_val, train_df, val_df, seed)
            print(f"  seed={seed:>3d}  recall={r:.4f}")
            rows.append({
                "feature_set": set_name,
                "n_features": n_feat,
                "seed": seed,
                "recall": r,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # Summary
    print("\n=== Summary ===")
    df = pd.DataFrame(rows)
    summary = df.groupby(["feature_set", "n_features"]).agg(
        mean_recall=("recall", "mean"),
        std_recall=("recall", "std"),
        min_recall=("recall", "min"),
        max_recall=("recall", "max"),
    ).reset_index().sort_values("n_features")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
