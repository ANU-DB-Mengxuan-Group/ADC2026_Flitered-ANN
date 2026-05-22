#!/usr/bin/env python3
"""Layers ablation for MLP-Reg routing model.

Trains MLP-Reg with 4 different hidden-layer configurations using the minimal
feature set (selectivity + LID_mean + scenario one-hot). Each variant is
trained on V1 (6 datasets) and evaluated on V2 (5 validation datasets),
producing per-query predictions + summary recall and inference timing.

Usage:
    python analysis/layers_ablation.py

Outputs:
    analysis/ml_router/results/v2_5methods/layers_ablation_summary.json
    analysis/ml_router/results/v2_5methods/layers_ablation_predictions_<n>.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Reuse infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_router_v2 import (
    CANDIDATE_METHODS, build_training_data, prepare_features,
    compute_routing_recall,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "analysis/ml_router/results/v2_5methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER_CONFIGS = [
    ("1L", (32,)),
    ("2L", (64, 32)),
    ("3L", (128, 64, 32)),
    ("4L", (256, 128, 64, 32)),
]


def train_and_eval(hidden, X_train, X_val, train_df, val_df, methods):
    """Train per-method MLPRegressor, predict per-method recall, argmax → method.
    Returns (pred_labels, per_method_pred, mean_inference_us)."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_te = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)

    method_preds, fitted = {}, {}
    base = MLPRegressor(
        hidden_layer_sizes=hidden, max_iter=500,
        random_state=42, early_stopping=True, validation_fraction=0.15,
    )
    for m in methods:
        model = clone(base)
        y = train_df[f"recall_{m}"].values
        model.fit(X_tr, y)
        method_preds[m] = model.predict(X_te)
        fitted[m] = model

    # Inference timing: full per-query routing = predict all methods + argmax
    t0 = time.perf_counter()
    for _ in range(100):
        for m in methods:
            fitted[m].predict(X_te)
    t1 = time.perf_counter()
    us_per_query = (t1 - t0) / 100 / len(X_te) * 1e6

    pred_matrix = np.column_stack([method_preds[m] for m in methods])
    best_idx = np.argmax(pred_matrix, axis=1)
    pred_labels = np.array([methods[i] for i in best_idx])

    return pred_labels, method_preds, us_per_query


def main():
    train_recall = REPO / "analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv"
    train_feats  = REPO / "analysis/ml_training_data.csv"
    val_recall   = REPO / "analysis/ml_router/perquery_recall/perquery_recall_validation_5methods.csv"
    val_feats    = REPO / "analysis/ml_v2_data.csv"

    print(f"Loading training data ...")
    train_df = build_training_data(train_recall, train_feats, CANDIDATE_METHODS)
    print(f"Loading validation data ...")
    val_df   = build_training_data(val_recall,   val_feats,   CANDIDATE_METHODS)

    feature_set = "minimal"
    X_train = prepare_features(train_df, feature_set=feature_set)
    X_val   = prepare_features(val_df,   feature_set=feature_set)
    # Align columns
    for c in set(X_train.columns) - set(X_val.columns): X_val[c] = 0
    for c in set(X_val.columns)   - set(X_train.columns): X_train[c] = 0
    X_val = X_val[X_train.columns]
    print(f"Feature set: {feature_set}, n_input_dim = {X_train.shape[1]}")

    oracle_recall = float(val_df["best_recall"].mean())
    print(f"Oracle recall: {oracle_recall:.4f}\n")
    print(f"  {'Variant':10s} {'#layers':>7s} {'Recall':>10s} {'Gap':>10s} {'µs/query':>12s}")
    print("  " + "-" * 55)

    summary = {
        "experiment": "layers_ablation",
        "feature_set": feature_set,
        "n_input_dim": int(X_train.shape[1]),
        "n_methods": len(CANDIDATE_METHODS),
        "methods": CANDIDATE_METHODS,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "oracle_recall": oracle_recall,
        "variants": {},
    }

    for tag, hidden in LAYER_CONFIGS:
        pred_labels, method_preds, us = train_and_eval(
            hidden, X_train, X_val, train_df, val_df, CANDIDATE_METHODS)
        recall = compute_routing_recall(pred_labels, val_df, CANDIDATE_METHODS)
        gap = oracle_recall - recall
        nlayers = len(hidden)
        print(f"  {tag:10s} {nlayers:>7d} {recall:>10.4f} {gap:>+10.4f} {us:>12.2f}")

        summary["variants"][tag] = {
            "n_hidden_layers": nlayers,
            "hidden_layer_sizes": list(hidden),
            "recall": float(recall),
            "gap_to_oracle": float(gap),
            "time_us_per_query": float(us),
        }

        pred_df = val_df[["query_id", "dataset", "scenario", "best_method", "best_recall"]].copy()
        pred_df["MLP-Reg_pred"] = pred_labels
        for m, arr in method_preds.items():
            pred_df[f"MLP-Reg_predrecall_{m}"] = arr
        for m in CANDIDATE_METHODS:
            col = f"recall_{m}"
            if col in val_df.columns:
                pred_df[f"actual_{col}"] = val_df[col].values
        pred_df.to_csv(OUT_DIR / f"layers_ablation_predictions_{tag}.csv", index=False)

    out_json = OUT_DIR / "layers_ablation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()
