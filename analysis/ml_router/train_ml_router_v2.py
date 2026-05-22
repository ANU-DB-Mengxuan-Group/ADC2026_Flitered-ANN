#!/usr/bin/env python3
"""
ML routing experiment v2: LODO + full-train/validation-test.

Two experiments in one script:
  1. LODO (existing 6 datasets, 3 methods only) — reproduces prior analysis
  2. Full-train + validation-test (train on 6, test on validation datasets)

Key features:
  - Computes actual routing RECALL (not just accuracy)
  - Feature set ablation: --feature-set {full, minimal, core, dataset-only}
  - Regression mode: --regression (predict per-method recall, pick max)

Usage:
  # LODO with all features (default)
  python analysis/train_ml_router_v2.py --mode lodo

  # Full train + validation with minimal features + regression
  python analysis/train_ml_router_v2.py --mode full \
      --val-recall analysis/ml_router/perquery_recall/perquery_recall_validation_3methods.csv \
      --feature-set minimal --regression

  # With centroid features
  python analysis/train_ml_router_v2.py --mode both \
      --val-recall analysis/ml_router/perquery_recall/perquery_recall_validation_3methods.csv \
      --centroid-train analysis/centroid_features_train.csv \
      --centroid-val analysis/centroid_features_val.csv \
      --regression
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed, skipping XGBoost model")


# ── Configuration ─────────────────────────────────────────────────────────

CANDIDATE_METHODS = ["UNG", "Post-filter", "SIEVE", "ACORN", "FilteredVamana"]
RECALL_THRESHOLD = 0.7

FEATURE_COLS = [
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
]

CENTROID_FEATURE_COLS = [
    "min_centroid_dist", "max_centroid_dist", "mean_centroid_dist",
]

# ── Feature sets for ablation ────────────────────────────────────────────

FEATURE_SETS = {
    "full": [
        "query_label_count", "selectivity",
        "min_label_freq", "max_label_freq", "mean_label_freq",
        "label_cooccurrence", "avg_labels_per_vec",
        "N", "dim",
        "LID_mean", "LID_median", "LID_std",
        "RC_median", "RC_trimmed_mean", "RC_p95",
        "label_cardinality", "label_entropy", "num_unique_combos",
        "DF_mean_swd",
        "correlation_ratio", "normalized_ratio",
    ],
    "minimal": [
        "selectivity", "LID_mean",
    ],
    # Intermediate sizes for finer-grained feature ablation
    # (selectivity + LID_mean always included; additional features added by
    # RandomForest importance order from V1 classification ablation)
    "size4": [
        "selectivity", "LID_mean", "label_cardinality",
    ],
    "size5": [
        "selectivity", "LID_mean", "label_cardinality",
        "label_entropy",
    ],
    "size8": [
        "selectivity", "LID_mean", "label_cardinality",
        "label_entropy", "avg_labels_per_vec", "N", "num_unique_combos",
    ],
    "size12": [
        "selectivity", "LID_mean", "label_cardinality",
        "label_entropy", "avg_labels_per_vec", "N", "num_unique_combos",
        "dim", "LID_median", "LID_std", "query_label_count",
    ],
    "size18": [
        "selectivity", "LID_mean", "label_cardinality",
        "label_entropy", "avg_labels_per_vec", "N", "num_unique_combos",
        "dim", "LID_median", "LID_std", "query_label_count",
        "label_cooccurrence", "DF_mean_swd", "RC_median",
        "RC_trimmed_mean", "correlation_ratio", "RC_p95",
    ],
    # Diagnostic sets to test the hypothesis that correlation_ratio is harmful
    "core_no_corr": [
        # core minus correlation_ratio (5 total = 4 numeric + scenario)
        "selectivity", "LID_mean", "label_cardinality", "avg_labels_per_vec",
    ],
    "minimal_plus_corr": [
        # minimal + correlation_ratio only (4 total = 3 numeric + scenario)
        "selectivity", "LID_mean", "correlation_ratio",
    ],
    "core": [
        "selectivity", "LID_mean", "label_cardinality",
        "correlation_ratio", "avg_labels_per_vec",
    ],
    "core+centroid": [
        "selectivity", "LID_mean", "label_cardinality",
        "correlation_ratio", "avg_labels_per_vec",
        "min_centroid_dist", "max_centroid_dist", "mean_centroid_dist",
    ],
    "dataset-only": [
        "N", "dim",
        "LID_mean", "LID_median", "LID_std",
        "RC_median", "RC_trimmed_mean", "RC_p95",
        "label_cardinality", "label_entropy", "num_unique_combos",
        "DF_mean_swd",
        "correlation_ratio", "normalized_ratio",
    ],
}


# ── Rule router ───────────────────────────────────────────────────────────

def rule_router(row: pd.Series) -> str:
    scenario = row["scenario"]
    lid = row["LID_mean"]
    card = row["label_cardinality"]

    if scenario == "equal":
        return "UNG"
    elif scenario == "and":
        if lid > 100:
            return "UNG"
        elif card < 100:
            return "UNG"
        else:
            return "SIEVE"
    elif scenario == "or":
        if lid > 100:
            return "UNG"
        else:
            return "Post-filter"
    return "UNG"


# ── Feature preparation ──────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, use_centroid: bool = False,
                     feature_set: str = "full") -> pd.DataFrame:
    cols = list(FEATURE_SETS.get(feature_set, FEATURE_SETS["full"]))
    if use_centroid:
        for c in CENTROID_FEATURE_COLS:
            if c in df.columns and c not in cols:
                cols.append(c)
    # Only keep columns that exist in df
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    scenario_dummies = pd.get_dummies(df["scenario"], prefix="scenario", dtype=float)
    X = pd.concat([X, scenario_dummies], axis=1)
    X = X.fillna(0)
    # Ensure all columns are numeric float (fixes xgboost/pandas compatibility)
    X = X.astype(np.float64)
    return X


# ── Build per-query recall lookup ─────────────────────────────────────────

def load_recall_lookup(recall_csv: Path, methods: list[str]) -> dict:
    """Load per-query recall, filtered to candidate methods.

    Returns:
        recall_lookup: {(query_id, dataset, scenario, method): recall}
    """
    lookup = {}
    with open(recall_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in methods:
                continue
            key = (int(row["query_id"]), row["dataset"], row["scenario"], method)
            lookup[key] = float(row["recall_at_10"])
    return lookup


def compute_best_method(recall_lookup: dict, methods: list[str],
                        threshold: float = 0.7) -> pd.DataFrame:
    """From recall lookup, determine best method per query (with threshold).

    Returns DataFrame with query_id, dataset, scenario, best_method, best_recall,
    and per-method recall columns.
    """
    # Group by query
    query_data = defaultdict(dict)
    for (qid, ds, sc, method), recall in recall_lookup.items():
        query_data[(qid, ds, sc)][method] = recall

    # Per-cell average recall for threshold
    cell_recalls = defaultdict(list)
    for (qid, ds, sc), mr in query_data.items():
        for method, recall in mr.items():
            cell_recalls[(ds, sc, method)].append(recall)
    cell_avg = {k: np.mean(v) for k, v in cell_recalls.items()}

    rows = []
    for (qid, ds, sc), method_recalls in sorted(query_data.items()):
        # Apply threshold
        filtered = {m: r for m, r in method_recalls.items()
                    if cell_avg.get((ds, sc, m), 0) >= threshold}
        if not filtered:
            filtered = method_recalls

        best = max(filtered, key=filtered.get)
        row = {
            "query_id": qid,
            "dataset": ds,
            "scenario": sc,
            "best_method": best,
            "best_recall": filtered[best],
        }
        # Add per-method recall for routing quality evaluation
        for m in methods:
            row[f"recall_{m}"] = method_recalls.get(m, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Compute routing recall ────────────────────────────────────────────────

def compute_routing_recall(predictions: np.ndarray, df: pd.DataFrame,
                           methods: list[str]) -> float:
    """Compute actual recall achieved by routing decisions.

    For each query, look up the recall of the predicted method.
    """
    recalls = []
    for i, pred in enumerate(predictions):
        col = f"recall_{pred}"
        if col in df.columns:
            recalls.append(df.iloc[i][col])
        else:
            recalls.append(0.0)
    return np.mean(recalls)


# ── Models ────────────────────────────────────────────────────────────────

def get_models():
    models = {}
    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="mlogloss", verbosity=0,
            ),
            False,
        )
    models["RandomForest"] = (
        RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
        False,
    )
    models["MLP"] = (
        MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.15,
        ),
        True,
    )
    models["LogisticRegression"] = (
        LogisticRegression(
            max_iter=1000, random_state=42, multi_class="multinomial", C=1.0,
        ),
        True,
    )
    return models


def get_regression_models():
    """Return regression models for predicting per-method recall."""
    models = {}
    if HAS_XGBOOST:
        models["XGB-Reg"] = (
            XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            ),
            False,
        )
    models["RF-Reg"] = (
        RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
        False,
    )
    models["MLP-Reg"] = (
        MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.15,
        ),
        True,
    )
    models["Ridge"] = (
        Ridge(alpha=1.0),
        True,
    )
    return models


def train_and_predict_regression(models: dict, X_train: pd.DataFrame,
                                 train_df: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 methods: list[str]) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict[str, np.ndarray]]]:
    """Train one regressor per method, predict recall, pick best.

    Returns (predictions, timings, per_method_preds) where:
      predictions: dict of model_name → predicted method labels (str array)
      timings: dict of model_name → inference time in µs/query
      per_method_preds: dict of model_name → dict of method → predicted recall array
        (let downstream apply recall threshold for QPS-aware routing)
    """
    predictions = {}
    timings = {}
    per_method_preds = {}

    for model_name, (model_template, needs_scaling) in models.items():
        # Predict recall for each method
        method_preds = {}
        fitted_models = {}
        for m in methods:
            from sklearn.base import clone
            model = clone(model_template)

            y_train = train_df[f"recall_{m}"].values
            X_tr, X_te = X_train.copy(), X_test.copy()
            if needs_scaling:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
                X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

            model.fit(X_tr, y_train)
            method_preds[m] = model.predict(X_te)
            fitted_models[m] = (model, X_te)

        # Timing: predict all methods + argmax
        t0 = time.perf_counter()
        for _ in range(100):
            for m in methods:
                mdl, xte = fitted_models[m]
                mdl.predict(xte)
        t1 = time.perf_counter()
        timings[model_name] = (t1 - t0) / 100 / len(X_test) * 1e6

        # Pick method with highest predicted recall
        pred_matrix = np.column_stack([method_preds[m] for m in methods])
        best_idx = np.argmax(pred_matrix, axis=1)
        pred_labels = np.array([methods[i] for i in best_idx])
        predictions[model_name] = pred_labels
        per_method_preds[model_name] = method_preds

    return predictions, timings, per_method_preds


# ── Experiment 1: LODO ───────────────────────────────────────────────────

def run_lodo(df: pd.DataFrame, output_dir: Path, use_centroid: bool = False,
             feature_set: str = "full", use_regression: bool = False):
    """Leave-One-Dataset-Out with 3 methods, computing both accuracy and recall."""
    datasets = sorted(df["dataset"].unique())
    cls_models = get_models()
    reg_models = get_regression_models() if use_regression else {}
    label_enc = LabelEncoder()
    label_enc.fit(df["best_method"])

    mode_str = "Classification + Regression" if use_regression else "Classification"
    print(f"\n{'='*70}")
    print(f"Experiment 1: LODO (3 methods, features={feature_set}, {mode_str})")
    print(f"{'='*70}")
    print(f"Datasets: {datasets}")
    print(f"Total queries: {len(df)}")
    print(f"Target distribution:\n{df['best_method'].value_counts().to_string()}")

    all_model_names = list(cls_models.keys()) + list(reg_models.keys())
    # Collect all predictions for aggregate analysis
    all_preds = {name: np.empty(len(df), dtype=object)
                 for name in all_model_names + ["RuleRouter"]}
    fold_results = []

    for held_out in datasets:
        train_fold = df[df["dataset"] != held_out].reset_index(drop=True)
        test_fold = df[df["dataset"] == held_out].reset_index(drop=True)
        test_mask = df["dataset"] == held_out

        X_train = prepare_features(train_fold, use_centroid=use_centroid, feature_set=feature_set)
        X_test = prepare_features(test_fold, use_centroid=use_centroid, feature_set=feature_set)
        y_train = label_enc.transform(train_fold["best_method"])
        y_test_labels = test_fold["best_method"].values

        # Align columns
        for c in set(X_train.columns) - set(X_test.columns):
            X_test[c] = 0
        X_test = X_test[X_train.columns]

        print(f"\n--- Held out: {held_out} (train={len(train_fold)}, test={len(test_fold)}) ---")

        # Rule router
        rule_preds = test_fold.apply(rule_router, axis=1).values
        rule_acc = accuracy_score(y_test_labels, rule_preds)
        rule_recall = compute_routing_recall(rule_preds, test_fold, CANDIDATE_METHODS)
        all_preds["RuleRouter"][test_mask] = rule_preds
        print(f"  {'RuleRouter':20s} acc={rule_acc:.4f}  recall={rule_recall:.4f}")

        fold_result = {"dataset": held_out, "n_test": len(test_fold)}
        fold_result["RuleRouter_acc"] = rule_acc
        fold_result["RuleRouter_recall"] = rule_recall

        # Classification models
        for model_name, (model, needs_scaling) in cls_models.items():
            X_tr, X_te = X_train.copy(), X_test.copy()
            if needs_scaling:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
                X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_pred_labels = label_enc.inverse_transform(y_pred)

            acc = accuracy_score(y_test_labels, y_pred_labels)
            recall = compute_routing_recall(y_pred_labels, test_fold, CANDIDATE_METHODS)
            all_preds[model_name][test_mask] = y_pred_labels

            fold_result[f"{model_name}_acc"] = acc
            fold_result[f"{model_name}_recall"] = recall
            print(f"  {model_name:20s} acc={acc:.4f}  recall={recall:.4f}")

        # Regression models
        if reg_models:
            reg_preds, _, _ = train_and_predict_regression(
                reg_models, X_train, train_fold, X_test, CANDIDATE_METHODS)
            for model_name, pred_labels in reg_preds.items():
                acc = accuracy_score(y_test_labels, pred_labels)
                recall = compute_routing_recall(pred_labels, test_fold, CANDIDATE_METHODS)
                all_preds[model_name][test_mask] = pred_labels
                fold_result[f"{model_name}_acc"] = acc
                fold_result[f"{model_name}_recall"] = recall
                print(f"  {model_name:20s} acc={acc:.4f}  recall={recall:.4f}")

        fold_results.append(fold_result)

    # ── Aggregate ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LODO AGGREGATE RESULTS")
    print(f"{'='*70}")

    summary = {"experiment": "LODO", "n_methods": len(CANDIDATE_METHODS),
               "methods": CANDIDATE_METHODS,
               "feature_set": feature_set, "regression": use_regression}

    all_names = ["RuleRouter"] + all_model_names
    print(f"\n  {'Model':20s} {'Mean Acc':>10s} {'Mean Recall':>12s} {'Oracle Gap':>12s}")
    print("  " + "-" * 56)

    # Oracle recall
    oracle_recall = df["best_recall"].mean()
    print(f"  {'Oracle':20s} {'—':>10s} {oracle_recall:>12.4f} {'—':>12s}")
    summary["oracle_recall"] = float(oracle_recall)

    for name in all_names:
        accs = [r[f"{name}_acc"] for r in fold_results]
        recalls = [r[f"{name}_recall"] for r in fold_results]
        mean_acc = np.mean(accs)
        # Weighted mean recall (by test set size)
        weights = [r["n_test"] for r in fold_results]
        mean_recall = np.average(recalls, weights=weights)
        gap = oracle_recall - mean_recall
        print(f"  {name:20s} {mean_acc:>10.4f} {mean_recall:>12.4f} {gap:>+12.4f}")
        summary[name] = {
            "mean_accuracy": float(mean_acc),
            "mean_recall": float(mean_recall),
            "oracle_gap": float(gap),
            "per_dataset": {r["dataset"]: {"acc": r[f"{name}_acc"], "recall": r[f"{name}_recall"]}
                            for r in fold_results},
        }

    # ── Per-scenario ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LODO PER-SCENARIO (weighted by test set size)")
    print(f"{'='*70}")

    for scenario in ["and", "or", "equal"]:
        mask = df["scenario"] == scenario
        n = mask.sum()
        print(f"\n  Scenario: {scenario} (n={n})")
        print(f"    {'Model':20s} {'Acc':>8s} {'Recall':>8s}")
        print(f"    {'-'*38}")

        sc_oracle = df.loc[mask, "best_recall"].mean()
        print(f"    {'Oracle':20s} {'—':>8s} {sc_oracle:>8.4f}")

        for name in all_names:
            preds = all_preds[name][mask]
            true = df.loc[mask, "best_method"].values
            acc = accuracy_score(true, preds)
            recall = compute_routing_recall(preds, df.loc[mask].reset_index(drop=True),
                                            CANDIDATE_METHODS)
            print(f"    {name:20s} {acc:>8.4f} {recall:>8.4f}")

    # ── Feature importance ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE (trained on full data)")
    print(f"{'='*70}")

    X_full = prepare_features(df, use_centroid=use_centroid, feature_set=feature_set)
    y_full = label_enc.transform(df["best_method"])
    tree_name = "XGBoost" if HAS_XGBOOST else "RandomForest"
    tree_model, _ = cls_models[tree_name]
    tree_model.fit(X_full, y_full)
    importances = tree_model.feature_importances_
    feat_imp = sorted(zip(X_full.columns, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:15]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:25s} {imp:.4f} {bar}")
    summary["feature_importance"] = {f: float(v) for f, v in feat_imp}

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{feature_set}" if feature_set != "full" else ""
    suffix += "_reg" if use_regression else ""
    fname = f"lodo_{len(CANDIDATE_METHODS)}methods_results{suffix}.json"
    with open(output_dir / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir / fname}")

    return summary


# ── Experiment 2: Full train + validation test ────────────────────────────

def run_full_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame, *,
                       output_dir: Path, use_centroid: bool = False,
                       feature_set: str = "full", use_regression: bool = False):
    """Train on all 6 original datasets, test on validation datasets."""
    cls_models = get_models()
    reg_models = get_regression_models() if use_regression else {}
    label_enc = LabelEncoder()
    all_methods = sorted(set(train_df["best_method"].unique()) |
                         set(val_df["best_method"].unique()))
    label_enc.fit(all_methods)

    mode_str = "Classification + Regression" if use_regression else "Classification"
    print(f"\n{'='*70}")
    print(f"Experiment 2: Full Train + Validation Test (features={feature_set}, {mode_str})")
    print(f"{'='*70}")
    print(f"Train: {len(train_df)} queries from {sorted(train_df['dataset'].unique())}")
    print(f"Val:   {len(val_df)} queries from {sorted(val_df['dataset'].unique())}")
    print(f"Train target:\n{train_df['best_method'].value_counts().to_string()}")
    print(f"Val target:\n{val_df['best_method'].value_counts().to_string()}")

    X_train = prepare_features(train_df, use_centroid=use_centroid, feature_set=feature_set)
    X_val = prepare_features(val_df, use_centroid=use_centroid, feature_set=feature_set)

    # Align columns
    for c in set(X_train.columns) - set(X_val.columns):
        X_val[c] = 0
    for c in set(X_val.columns) - set(X_train.columns):
        X_train[c] = 0
    X_val = X_val[X_train.columns]

    y_train = label_enc.transform(train_df["best_method"])
    y_val_labels = val_df["best_method"].values

    all_model_names = list(cls_models.keys()) + list(reg_models.keys())
    summary = {"experiment": "full_train_val",
               "n_methods": len(CANDIDATE_METHODS),
               "methods": CANDIDATE_METHODS, "feature_set": feature_set,
               "regression": use_regression,
               "n_train": len(train_df), "n_val": len(val_df)}

    # Oracle
    oracle_recall = val_df["best_recall"].mean()
    summary["oracle_recall"] = float(oracle_recall)

    all_names = ["RuleRouter"] + all_model_names
    all_preds = {}

    print(f"\n  {'Model':20s} {'Acc':>10s} {'Recall':>12s} {'Oracle Gap':>12s}")
    print("  " + "-" * 56)
    print(f"  {'Oracle':20s} {'—':>10s} {oracle_recall:>12.4f} {'—':>12s}")

    # Rule router (with timing)
    n_queries = len(val_df)
    t0 = time.perf_counter()
    for _ in range(100):  # 100 reps for stable timing
        rule_preds = val_df.apply(rule_router, axis=1).values
    t1 = time.perf_counter()
    rule_time_us = (t1 - t0) / 100 / n_queries * 1e6  # microseconds per query
    rule_acc = accuracy_score(y_val_labels, rule_preds)
    rule_recall = compute_routing_recall(rule_preds, val_df, CANDIDATE_METHODS)
    all_preds["RuleRouter"] = rule_preds
    print(f"  {'RuleRouter':20s} {rule_acc:>10.4f} {rule_recall:>12.4f} {oracle_recall - rule_recall:>+12.4f}  [{rule_time_us:.1f} µs/query]")
    summary["RuleRouter"] = {"accuracy": float(rule_acc), "recall": float(rule_recall), "time_us_per_query": float(rule_time_us)}

    # Classification models
    for model_name, (model, needs_scaling) in cls_models.items():
        X_tr, X_te = X_train.copy(), X_val.copy()
        if needs_scaling:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
            X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

        model.fit(X_tr, y_train)
        # Warm up + timing
        y_pred = model.predict(X_te)
        t0 = time.perf_counter()
        for _ in range(100):
            y_pred = model.predict(X_te)
        t1 = time.perf_counter()
        pred_time_us = (t1 - t0) / 100 / len(X_te) * 1e6
        y_pred_labels = label_enc.inverse_transform(y_pred)

        acc = accuracy_score(y_val_labels, y_pred_labels)
        recall = compute_routing_recall(y_pred_labels, val_df, CANDIDATE_METHODS)
        all_preds[model_name] = y_pred_labels
        gap = oracle_recall - recall
        print(f"  {model_name:20s} {acc:>10.4f} {recall:>12.4f} {gap:>+12.4f}  [{pred_time_us:.1f} µs/query]")
        summary[model_name] = {"accuracy": float(acc), "recall": float(recall), "time_us_per_query": float(pred_time_us)}

    # Regression models
    reg_per_method_preds = {}  # model_name → dict[method, predicted_recall_array]
    if reg_models:
        reg_preds, reg_timings, reg_per_method_preds = train_and_predict_regression(
            reg_models, X_train, train_df, X_val, CANDIDATE_METHODS)
        for model_name, pred_labels in reg_preds.items():
            acc = accuracy_score(y_val_labels, pred_labels)
            recall = compute_routing_recall(pred_labels, val_df, CANDIDATE_METHODS)
            all_preds[model_name] = pred_labels
            gap = oracle_recall - recall
            t_us = reg_timings.get(model_name, 0.0)
            print(f"  {model_name:20s} {acc:>10.4f} {recall:>12.4f} {gap:>+12.4f}  [{t_us:.1f} µs/query]")
            summary[model_name] = {"accuracy": float(acc), "recall": float(recall), "time_us_per_query": float(t_us)}

    # ── Per-dataset breakdown ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-DATASET RESULTS (Validation)")
    print(f"{'='*70}")

    for ds in sorted(val_df["dataset"].unique()):
        mask = val_df["dataset"] == ds
        n = mask.sum()
        print(f"\n  {ds} (n={n}):")
        ds_df = val_df.loc[mask].reset_index(drop=True)
        for name in all_names:
            preds = all_preds[name][mask]
            acc = accuracy_score(val_df.loc[mask, "best_method"].values, preds)
            recall = compute_routing_recall(preds, ds_df, CANDIDATE_METHODS)
            print(f"    {name:20s} acc={acc:.4f}  recall={recall:.4f}")

    # ── Per-scenario breakdown ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-SCENARIO RESULTS (Validation)")
    print(f"{'='*70}")

    for scenario in ["and", "or", "equal"]:
        mask = val_df["scenario"] == scenario
        if mask.sum() == 0:
            continue
        n = mask.sum()
        print(f"\n  Scenario: {scenario} (n={n})")
        sc_df = val_df.loc[mask].reset_index(drop=True)
        sc_oracle = sc_df["best_recall"].mean()
        print(f"    {'Oracle':20s} {'—':>8s} {sc_oracle:>8.4f}")
        for name in all_names:
            preds = all_preds[name][mask]
            acc = accuracy_score(val_df.loc[mask, "best_method"].values, preds)
            recall = compute_routing_recall(preds, sc_df, CANDIDATE_METHODS)
            print(f"    {name:20s} {acc:>8.4f} {recall:>8.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{feature_set}" if feature_set != "full" else ""
    suffix += "_reg" if use_regression else ""
    fname = f"full_train_val_results{suffix}.json"
    with open(output_dir / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir / fname}")

    # Save predictions
    pred_df = val_df[["query_id", "dataset", "scenario", "best_method", "best_recall"]].copy()
    for name in all_names:
        pred_df[f"{name}_pred"] = all_preds[name]
    # 加 actual per-method recall (从 val_df, 已是 ground truth)
    for m in CANDIDATE_METHODS:
        col = f"recall_{m}"
        if col in val_df.columns:
            pred_df[f"actual_{col}"] = val_df[col].values
    # 加 regression 模型的 predicted per-method recall (用于 QPS-aware threshold routing)
    for model_name, method_preds in reg_per_method_preds.items():
        for m, pred_arr in method_preds.items():
            pred_df[f"{model_name}_predrecall_{m}"] = pred_arr
    pred_df.to_csv(output_dir / f"full_train_val_predictions{suffix}.csv", index=False)

    return summary


# ── Build training data from recall CSV ───────────────────────────────────

def build_training_data(recall_csv: Path, features_csv: Path,
                        methods: list[str]) -> pd.DataFrame:
    """Build training data: recompute best_method using only candidate methods."""
    print(f"Loading recall data from {recall_csv} ...")
    recall_lookup = load_recall_lookup(recall_csv, methods)
    print(f"  {len(recall_lookup)} entries for methods {methods}")

    best_df = compute_best_method(recall_lookup, methods, RECALL_THRESHOLD)
    print(f"  {len(best_df)} queries with best method computed")
    print(f"  Distribution: {dict(Counter(best_df['best_method']))}")

    # Load features
    print(f"Loading features from {features_csv} ...")
    feat_df = pd.read_csv(features_csv)
    # Drop old best_method columns
    drop_cols = [c for c in feat_df.columns if c.startswith("best_") or c == "n_candidates"
                 or c.startswith("recall_")]
    feat_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])

    # Merge
    merged = best_df.merge(feat_df, on=["query_id", "dataset", "scenario"], how="inner")
    print(f"  Merged: {len(merged)} rows")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML routing v2: LODO + full-train/val")
    parser.add_argument("--mode", choices=["lodo", "full", "both"], default="lodo",
                        help="Which experiment(s) to run")
    parser.add_argument("--train-recall",  type=Path,
                        default=Path("analysis/ml_router/perquery_recall/perquery_recall_combined_no_prefilter.csv"))
    parser.add_argument("--train-features", type=Path,
                        default=Path("analysis/ml_training_data.csv"))
    parser.add_argument("--val-recall", type=Path,
                        default=Path("analysis/ml_router/perquery_recall/perquery_recall_validation_5methods.csv"))
    parser.add_argument("--val-features", type=Path,
                        default=Path("analysis/ml_validation_data.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/ml_router/results/v2_5methods"))
    parser.add_argument("--centroid-train", type=Path, default=None,
                        help="Centroid features CSV for training data")
    parser.add_argument("--centroid-val", type=Path, default=None,
                        help="Centroid features CSV for validation data")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS.keys()),
                        default="full",
                        help="Feature set to use (default: full)")
    parser.add_argument("--regression", action="store_true",
                        help="Also train regression models (predict recall per method)")
    args = parser.parse_args()

    use_centroid = args.centroid_train is not None
    print(f"Feature set: {args.feature_set} ({len(FEATURE_SETS[args.feature_set])} base features)")
    if args.regression:
        print("Regression mode: ON (predicting per-method recall)")
    if use_centroid:
        print("Centroid features: ON")

    # Build training data (recompute best_method for 3 methods)
    train_df = build_training_data(args.train_recall, args.train_features,
                                   CANDIDATE_METHODS)

    # Merge centroid features if provided
    if args.centroid_train and args.centroid_train.exists():
        centroid_df = pd.read_csv(args.centroid_train)
        train_df = train_df.merge(centroid_df, on=["query_id", "dataset", "scenario"],
                                  how="left")
        for c in CENTROID_FEATURE_COLS:
            train_df[c] = train_df[c].fillna(0)
        print(f"  Merged centroid features: {len(centroid_df)} rows from {args.centroid_train}")

    if args.mode in ("lodo", "both"):
        run_lodo(train_df, args.output_dir, use_centroid=use_centroid,
                 feature_set=args.feature_set, use_regression=args.regression)

    if args.mode in ("full", "both"):
        if not args.val_features.exists():
            print(f"\nERROR: Validation features not found: {args.val_features}")
            print("Run build_ml_training_data.py for validation datasets first.")
            print("See analysis/build_ml_validation_data.sh for instructions.")
            return
        if not args.val_recall.exists():
            print(f"\nERROR: Validation recall not found: {args.val_recall}")
            return

        val_df = build_training_data(args.val_recall, args.val_features,
                                     CANDIDATE_METHODS)

        # Merge centroid features for validation
        if args.centroid_val and args.centroid_val.exists():
            centroid_df = pd.read_csv(args.centroid_val)
            val_df = val_df.merge(centroid_df, on=["query_id", "dataset", "scenario"],
                                  how="left")
            for c in CENTROID_FEATURE_COLS:
                val_df[c] = val_df[c].fillna(0)
            print(f"  Merged centroid features: {len(centroid_df)} rows from {args.centroid_val}")

        run_full_train_val(train_df, val_df, output_dir=args.output_dir,
                           use_centroid=use_centroid, feature_set=args.feature_set,
                           use_regression=args.regression)


if __name__ == "__main__":
    main()
