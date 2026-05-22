#!/usr/bin/env python3
"""
Train and evaluate ML routing models for filtered ANN method selection.

Compares multiple ML models against the rule-based router:
  1. XGBoost
  2. Random Forest
  3. MLP (Multi-layer Perceptron)
  4. Logistic Regression (linear baseline)

Evaluation: Leave-One-Dataset-Out cross-validation.
  - Train on 5 datasets, test on the held-out dataset
  - This simulates deploying the router on a new, unseen dataset

Usage:
  python analysis/train_ml_router.py \
      --input analysis/ml_training_data.csv \
      --output-dir analysis/ml_routing_results
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed, skipping XGBoost model")


# ── Rule-based router ───────────────────────────────────────────────────────


def rule_router(row: pd.Series) -> str:
    """The hand-crafted rule router from Chapter 5.4."""
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


# ── Features ─────────────────────────────────────────────────────────────────


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

# Scenario is categorical, will be one-hot encoded
SCENARIO_COL = "scenario"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix with one-hot encoded scenario."""
    X = df[FEATURE_COLS].copy()
    # One-hot encode scenario
    scenario_dummies = pd.get_dummies(df[SCENARIO_COL], prefix="scenario")
    X = pd.concat([X, scenario_dummies], axis=1)
    # Fill NaN (e.g., normalized_ratio might be None)
    X = X.fillna(0)
    return X


# ── Models ───────────────────────────────────────────────────────────────────


def get_models():
    """Return dict of model name -> (model, needs_scaling)."""
    models = {}

    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            False,  # no scaling needed
        )

    models["RandomForest"] = (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        False,
    )

    models["MLP"] = (
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ),
        True,  # needs scaling
    )

    models["LogisticRegression"] = (
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class="multinomial",
            C=1.0,
        ),
        True,
    )

    return models


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_lodo(df: pd.DataFrame, output_dir: Path):
    """Leave-One-Dataset-Out cross-validation."""
    datasets = sorted(df["dataset"].unique())
    models = get_models()
    label_enc = LabelEncoder()
    label_enc.fit(df["best_method"])

    # Results storage
    all_results = {}  # model_name -> list of per-dataset results
    rule_results = []

    print(f"\n{'='*70}")
    print("Leave-One-Dataset-Out Cross-Validation")
    print(f"{'='*70}")
    print(f"Datasets: {datasets}")
    print(f"Models: {list(models.keys())} + RuleRouter")
    print(f"Classes: {list(label_enc.classes_)}")

    for model_name in models:
        all_results[model_name] = []

    for held_out in datasets:
        train_df = df[df["dataset"] != held_out]
        test_df = df[df["dataset"] == held_out]

        X_train = prepare_features(train_df)
        X_test = prepare_features(test_df)
        y_train = label_enc.transform(train_df["best_method"])
        y_test = label_enc.transform(test_df["best_method"])
        y_test_labels = test_df["best_method"].values

        # Align columns (in case one-hot creates different columns)
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[X_train.columns]

        print(f"\n--- Held out: {held_out} (train={len(train_df)}, test={len(test_df)}) ---")

        # Rule router
        rule_preds = test_df.apply(rule_router, axis=1).values
        rule_acc = accuracy_score(y_test_labels, rule_preds)
        # Recall-weighted accuracy: weight by best_recall to measure routing quality
        rule_recall_match = np.mean([
            test_df.iloc[i]["best_recall"] if rule_preds[i] == y_test_labels[i]
            else 0  # placeholder, will compute actual recall below
            for i in range(len(test_df))
        ])
        rule_results.append({
            "dataset": held_out,
            "accuracy": rule_acc,
            "predictions": rule_preds,
        })
        print(f"  RuleRouter:          acc={rule_acc:.4f}")

        for model_name, (model, needs_scaling) in models.items():
            X_tr = X_train.copy()
            X_te = X_test.copy()

            if needs_scaling:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    scaler.fit_transform(X_tr),
                    columns=X_tr.columns, index=X_tr.index
                )
                X_te = pd.DataFrame(
                    scaler.transform(X_te),
                    columns=X_te.columns, index=X_te.index
                )

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_pred_labels = label_enc.inverse_transform(y_pred)
            acc = accuracy_score(y_test, y_pred)

            all_results[model_name].append({
                "dataset": held_out,
                "accuracy": acc,
                "predictions": y_pred_labels,
            })
            print(f"  {model_name:20s} acc={acc:.4f}")

    # ── Aggregate results ────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (Leave-One-Dataset-Out)")
    print(f"{'='*70}")

    summary = {}

    # Rule router aggregate
    rule_accs = [r["accuracy"] for r in rule_results]
    print(f"\n  {'RuleRouter':20s}  mean_acc={np.mean(rule_accs):.4f}  "
          f"min={np.min(rule_accs):.4f}  max={np.max(rule_accs):.4f}")
    summary["RuleRouter"] = {
        "mean_accuracy": float(np.mean(rule_accs)),
        "per_dataset": {r["dataset"]: r["accuracy"] for r in rule_results},
    }

    for model_name in models:
        accs = [r["accuracy"] for r in all_results[model_name]]
        print(f"  {model_name:20s}  mean_acc={np.mean(accs):.4f}  "
              f"min={np.min(accs):.4f}  max={np.max(accs):.4f}")
        summary[model_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "per_dataset": {
                r["dataset"]: r["accuracy"]
                for r in all_results[model_name]
            },
        }

    # ── Per-dataset breakdown ────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("PER-DATASET ACCURACY")
    print(f"{'='*70}")

    header = f"  {'Dataset':12s} {'RuleRouter':>12s}"
    for mn in models:
        header += f" {mn:>15s}"
    print(header)
    print("  " + "-" * (12 + 13 + 16 * len(models)))

    for ds in datasets:
        line = f"  {ds:12s} {summary['RuleRouter']['per_dataset'][ds]:>12.4f}"
        for mn in models:
            line += f" {summary[mn]['per_dataset'][ds]:>15.4f}"
        print(line)

    # ── Feature importance (from best tree model) ────────────────────────

    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE (trained on full dataset)")
    print(f"{'='*70}")

    X_full = prepare_features(df)
    y_full = label_enc.transform(df["best_method"])

    # Train on full data for feature importance
    tree_model_name = "XGBoost" if HAS_XGBOOST else "RandomForest"
    tree_model, _ = models[tree_model_name]
    tree_model.fit(X_full, y_full)
    importances = tree_model.feature_importances_
    feat_imp = sorted(
        zip(X_full.columns, importances),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  {tree_model_name} feature importance (trained on all data):")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 100)
        print(f"    {feat:25s} {imp:.4f} {bar}")

    summary["feature_importance"] = {
        "model": tree_model_name,
        "features": {f: float(v) for f, v in feat_imp},
    }

    # ── Per-scenario analysis ────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("PER-SCENARIO ACCURACY (aggregated across LODO folds)")
    print(f"{'='*70}")

    # Collect all predictions
    all_preds = {}  # model_name -> full prediction array aligned with df
    # Rule router predictions for all data
    all_preds["RuleRouter"] = df.apply(rule_router, axis=1).values

    for model_name, (model, needs_scaling) in models.items():
        preds = np.empty(len(df), dtype=object)
        for res in all_results[model_name]:
            mask = df["dataset"] == res["dataset"]
            preds[mask] = res["predictions"]
        all_preds[model_name] = preds

    for scenario in ["and", "or", "equal"]:
        mask = df["scenario"] == scenario
        print(f"\n  Scenario: {scenario} (n={mask.sum()})")
        for name in ["RuleRouter"] + list(models.keys()):
            acc = accuracy_score(
                df.loc[mask, "best_method"].values,
                all_preds[name][mask]
            )
            print(f"    {name:20s}  acc={acc:.4f}")

    # ── Save results ─────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ml_routing_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'ml_routing_results.json'}")

    # Save full predictions for analysis
    pred_df = df[["query_id", "dataset", "scenario", "best_method", "best_recall"]].copy()
    pred_df["rule_pred"] = all_preds["RuleRouter"]
    for name in models:
        pred_df[f"{name}_pred"] = all_preds[name]
    pred_df.to_csv(output_dir / "ml_routing_predictions.csv", index=False)
    print(f"Predictions saved to {output_dir / 'ml_routing_predictions.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML routing models")
    parser.add_argument("--input", type=Path,
                        default=Path("analysis/ml_training_data.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/ml_router/results/v1"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets, "
          f"{df['scenario'].nunique()} scenarios")
    print(f"Target distribution:\n{df['best_method'].value_counts().to_string()}")

    evaluate_lodo(df, args.output_dir)


if __name__ == "__main__":
    main()
