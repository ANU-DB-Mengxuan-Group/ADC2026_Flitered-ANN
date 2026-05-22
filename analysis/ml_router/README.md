# Adaptive Query Routing (ML Router)

This directory contains the full reproducibility pipeline for the per-query ML router developed in the paper. The router predicts each candidate method's recall@10 on the incoming query using a small set of features, then consults an offline (method, configuration) → (recall, QPS) lookup table to pick the highest-QPS pair whose predicted recall meets a deployment threshold.

The pipeline has five stages. Each stage's scripts live directly in this directory; intermediate / output data live in the subdirectories described at the bottom.

```
[Stage 1] Per-query recall data collection
[Stage 2] Feature engineering and training-data assembly
[Stage 3] Train per-method recall regressors  (+ ablation studies)
[Stage 4] Evaluate on V1 / V2 validation sets
[Stage 5] Online inference and end-to-end latency benchmark
```

## Stage 1 — Per-query recall data collection

Each candidate method has an orchestrator that runs the method's search binary across all (dataset, scenario) cells and writes per-query top-$k$ result IDs plus recall@10 to CSV files under `per_query_results/`.

| Script                  | Method                |
|-------------------------|-----------------------|
| `run_perquery_ung.py`   | UNG                   |
| `run_perquery_pf.py`    | Post-filter HNSW      |
| `run_perquery_sieve.py` | SIEVE                 |
| `run_perquery_acorn.py` | ACORN-1 and ACORN-$\gamma$ |
| `run_perquery_fv.py`    | FilteredVamana        |
| `run_sieve_perquery.py` | SIEVE V2 batch runner |

Once the per-method orchestrators finish, two helpers consolidate their output:

| Script                            | Output                                                  |
|-----------------------------------|---------------------------------------------------------|
| `merge_per_query.py`              | `master_per_query.pkl` (one row per (query, method, cfg))   |
| `extract_perquery_recall.py`      | `perquery_recall/perquery_recall.csv` (training set)    |
| `extract_v2_perquery_recall.py`   | `perquery_recall/perquery_recall_v2.csv` (V2 set)       |
| `extract_v2_perquery_acorn.py`    | `perquery_recall/perquery_recall_v2_acorn.csv`          |
| `extract_v2_perquery_diskann.py`  | `perquery_recall/perquery_recall_v2_diskann.csv`        |

## Stage 2 — Feature engineering and training-data assembly

| Script                        | Purpose                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `compute_centroid_features.py`| Computes the centroid-based dataset-difficulty features used as routing inputs.                  |
| `build_ml_training_data.py`   | Joins per-query recall, per-query features, and dataset metadata into the V1 training table.     |
| `build_ml_validation_data.py` | Same for the V1 (11-dataset) and V2 (5-dataset) validation tables.                               |
| `audit_ml_data_completeness.py`| Verifies that every (dataset, scenario, method, config) cell has the expected number of rows.   |

Outputs: `ml_training_data.csv`, `ml_validation_data.csv`, `ml_v2_data.csv`, `centroid_features_{train,val}.csv` — these large CSVs are gitignored; regenerate them once Stage 1 has produced `master_per_query.pkl`.

## Stage 3 — Train the regressors

| Script                              | Purpose                                                                                     |
|-------------------------------------|---------------------------------------------------------------------------------------------|
| `train_ml_router.py`                | V1 baseline: per-method RandomForest regressor over 5 core features.                        |
| `train_ml_router_v2.py`             | V2 headline model: per-method MLP regressor over the 3-feature minimal set; also LODO mode. |
| `multi_seed_feature_ablation.py`    | 22 → 3 feature ablation under 5 random seeds for variance estimation.                       |
| `layers_ablation.py`                | MLP-depth ablation (1L / 2L / 3L / 4L hidden layers).                                       |

All four training scripts use the per-query recall + feature CSVs produced in Stage 2.

## Stage 4 — Evaluate

| Script                          | Purpose                                                                       |
|---------------------------------|-------------------------------------------------------------------------------|
| `evaluate_v2_routing.py`        | Runs the trained V2 router against the V2 validation set and computes the recall / QPS Pareto operating points used in the paper. |
| `aggregate_ablation_recall.py`  | Aggregates the multi-seed feature-ablation results into summary tables.       |
| `perquery_oracle_analysis.py`   | Computes the per-query Oracle upper bound under different candidate sets.     |

Results land in `results/v2_5methods/`, `results/v2_on_v2/`, `results/v2_3methods/`, etc. The large `full_train_val_predictions_*.csv` and `layers_ablation_predictions_*.csv` files in those subdirs are gitignored; the small `*.json` and `ablation_metrics.csv` / `feature_ablation_multiseed.csv` summaries are kept and are what the paper's plots read.

## Stage 5 — Online inference and latency

| Script                       | Purpose                                                                       |
|------------------------------|-------------------------------------------------------------------------------|
| `route_online.py`            | Serves a trained router: takes a query, computes features (selectivity via Roaring bitmap, LID lookup), runs the per-method regressors, applies the recall-threshold filter, and returns the (method, configuration) pick. |
| `latency_benchmark.py`       | End-to-end streaming-pipeline latency benchmark used to report the headline 54 μs / query number in the paper. |
| `audit_thread_scaling.py`    | Empirically measures per-method 16-thread scaling factors (used to convert single-thread per-query measurements into the multi-thread numbers reported elsewhere). |
| `merge_scaling_factors.py`   | Combines the per-method scaling-factor CSVs into one master table.            |
| `load_results.py`            | Small utility for the figure scripts to load the V1 / V2 result JSONs.        |

## Directory layout summary

```
analysis/ml_router/
├── README.md                              # this file
├── *.py                                   # 27 pipeline scripts
├── perquery_recall/                       # extracted per-query recall CSVs (gitignored; regenerable)
├── per_query_results/                     # raw per-query CSV per (dataset, scenario, method, cfg) (gitignored)
├── results/
│   ├── v1/                                # V1 train_ml_router.py outputs
│   ├── v1_backtest/                       # V1 backtest tables
│   ├── v2_3methods/                       # V2 with UNG/Post-filter/SIEVE candidates only
│   ├── v2_5methods/                       # V2 with UNG/PF/SIEVE/ACORN-γ/FV candidates (paper's main config)
│   └── v2_on_v2/                          # V2 router trained on V1+V2, evaluated on V2 (sensitivity)
├── master_per_query.pkl                   # merged per-query table (gitignored; 35 MB)
├── latency_*.csv                          # small latency-benchmark summary outputs (kept)
├── scaling_factors_audited*.csv           # small per-method scaling-factor outputs (kept)
└── router_used_configs.csv                # which (method, cfg) the router actually picks per cell
```

## Reproduction quick-start

Assuming the raw vector / label datasets are already in `~/benchmarks/datasets/discrete/<dataset>/` and each method's index is already built:

```bash
# 1. collect per-query recall (one orchestrator per method; can run in parallel)
for m in ung pf sieve acorn fv; do
    python analysis/ml_router/run_perquery_$m.py &
done
wait

# 2. merge and extract
python analysis/ml_router/merge_per_query.py
python analysis/ml_router/extract_perquery_recall.py
python analysis/ml_router/extract_v2_perquery_recall.py

# 3. feature build
python analysis/ml_router/compute_centroid_features.py
python analysis/ml_router/build_ml_training_data.py
python analysis/ml_router/build_ml_validation_data.py

# 4. train + evaluate (V2 headline)
python analysis/ml_router/train_ml_router_v2.py --feature-set minimal --regression
python analysis/ml_router/evaluate_v2_routing.py

# 5. latency benchmark
python analysis/ml_router/latency_benchmark.py --selectivity-impl roaring --n-sample-queries 1000
```

Each script has a `--help` flag listing its options. The cluster scripts that drove the actual experiments (`run_v2_ml_prep_*.sh`, etc.) sit in `analysis/` one level up and document the exact parameters used to produce the paper's numbers.
