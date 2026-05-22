#!/usr/bin/env python3
"""
Adaptive Query Routing Analysis for Filtered ANN.

Phase 1: Oracle upper bound + observation table + rule-based router backtest.
All using existing aggregated (per-dataset+scenario) data.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from load_results import load_all_results, get_pareto_front

ANALYSIS_DIR = Path(__file__).parent
DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
SCENARIOS = ['and', 'or', 'equal']
# Methods to consider for routing (exclude baselines from "methods to route to"
# but keep Post-filter as it can be the best option)
ROUTING_METHODS = ['UNG', 'ACORN-1', 'ACORN-γ', 'FilteredVamana', 'StitchedVamana',
                   'Post-filter', 'CAPS', 'SIEVE']
# Recall thresholds for QPS comparison
RECALL_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]


def load_dataset_metrics():
    """Load all dataset metrics JSONs into a dict."""
    metrics = {}
    for ds in DATASETS:
        path = ANALYSIS_DIR / f"{ds}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[ds] = json.load(f)
    return metrics


def compute_best_qps_at_recall(df, threshold):
    """For a group of results, find the highest QPS among points with recall >= threshold."""
    above = df[df['recall'] >= threshold]
    if above.empty:
        return None
    return above['QPS'].max()


def compute_max_recall(df):
    """Find the maximum recall achieved by any parameter config."""
    if df.empty:
        return None
    return df['recall'].max()


def oracle_analysis(all_df):
    """
    Experiment 1: Oracle upper bound analysis.

    For each (dataset, scenario):
    - Find which method achieves best max-recall and best QPS@recall>=T
    - Compare Oracle (pick best per cell) vs fixed single method
    """
    print("=" * 80)
    print("EXPERIMENT 1: Oracle Upper Bound Analysis")
    print("=" * 80)

    # --- Part A: Max recall achieved per (dataset, scenario, method) ---
    print("\n--- Part A: Maximum Recall Achieved ---")
    max_recall_table = {}
    for ds in DATASETS:
        for sc in SCENARIOS:
            subset = all_df[(all_df['dataset'] == ds) & (all_df['scenario'] == sc)]
            max_recall_table[(ds, sc)] = {}
            for method in ROUTING_METHODS:
                method_df = subset[subset['method'] == method]
                mr = compute_max_recall(method_df)
                if mr is not None:
                    max_recall_table[(ds, sc)][method] = mr

    # Build a DataFrame for display
    rows = []
    for ds in DATASETS:
        for sc in SCENARIOS:
            row = {'dataset': ds, 'scenario': sc}
            cell = max_recall_table.get((ds, sc), {})
            for m in ROUTING_METHODS:
                row[m] = cell.get(m, None)
            # Oracle: best across methods
            vals = [v for v in cell.values() if v is not None]
            row['Oracle'] = max(vals) if vals else None
            # Best method name
            if cell:
                best_m = max(cell, key=cell.get)
                row['Best_Method'] = best_m
            else:
                row['Best_Method'] = 'N/A'
            rows.append(row)

    recall_df = pd.DataFrame(rows)
    print("\nMax Recall per (dataset, scenario):")
    display_cols = ['dataset', 'scenario'] + [m for m in ROUTING_METHODS if m in recall_df.columns] + ['Oracle', 'Best_Method']
    with pd.option_context('display.max_columns', 20, 'display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(recall_df[display_cols].to_string(index=False))

    # --- Part B: QPS at recall thresholds ---
    print("\n--- Part B: Best QPS at Recall Thresholds ---")
    for threshold in RECALL_THRESHOLDS:
        print(f"\n  Recall >= {threshold}:")
        qps_rows = []
        for ds in DATASETS:
            for sc in SCENARIOS:
                subset = all_df[(all_df['dataset'] == ds) & (all_df['scenario'] == sc)]
                row = {'dataset': ds, 'scenario': sc}
                best_qps = -1
                best_method = 'N/A'
                for method in ROUTING_METHODS:
                    method_df = subset[subset['method'] == method]
                    qps = compute_best_qps_at_recall(method_df, threshold)
                    row[method] = qps
                    if qps is not None and qps > best_qps:
                        best_qps = qps
                        best_method = method
                row['Oracle_QPS'] = best_qps if best_qps > 0 else None
                row['Best_Method'] = best_method
                qps_rows.append(row)

        qps_df = pd.DataFrame(qps_rows)
        with pd.option_context('display.max_columns', 20, 'display.width', 200, 'display.float_format', '{:.1f}'.format):
            print(qps_df[['dataset', 'scenario', 'Oracle_QPS', 'Best_Method']].to_string(index=False))

    # --- Part C: How much does Oracle improve over each fixed method? ---
    print("\n--- Part C: Oracle vs Fixed Single Method ---")
    print("(Average max-recall across all dataset×scenario cells)\n")

    method_avg_recall = {}
    n_cells = 0
    oracle_sum = 0

    for ds in DATASETS:
        for sc in SCENARIOS:
            cell = max_recall_table.get((ds, sc), {})
            vals = [v for v in cell.values() if v is not None]
            if vals:
                oracle_sum += max(vals)
                n_cells += 1
                for m, v in cell.items():
                    method_avg_recall.setdefault(m, []).append(v)

    oracle_avg = oracle_sum / n_cells if n_cells > 0 else 0
    print(f"  Oracle (pick best per cell):  {oracle_avg:.4f}  (across {n_cells} cells)")
    for m in ROUTING_METHODS:
        vals = method_avg_recall.get(m, [])
        if vals:
            # For cells where method has no data, use 0
            all_vals = []
            for ds in DATASETS:
                for sc in SCENARIOS:
                    cell = max_recall_table.get((ds, sc), {})
                    all_vals.append(cell.get(m, 0))
            avg = np.mean(all_vals)
            present = len(vals)
            print(f"  Always-{m:20s}: {avg:.4f}  (present in {present}/{n_cells} cells)")

    # --- Part D: Winner frequency ---
    print("\n--- Part D: How Often Each Method Wins (Max Recall) ---")
    win_count = {}
    for ds in DATASETS:
        for sc in SCENARIOS:
            cell = max_recall_table.get((ds, sc), {})
            if cell:
                best_m = max(cell, key=cell.get)
                win_count[best_m] = win_count.get(best_m, 0) + 1

    for m, count in sorted(win_count.items(), key=lambda x: -x[1]):
        print(f"  {m:20s}: {count}/{n_cells} cells")

    return recall_df, max_recall_table


def observation_table(all_df, ds_metrics):
    """
    Build the observation table: what scenario features predict which method wins.
    """
    print("\n" + "=" * 80)
    print("OBSERVATION TABLE: Scenario Features → Best Method")
    print("=" * 80)

    rows = []
    for ds in DATASETS:
        m = ds_metrics.get(ds, {})
        lid = m.get('lid', {}).get('mean', None)
        rc = m.get('rc', {}).get('trimmed_mean', None)
        df_mean = m.get('distribution_factor', {}).get('mean_swd', None)
        label_card = m.get('label_stats', {}).get('label_cardinality', None)
        label_entropy = m.get('label_stats', {}).get('label_entropy', None)
        n_vectors = m.get('n_vectors', None)

        for sc in SCENARIOS:
            subset = all_df[(all_df['dataset'] == ds) & (all_df['scenario'] == sc)]

            # Find best method by max recall
            best_recall = -1
            best_method_recall = 'N/A'
            method_recalls = {}
            for method in ROUTING_METHODS:
                method_df = subset[subset['method'] == method]
                mr = compute_max_recall(method_df)
                if mr is not None:
                    method_recalls[method] = mr
                    if mr > best_recall:
                        best_recall = mr
                        best_method_recall = method

            # Find best method by QPS at recall>=0.9
            best_qps_90 = -1
            best_method_qps = 'N/A'
            for method in ROUTING_METHODS:
                method_df = subset[subset['method'] == method]
                qps = compute_best_qps_at_recall(method_df, 0.9)
                if qps is not None and qps > best_qps_90:
                    best_qps_90 = qps
                    best_method_qps = method

            # Top-2 methods by recall
            sorted_methods = sorted(method_recalls.items(), key=lambda x: -x[1])
            top2 = [f"{m}({v:.3f})" for m, v in sorted_methods[:3]]

            rows.append({
                'dataset': ds,
                'scenario': sc,
                'n_vectors': n_vectors,
                'LID': f"{lid:.1f}" if lid else 'N/A',
                'RC': f"{rc:.2f}" if rc else 'N/A',
                'DF(mean_swd)': f"{df_mean:.3f}" if df_mean else 'N/A',
                'label_card': label_card,
                'label_entropy': f"{label_entropy:.2f}" if label_entropy else 'N/A',
                'best_recall_method': best_method_recall,
                'best_recall': f"{best_recall:.4f}" if best_recall > 0 else 'N/A',
                'best_qps@90_method': best_method_qps,
                'best_qps@90': f"{best_qps_90:.1f}" if best_qps_90 > 0 else 'N/A',
                'top_methods': ' > '.join(top2),
            })

    obs_df = pd.DataFrame(rows)
    with pd.option_context('display.max_columns', 20, 'display.width', 250, 'display.max_colwidth', 50):
        print(obs_df.to_string(index=False))

    return obs_df


def rule_based_router(dataset, scenario, ds_metrics):
    """
    Rule-based router: given dataset features and scenario, recommend a method.

    Rules derived from empirical observations on 6 datasets × 3 scenarios:

    EQUALITY: UNG wins ALL 6 datasets (physical partitioning = direct group access).
      Even on high-DF yfcc, UNG achieves recall=1.0 because equality queries
      target exactly one partition.

    AND (containment): Split between SIEVE and UNG.
      - SIEVE wins on datasets with many labels (arxiv=4231, yfcc=181931, ytb_audio=3862)
        Its workload-aware sub-index selection handles multi-label containment well.
      - UNG wins on datasets with few labels (LAION1M=30, tripclick=29) where
        partition count is small and containment is efficiently resolved.
      - Exception: ytb_video (labels=3862 but LID=236) → UNG wins because
        SIEVE's sub-indices struggle with very high intrinsic dimensionality.

    OR (overlap): Post-filter is safest; high-LID is the exception.
      - Post-filter or ACORN-γ work well (global graph, relaxed acceptance).
      - On ytb_video (LID=236), all methods struggle; UNG is least bad.
    """
    m = ds_metrics.get(dataset, {})
    label_card = m.get('label_stats', {}).get('label_cardinality', 0)
    lid = m.get('lid', {}).get('mean', 0)

    if scenario == 'equal':
        # UNG dominates all equality scenarios regardless of DF or labels
        return 'UNG'

    elif scenario == 'and':
        if lid > 100:
            # Very high LID: SIEVE sub-indices degrade, UNG more robust
            return 'UNG'
        elif label_card < 100:
            # Few labels: UNG partitions are coarse-grained, efficient
            return 'UNG'
        else:
            # Many labels + moderate LID: SIEVE's workload-aware routing shines
            return 'SIEVE'

    elif scenario == 'or':
        if lid > 100:
            # Very high LID: everything struggles, UNG least bad
            return 'UNG'
        else:
            # Global graph methods handle OR well
            return 'Post-filter'

    return 'Post-filter'


def backtest_router(all_df, ds_metrics):
    """
    Experiment 2: Backtest rule-based router vs fixed single method.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Rule-Based Router Backtest")
    print("=" * 80)

    # For each (dataset, scenario), compute each method's best recall and QPS@90
    results = []
    for ds in DATASETS:
        for sc in SCENARIOS:
            subset = all_df[(all_df['dataset'] == ds) & (all_df['scenario'] == sc)]

            cell = {'dataset': ds, 'scenario': sc}

            # Router's recommendation
            recommended = rule_based_router(ds, sc, ds_metrics)
            cell['router_choice'] = recommended

            # Compute max recall and QPS@0.9 for each method
            for method in ROUTING_METHODS:
                method_df = subset[subset['method'] == method]
                cell[f'{method}_recall'] = compute_max_recall(method_df) or 0
                cell[f'{method}_qps90'] = compute_best_qps_at_recall(method_df, 0.9) or 0

            # Router's score = the recommended method's score
            cell['router_recall'] = cell.get(f'{recommended}_recall', 0)
            cell['router_qps90'] = cell.get(f'{recommended}_qps90', 0)

            # Oracle score: pick recall-optimal method, report its QPS
            # (break ties by QPS to get the best method at max recall)
            best_oracle_recall = -1
            best_oracle_qps = 0
            for m in ROUTING_METHODS:
                r = cell.get(f'{m}_recall', 0)
                q = cell.get(f'{m}_qps90', 0)
                if r > best_oracle_recall or (r == best_oracle_recall and q > best_oracle_qps):
                    best_oracle_recall = r
                    best_oracle_qps = q
            cell['oracle_recall'] = best_oracle_recall
            cell['oracle_qps90'] = best_oracle_qps

            results.append(cell)

    results_df = pd.DataFrame(results)

    # --- Summary: Average recall across all cells ---
    print("\n--- Average Max-Recall Across All (dataset, scenario) Cells ---\n")
    print(f"  {'Method':<25s} {'Avg Recall':>12s} {'Worst Case':>12s} {'Avg QPS@90':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    # Oracle
    avg_r = results_df['oracle_recall'].mean()
    worst_r = results_df['oracle_recall'].min()
    avg_q = results_df['oracle_qps90'].mean()
    print(f"  {'Oracle':<25s} {avg_r:>12.4f} {worst_r:>12.4f} {avg_q:>12.1f}")

    # Rule router
    avg_r = results_df['router_recall'].mean()
    worst_r = results_df['router_recall'].min()
    avg_q = results_df['router_qps90'].mean()
    print(f"  {'Rule Router':<25s} {avg_r:>12.4f} {worst_r:>12.4f} {avg_q:>12.1f}")

    # Fixed methods
    for method in ROUTING_METHODS:
        col_r = f'{method}_recall'
        col_q = f'{method}_qps90'
        if col_r in results_df.columns:
            avg_r = results_df[col_r].mean()
            worst_r = results_df[col_r].min()
            avg_q = results_df[col_q].mean()
            print(f"  Always-{method:<17s} {avg_r:>12.4f} {worst_r:>12.4f} {avg_q:>12.1f}")

    # --- Per-cell comparison ---
    print("\n--- Per-Cell Router Decisions ---\n")
    print(f"  {'Dataset':<12s} {'Scenario':<8s} {'Router Choice':<20s} {'Router Recall':>14s} {'Oracle Recall':>14s} {'Gap':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*20} {'-'*14} {'-'*14} {'-'*8}")

    for _, row in results_df.iterrows():
        gap = row['oracle_recall'] - row['router_recall']
        print(f"  {row['dataset']:<12s} {row['scenario']:<8s} {row['router_choice']:<20s} "
              f"{row['router_recall']:>14.4f} {row['oracle_recall']:>14.4f} {gap:>8.4f}")

    # --- Per-scenario summary ---
    print("\n--- Per-Scenario Summary ---\n")
    for sc in SCENARIOS:
        sc_df = results_df[results_df['scenario'] == sc]
        print(f"  Scenario: {sc}")
        print(f"    Oracle avg recall:  {sc_df['oracle_recall'].mean():.4f}")
        print(f"    Router avg recall:  {sc_df['router_recall'].mean():.4f}")
        # Best fixed method for this scenario
        best_fixed = None
        best_fixed_recall = -1
        for method in ROUTING_METHODS:
            col = f'{method}_recall'
            avg = sc_df[col].mean()
            if avg > best_fixed_recall:
                best_fixed_recall = avg
                best_fixed = method
        print(f"    Best fixed method:  {best_fixed} ({best_fixed_recall:.4f})")
        print()

    return results_df


def per_scenario_winner_analysis(all_df, ds_metrics):
    """
    Detailed analysis: for each scenario type, which method wins on which datasets and why.
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: Per-Scenario Winner Patterns")
    print("=" * 80)

    for sc in SCENARIOS:
        print(f"\n--- Scenario: {sc.upper()} ---")
        for ds in DATASETS:
            subset = all_df[(all_df['dataset'] == ds) & (all_df['scenario'] == sc)]
            m = ds_metrics.get(ds, {})
            df_mean = m.get('distribution_factor', {}).get('mean_swd', 0)
            label_card = m.get('label_stats', {}).get('label_cardinality', 0)
            lid = m.get('lid', {}).get('mean', 0)

            method_recalls = {}
            for method in ROUTING_METHODS:
                method_df = subset[subset['method'] == method]
                mr = compute_max_recall(method_df)
                if mr is not None:
                    method_recalls[method] = mr

            sorted_m = sorted(method_recalls.items(), key=lambda x: -x[1])
            top_str = ', '.join([f"{m}={v:.4f}" for m, v in sorted_m[:4]])

            print(f"  {ds:<12s} [LID={lid:.1f}, DF={df_mean:.3f}, labels={label_card}]")
            print(f"    {top_str}")


def main():
    print("Loading all results...")
    all_df = load_all_results()
    print(f"Loaded {len(all_df)} rows")
    print(f"Methods: {sorted(all_df['method'].unique())}")
    print(f"Datasets: {sorted(all_df['dataset'].unique())}")
    print(f"Scenarios: {sorted(all_df['scenario'].unique())}")

    # Check coverage
    print("\n--- Data Coverage ---")
    for method in ROUTING_METHODS:
        mdf = all_df[all_df['method'] == method]
        combos = set(zip(mdf['dataset'], mdf['scenario']))
        print(f"  {method:<20s}: {len(mdf):>5d} rows, {len(combos):>2d}/18 cells")

    ds_metrics = load_dataset_metrics()
    print(f"\nLoaded metrics for: {list(ds_metrics.keys())}")

    # Print dataset characteristics summary
    print("\n--- Dataset Characteristics ---")
    print(f"  {'Dataset':<12s} {'N':>10s} {'Dim':>5s} {'LID':>6s} {'RC':>6s} {'DF(swd)':>8s} {'Labels':>8s} {'Entropy':>8s}")
    for ds in DATASETS:
        m = ds_metrics.get(ds, {})
        print(f"  {ds:<12s} {m.get('n_vectors',0):>10d} {m.get('dimension',0):>5d} "
              f"{m.get('lid',{}).get('mean',0):>6.1f} {m.get('rc',{}).get('trimmed_mean',0):>6.2f} "
              f"{m.get('distribution_factor',{}).get('mean_swd',0):>8.3f} "
              f"{m.get('label_stats',{}).get('label_cardinality',0):>8d} "
              f"{m.get('label_stats',{}).get('label_entropy',0):>8.2f}")

    # Run analyses
    recall_df, max_recall_table = oracle_analysis(all_df)
    obs_df = observation_table(all_df, ds_metrics)
    per_scenario_winner_analysis(all_df, ds_metrics)
    results_df = backtest_router(all_df, ds_metrics)

    # Save results
    output_dir = ANALYSIS_DIR / "routing_results"
    output_dir.mkdir(exist_ok=True)
    recall_df.to_csv(output_dir / "oracle_max_recall.csv", index=False)
    obs_df.to_csv(output_dir / "observation_table.csv", index=False)
    results_df.to_csv(output_dir / "router_backtest.csv", index=False)
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
