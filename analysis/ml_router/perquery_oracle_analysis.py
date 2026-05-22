#!/usr/bin/env python3
"""
Per-query Oracle analysis for adaptive routing.

Reads per-query recall data and computes:
1. Per-query Oracle: for each query, pick the method with highest recall
2. Compare Oracle vs always-using-one-method
3. Analyze per-query recall variance across methods
4. Count queries where routing makes a difference
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "perquery_recall" / "perquery_recall.csv"


def load_data(filepath):
    """Load per-query recall data. Returns dict of (dataset, scenario, query_id) -> {method: recall}"""
    data = defaultdict(dict)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'], row['scenario'], int(row['query_id']))
            data[key][row['method']] = float(row['recall_at_10'])
    return data


def analyze_scenario(data, dataset, scenario):
    """Analyze a single (dataset, scenario) combination."""
    # Get all queries for this (dataset, scenario)
    queries = {qid: methods for (ds, sc, qid), methods in data.items()
               if ds == dataset and sc == scenario}

    if not queries:
        return None

    methods = sorted(set(m for q in queries.values() for m in q))
    nq = len(queries)

    # Per-method avg recall
    method_avg = {}
    for m in methods:
        recalls = [queries[qid].get(m, 0) for qid in queries]
        method_avg[m] = sum(recalls) / len(recalls)

    # Per-query Oracle: for each query, pick max recall across methods
    oracle_recalls = []
    oracle_choices = defaultdict(int)
    queries_where_routing_helps = 0
    recall_gaps = []

    for qid in sorted(queries.keys()):
        q = queries[qid]
        if not q:
            continue

        best_method = max(q, key=q.get)
        best_recall = q[best_method]
        oracle_recalls.append(best_recall)
        oracle_choices[best_method] += 1

        # How much better is Oracle vs best single method?
        best_single = max(methods, key=lambda m: method_avg.get(m, 0))
        single_recall = q.get(best_single, 0)
        gap = best_recall - single_recall
        recall_gaps.append(gap)
        if gap > 0.01:  # >1% improvement
            queries_where_routing_helps += 1

    oracle_avg = sum(oracle_recalls) / len(oracle_recalls) if oracle_recalls else 0
    best_single = max(methods, key=lambda m: method_avg.get(m, 0))

    return {
        'dataset': dataset,
        'scenario': scenario,
        'nq': nq,
        'methods': methods,
        'method_avg': method_avg,
        'oracle_avg': oracle_avg,
        'oracle_choices': dict(oracle_choices),
        'best_single_method': best_single,
        'best_single_avg': method_avg[best_single],
        'oracle_gain': oracle_avg - method_avg[best_single],
        'queries_routing_helps': queries_where_routing_helps,
        'pct_routing_helps': queries_where_routing_helps / nq * 100 if nq > 0 else 0,
        'avg_gap': sum(recall_gaps) / len(recall_gaps) if recall_gaps else 0,
        'max_gap': max(recall_gaps) if recall_gaps else 0,
    }


def analyze_recall_distribution(data, dataset, scenario):
    """Analyze distribution of per-query recall differences between methods."""
    queries = {qid: methods for (ds, sc, qid), methods in data.items()
               if ds == dataset and sc == scenario}

    methods = sorted(set(m for q in queries.values() for m in q))
    nq = len(queries)

    # For each query, compute max - min recall across methods
    spreads = []
    for qid in sorted(queries.keys()):
        q = queries[qid]
        if len(q) < 2:
            continue
        vals = list(q.values())
        spreads.append(max(vals) - min(vals))

    if not spreads:
        return None

    # Percentiles
    spreads.sort()
    n = len(spreads)
    return {
        'p25': spreads[n // 4],
        'p50': spreads[n // 2],
        'p75': spreads[3 * n // 4],
        'p90': spreads[int(n * 0.9)],
        'p99': spreads[int(n * 0.99)],
        'mean': sum(spreads) / n,
    }


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE

    print(f"Loading data from {filepath}...")
    data = load_data(filepath)

    # Get all (dataset, scenario) combos
    combos = sorted(set((ds, sc) for ds, sc, _ in data.keys()))

    print(f"Loaded {len(data)} query records across {len(combos)} (dataset, scenario) cells\n")

    # ============ Per-scenario analysis ============
    print("=" * 100)
    print("PER-QUERY ORACLE ANALYSIS")
    print("=" * 100)

    all_results = []
    for dataset, scenario in combos:
        result = analyze_scenario(data, dataset, scenario)
        if result:
            all_results.append(result)

    # Print table
    print(f"\n{'Dataset':12s} {'Scenario':8s} {'#Q':>5s}  ", end='')
    all_methods = sorted(set(m for r in all_results for m in r['methods']))
    for m in all_methods:
        print(f"{m:>18s}", end='')
    print(f"  {'Oracle':>8s} {'Gain':>7s} {'%Help':>7s}")
    print("-" * (40 + 18 * len(all_methods) + 25))

    for r in all_results:
        print(f"{r['dataset']:12s} {r['scenario']:8s} {r['nq']:5d}  ", end='')
        for m in all_methods:
            avg = r['method_avg'].get(m, float('nan'))
            marker = '*' if m == r['best_single_method'] else ' '
            print(f"{avg:>17.4f}{marker}", end='')
        print(f"  {r['oracle_avg']:>8.4f} {r['oracle_gain']:>+7.4f} {r['pct_routing_helps']:>6.1f}%")

    # ============ Summary statistics ============
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_queries = sum(r['nq'] for r in all_results)
    total_helped = sum(r['queries_routing_helps'] for r in all_results)
    avg_oracle = sum(r['oracle_avg'] * r['nq'] for r in all_results) / total_queries
    avg_best_single = sum(r['best_single_avg'] * r['nq'] for r in all_results) / total_queries

    print(f"Total queries: {total_queries}")
    print(f"Weighted avg Oracle recall:      {avg_oracle:.4f}")
    print(f"Weighted avg best-single recall: {avg_best_single:.4f}")
    print(f"Weighted avg Oracle gain:        {avg_oracle - avg_best_single:+.4f}")
    print(f"Queries where routing helps:     {total_helped}/{total_queries} ({total_helped/total_queries*100:.1f}%)")

    # ============ Oracle method distribution ============
    print(f"\n{'='*80}")
    print("ORACLE METHOD CHOICES (per query)")
    print(f"{'='*80}")

    for r in all_results:
        total = sum(r['oracle_choices'].values())
        choices_str = ", ".join(f"{m}: {c}/{total} ({c/total*100:.0f}%)"
                                for m, c in sorted(r['oracle_choices'].items(),
                                                   key=lambda x: -x[1]))
        print(f"  {r['dataset']:12s} {r['scenario']:8s}: {choices_str}")

    # ============ Recall spread analysis ============
    print(f"\n{'='*80}")
    print("RECALL SPREAD (max - min across methods, per query)")
    print(f"{'='*80}")
    print(f"{'Dataset':12s} {'Scenario':8s} {'Mean':>8s} {'P25':>8s} {'P50':>8s} {'P75':>8s} {'P90':>8s} {'P99':>8s}")
    print("-" * 70)

    for dataset, scenario in combos:
        dist = analyze_recall_distribution(data, dataset, scenario)
        if dist:
            print(f"{dataset:12s} {scenario:8s} {dist['mean']:>8.4f} {dist['p25']:>8.4f} "
                  f"{dist['p50']:>8.4f} {dist['p75']:>8.4f} {dist['p90']:>8.4f} {dist['p99']:>8.4f}")


if __name__ == '__main__':
    main()
