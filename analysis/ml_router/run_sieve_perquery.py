#!/usr/bin/env python3
"""
Run SIEVE with best configs and output per-query recall.

Based on run_sieve.py but:
  - Automatically finds best config from existing results
  - Only runs the best config per (dataset, scenario)
  - Outputs per-query recall CSV compatible with extract_perquery_recall.py

Usage:
    cd ~/benchmarks/discrete
    python analysis/run_sieve_perquery.py
    python analysis/run_sieve_perquery.py --dataset arxiv
    python analysis/run_sieve_perquery.py --dataset arxiv --scenario original_and
"""

import argparse
import csv
import gc
import glob
import os
import pickle
import struct
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('DATA_DIR',
    os.path.expanduser("~/benchmarks/datasets/discrete")))
SIEVE_DIR = BASE_DIR / "SIEVE"
RESULTS_DIR = SIEVE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "analysis" / "perquery_recall"

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
VALIDATION_DATASETS = ['synth200', 'arxiv_fanns_real',
                       'lid50', 'lid80', 'lid100', 'lid120', 'lid150',
                       'synth5', 'synth30', 'synth100', 'hm21']
SCENARIOS = ['original_and', 'original_or', 'original_eq']
SCENARIO_MAP = {'original_and': 'and', 'original_or': 'or', 'original_eq': 'equal'}

K = 10
NUM_THREADS = 16


def find_best_configs():
    """Find best SIEVE config per (dataset, scenario) from existing results."""
    best = {}
    for f in sorted(glob.glob(str(RESULTS_DIR / "sieve_*_original_*.csv"))):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row['dataset'], row['scenario'])
                recall = float(row['recall_at_10'])
                if key not in best or recall > best[key]['recall']:
                    best[key] = {
                        'recall': recall,
                        'M': int(row['M']),
                        'ef_construction': int(row['ef_construction']),
                        'index_budget': float(row['index_budget']),
                        'hist_pct': float(row['hist_pct']),
                        'ef_search': int(row['ef_search']),
                    }
    return best


def read_bin_header(filepath):
    with open(filepath, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        d = struct.unpack('<I', f.read(4))[0]
    return n, d


def read_bin(filepath, dtype=np.float32):
    with open(filepath, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        d = struct.unpack('<I', f.read(4))[0]
        data = np.fromfile(f, dtype=dtype, count=n * d)
    return data.reshape(n, d), n, d


def read_gt_bin(filepath, k=10, nq=None):
    file_size = os.path.getsize(filepath)
    if nq is None:
        n_entries = file_size // 8
        nq = n_entries // k
    gt = []
    with open(filepath, 'rb') as f:
        for _ in range(nq):
            ids = []
            for _ in range(k):
                neighbor_id = struct.unpack('<I', f.read(4))[0]
                _dist = struct.unpack('<f', f.read(4))[0]
                ids.append(neighbor_id)
            gt.append(ids)
    return gt


def read_gt_txt(filepath, k=10):
    gt = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_query_filters(pickle_path, is_and=True):
    import hnswlib
    with open(pickle_path, 'rb') as f:
        sparse_mat = pickle.load(f)
    rows, cols = sparse_mat.nonzero()
    filter_dict = defaultdict(list)
    for row, col in zip(rows, cols):
        filter_dict[row].append(col)
    n_queries = sparse_mat.shape[0]
    filters = []
    for i in range(n_queries):
        if i in filter_dict:
            filters.append(hnswlib.QueryFilter(set(filter_dict[i]), is_and))
        else:
            filters.append(hnswlib.QueryFilter(set(), is_and))
    return filters


def compute_perquery_recall(results, gt, k=10):
    """Compute per-query recall@k. Returns list of floats."""
    n = min(len(gt), len(results))
    recalls = []
    for i in range(n):
        result_set = set(results[i][:k])
        gt_set = set(gt[i][:k])
        result_set.discard(4294967295)
        gt_set.discard(4294967295)
        if len(gt_set) > 0:
            recalls.append(len(result_set & gt_set) / min(k, len(gt_set)))
        else:
            recalls.append(0.0)
    return recalls


def run_single(dataset, scenario, config):
    """Run SIEVE for one (dataset, scenario) with given config. Returns per-query recalls."""
    import hnswlib

    is_and = scenario != 'original_or'
    suffix = SCENARIO_MAP[scenario]

    # Paths
    data_file = DATA_DIR / dataset / f'{dataset}_base.bin'
    label_dir = SIEVE_DIR / "sieve_labels"
    base_filter_file = label_dir / dataset / f'{scenario}_base_filters.bin'
    query_filter_file = label_dir / dataset / f'{scenario}_query_filters.pkl'

    if suffix == 'equal':
        query_file = DATA_DIR / dataset / f'{dataset}_query_equal.bin'
        gt_file = DATA_DIR / dataset / f'{dataset}_gt_equal.bin'
    elif suffix == 'and':
        query_file = DATA_DIR / dataset / f'{dataset}_query_and.bin'
        gt_file = DATA_DIR / dataset / f'{dataset}_gt_and.bin'
    else:  # or
        query_file = DATA_DIR / dataset / f'{dataset}_query_or.bin'
        gt_file = DATA_DIR / dataset / f'{dataset}_gt_or.bin'

    # GT can be .bin or .txt
    gt_txt = DATA_DIR / dataset / f'{dataset}_gt_{suffix}.txt'
    has_gt = gt_file.exists() or gt_txt.exists()

    # Check files
    for path, desc in [(data_file, 'data'), (base_filter_file, 'base filters'),
                       (query_filter_file, 'query filters'), (query_file, 'queries')]:
        if not path.exists():
            print(f"    ERROR: {desc} not found: {path}")
            return None
    if not has_gt:
        print(f"    ERROR: ground truth not found: {gt_file} or {gt_txt}")
        return None

    n, dim = read_bin_header(str(data_file))
    print(f"    n={n}, dim={dim}")

    # Load GT (gt_txt already defined above)
    if gt_txt.exists():
        gt = read_gt_txt(str(gt_txt), k=K)
    else:
        gt = read_gt_bin(str(gt_file), k=K)

    # Load query filters
    all_filters = load_query_filters(str(query_filter_file), is_and)
    hist_count = int(len(all_filters) * config['hist_pct'])
    historical_filters = all_filters[:hist_count]

    M = config['M']
    ef_construction = config['ef_construction']
    index_budget = config['index_budget']
    ef_search = config['ef_search']

    print(f"    Building index: M={M}, ef_c={ef_construction}, budget={index_budget}, hist={config['hist_pct']}")
    build_start = time.time()
    index = hnswlib.HierarchicalIndexFloat(
        str(data_file),
        str(base_filter_file),
        historical_filters,
        n, dim,
        M, ef_construction,
        int(index_budget * n),
        1000,    # bitvector_cutoff
        100000,  # workload_window_size
        True, True,
        NUM_THREADS,
    )
    build_time = time.time() - build_start
    print(f"    Build: {build_time:.1f}s")

    # Load queries
    queries, nq, qd = read_bin(str(query_file))
    nq = min(nq, len(all_filters), len(gt))

    # Search
    print(f"    Searching: ef={ef_search}, nq={nq}")
    results, times, cardinalities = index.batch_filter_search(
        queries[:nq], all_filters[:nq], nq, K, ef_search, NUM_THREADS)

    # Compute per-query recall
    recalls = compute_perquery_recall(results, gt[:nq], K)
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"    recall={avg_recall:.4f} ({len(recalls)} queries)")

    # Cleanup
    del index
    gc.collect()

    return recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', default=None)
    parser.add_argument('--scenario', nargs='+', default=SCENARIOS)
    parser.add_argument('--include-validation', action='store_true',
                        help='Include validation datasets')
    parser.add_argument('--validation-only', action='store_true',
                        help='Process only validation datasets')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: analysis/ml_router/perquery_recall/perquery_recall_sieve.csv). '
                             '设独立文件便于多 job 并行写不互相覆盖.')
    args = parser.parse_args()

    if args.dataset:
        pass
    elif args.validation_only:
        args.dataset = VALIDATION_DATASETS
    elif args.include_validation:
        args.dataset = DATASETS + VALIDATION_DATASETS
    else:
        args.dataset = DATASETS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find best configs
    best_configs = find_best_configs()
    print(f"Found {len(best_configs)} best configs from existing results\n")

    all_rows = []

    for dataset in args.dataset:
        for scenario in args.scenario:
            key = (dataset, scenario)
            if key not in best_configs:
                print(f"  SKIP {dataset}/{scenario}: no existing results")
                continue

            config = best_configs[key]
            scenario_std = SCENARIO_MAP[scenario]
            print(f"\n{'='*60}")
            print(f"SIEVE {dataset}/{scenario_std} (best recall={config['recall']:.4f})")
            print(f"{'='*60}")

            recalls = run_single(dataset, scenario, config)
            if recalls:
                params = f"M={config['M']}_b={config['index_budget']}_h={config['hist_pct']}_ef={config['ef_search']}"
                for qid, r in enumerate(recalls):
                    all_rows.append({
                        'query_id': qid,
                        'dataset': dataset,
                        'scenario': scenario_std,
                        'method': 'SIEVE',
                        'recall_at_10': round(r, 6),
                        'params': params,
                    })

    # Write output (append mode: load existing, merge, rewrite)
    output_file = Path(args.output) if args.output else OUTPUT_DIR / "perquery_recall_sieve.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['query_id', 'dataset', 'scenario', 'method', 'recall_at_10', 'params']

    # Load existing rows, excluding (dataset, scenario) combos we just re-ran
    new_keys = set((r['dataset'], r['scenario']) for r in all_rows)
    existing_rows = []
    if output_file.exists():
        with open(str(output_file)) as f:
            for row in csv.DictReader(f):
                if (row['dataset'], row['scenario']) not in new_keys:
                    existing_rows.append(row)
        print(f"Keeping {len(existing_rows)} existing rows, adding {len(all_rows)} new rows")

    merged = existing_rows + all_rows

    with open(str(output_file), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"Wrote {len(merged)} total rows to {output_file}")

    # Summary (over all data in file)
    from collections import Counter
    counts = Counter((r['dataset'], r['scenario']) for r in merged)
    for (ds, sc), count in sorted(counts.items()):
        recalls = [float(r['recall_at_10']) for r in merged if r['dataset'] == ds and r['scenario'] == sc]
        avg = sum(recalls) / len(recalls)
        print(f"  {ds:12s} {sc:6s}: {count} queries, avg recall={avg:.4f}")

    print(f"\nDone! Output: {output_file}")


if __name__ == '__main__':
    main()
