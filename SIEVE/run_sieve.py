#!/usr/bin/env python3
"""
SIEVE benchmark runner - single experiment mode.

Usage:
    python run_sieve.py --dataset arxiv --scenario fixed_eq \
        --M 32 --index_budget 2.0 --hist_pct 0.25

Output: CSV file with recall-QPS curve for varying ef_search values.
"""
import argparse
import gc
import os
import sys
import time
import struct
import pickle
import numpy as np
from collections import defaultdict

# Paths (can be overridden by args)
DEFAULT_DATA_DIR = os.path.expanduser("~/benchmarks/datasets/discrete")
DEFAULT_BENCHMARK_DIR = os.path.expanduser("~/benchmarks/discrete")
DEFAULT_NUM_THREADS = 16


def get_label_file(path):
    """V2 数据集标签从 0 开始，SIEVE 内部减 1 会变成 -1。
    如果存在 _1based 版本就自动使用，让 SIEVE 的 -1 逻辑正常工作。"""
    base, ext = os.path.splitext(path)
    onebased = f"{base}_1based{ext}"
    if os.path.exists(onebased):
        print(f"  Using 1-based label file: {os.path.basename(onebased)}")
        return onebased
    return path


def read_bin_header(filepath):
    """Read only the header (n, dim) from a BigANN-style binary file."""
    with open(filepath, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        d = struct.unpack('<I', f.read(4))[0]
    return n, d


def read_bin(filepath, dtype=np.float32):
    """Read BigANN-style binary file: (n_points uint32, dim uint32, data)."""
    with open(filepath, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        d = struct.unpack('<I', f.read(4))[0]
        data = np.fromfile(f, dtype=dtype, count=n * d)
    return data.reshape(n, d), n, d


def read_gt_bin(filepath, k=10, nq=None):
    """Read ground truth from binary file.

    Format: (id: int32, distance: float32) pairs, no header.
    Each query has k neighbors, total file size = nq * k * 8 bytes.
    """
    file_size = os.path.getsize(filepath)

    # Each entry is (id: 4 bytes, dist: 4 bytes) = 8 bytes
    # Try to infer nq from file size if not provided
    if nq is None:
        # Assume k=10 neighbors per query
        n_entries = file_size // 8
        nq = n_entries // k

    gt = []
    with open(filepath, 'rb') as f:
        for _ in range(nq):
            ids = []
            for _ in range(k):
                neighbor_id = struct.unpack('<I', f.read(4))[0]
                _dist = struct.unpack('<f', f.read(4))[0]  # skip distance
                ids.append(neighbor_id)
            gt.append(ids)
    return gt


def read_gt_txt(filepath, k=10):
    """Read ground truth from text file (space-separated IDs per line)."""
    gt = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_query_filters(pickle_path, is_and=True):
    """Load query filters from pickle and create QueryFilter objects."""
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


def compute_recall(results, gt, k=10):
    """Compute recall@k."""
    n = len(gt)
    if n == 0:
        return 0.0
    total_recall = 0
    for i in range(n):
        result_set = set(results[i][:k])
        gt_set = set(gt[i][:k])
        if len(gt_set) > 0:
            total_recall += len(result_set & gt_set) / min(k, len(gt_set))
    return total_recall / n


def validate_gt(gt, base_labels_file, query_labels_file, scenario, sample_size=100):
    """Validate ground truth by checking label consistency for a sample of queries."""
    print(f"Validating GT ({sample_size} samples)...")

    # Read base labels
    base_labels = []
    with open(base_labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                base_labels.append(set(int(x) - 1 for x in line.split(',')))  # 1-based to 0-based
            else:
                base_labels.append(set())

    # Read query labels
    query_labels = []
    with open(query_labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                query_labels.append(set(int(x) - 1 for x in line.split(',')))
            else:
                query_labels.append(set())

    # Check sample
    is_and = scenario != 'original_or'
    errors = 0
    checked = min(sample_size, len(gt), len(query_labels))

    for i in range(checked):
        if i >= len(gt) or i >= len(query_labels):
            break
        q_labels = query_labels[i]
        for neighbor_id in gt[i][:10]:
            if neighbor_id >= len(base_labels):
                errors += 1
                continue
            b_labels = base_labels[neighbor_id]

            if is_and:
                # AND/Equality: query labels must be subset of base labels
                if not q_labels.issubset(b_labels):
                    errors += 1
                    if errors <= 3:
                        print(f"  GT error query {i}: neighbor {neighbor_id} labels {b_labels} don't contain query labels {q_labels}")
            else:
                # OR: at least one query label must be in base labels
                if not q_labels.intersection(b_labels):
                    errors += 1
                    if errors <= 3:
                        print(f"  GT error query {i}: neighbor {neighbor_id} labels {b_labels} have no overlap with query labels {q_labels}")

    if errors > 0:
        print(f"  WARNING: {errors}/{checked} GT entries have label mismatches!")
        return False
    else:
        print(f"  GT validation passed ({checked} samples checked)")
        return True


def run_experiment(args):
    import hnswlib

    # Paths
    data_dir = args.data_dir
    benchmark_dir = args.benchmark_dir
    sieve_dir = os.path.join(benchmark_dir, "SIEVE")
    label_dir = os.path.join(sieve_dir, "sieve_labels")
    result_dir = os.path.join(sieve_dir, "results")

    os.makedirs(result_dir, exist_ok=True)

    # Data file
    data_file = os.path.join(data_dir, args.dataset, f'{args.dataset}_base.bin')
    n, dim = read_bin_header(data_file)
    print(f"Dataset: {args.dataset}, n={n}, dim={dim}")

    # Label files
    base_filter_file = os.path.join(label_dir, args.dataset, f'{args.scenario}_base_filters.bin')
    query_filter_file = os.path.join(label_dir, args.dataset, f'{args.scenario}_query_filters.pkl')

    # Query/GT files based on scenario
    # Original labels: 3 scenarios (Equality, Containment/AND, Overlap/OR)
    # Fixed-EQ: ACORN synthetic labels (12 labels), only arxiv & yfcc
    if args.scenario == 'original_eq':
        query_file = os.path.join(data_dir, args.dataset, f'{args.dataset}_query_equal.bin')
        gt_bin = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_equal.bin')
        gt_txt = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_equal.txt')
        gt_file = gt_txt if os.path.exists(gt_txt) else gt_bin
        is_and = True
        orig_base_labels = get_label_file(os.path.join(data_dir, args.dataset, 'label_base.txt'))
        orig_query_labels = get_label_file(os.path.join(data_dir, args.dataset, f'{args.dataset}_query_equal.txt'))
    elif args.scenario == 'original_and':
        query_file = os.path.join(data_dir, args.dataset, f'{args.dataset}_query_and.bin')
        gt_bin = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_and.bin')
        gt_txt = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_and.txt')
        gt_file = gt_txt if os.path.exists(gt_txt) else gt_bin
        is_and = True
        orig_base_labels = get_label_file(os.path.join(data_dir, args.dataset, 'label_base.txt'))
        orig_query_labels = get_label_file(os.path.join(data_dir, args.dataset, f'{args.dataset}_query_and.txt'))
    elif args.scenario == 'original_or':
        query_file = os.path.join(data_dir, args.dataset, f'{args.dataset}_query_or.bin')
        gt_bin = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_or.bin')
        gt_txt = os.path.join(data_dir, args.dataset, f'{args.dataset}_gt_or.txt')
        gt_file = gt_txt if os.path.exists(gt_txt) else gt_bin
        is_and = False
        orig_base_labels = get_label_file(os.path.join(data_dir, args.dataset, 'label_base.txt'))
        orig_query_labels = get_label_file(os.path.join(data_dir, args.dataset, f'{args.dataset}_query_or.txt'))
    elif args.scenario == 'fixed_eq':
        # ACORN synthetic labels (12 labels) with synthetic GT (only arxiv & yfcc)
        query_file = os.path.join(data_dir, args.dataset, f'{args.dataset}_query_equal.bin')
        gt_file = os.path.join(benchmark_dir, 'ACORN', 'synthetic_labels', args.dataset, 'gt_synthetic.txt')
        is_and = True
        orig_query_labels = None  # Skip validation for synthetic labels
        orig_base_labels = None
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    # Verify files exist
    for path, desc in [(data_file, 'data'), (base_filter_file, 'base filters'),
                       (query_filter_file, 'query filters'), (query_file, 'queries'),
                       (gt_file, 'ground truth')]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} file not found: {path}")
            sys.exit(1)

    # Load ground truth
    print("Loading ground truth...")
    if gt_file.endswith('.txt'):
        gt = read_gt_txt(gt_file, k=10)
    else:
        try:
            gt = read_gt_bin(gt_file, k=10)
        except Exception:
            gt_txt = gt_file.replace('.bin', '.txt')
            gt = read_gt_txt(gt_txt, k=10)
    print(f"  Loaded {len(gt)} GT entries")

    # Validate GT for original label scenarios
    if orig_base_labels and orig_query_labels and args.validate_gt:
        if os.path.exists(orig_base_labels) and os.path.exists(orig_query_labels):
            validate_gt(gt, orig_base_labels, orig_query_labels, args.scenario)

    # Load query filters
    print("Loading query filters...")
    all_filters = load_query_filters(query_filter_file, is_and)

    # Historical workload
    hist_count = int(len(all_filters) * args.hist_pct)
    historical_filters = all_filters[:hist_count]
    print(f"Historical workload: {hist_count}/{len(all_filters)} queries ({args.hist_pct*100:.0f}%)")

    # Parameters
    print(f"\nParameters:")
    print(f"  M={args.M}, ef_construction={args.ef_construction}")
    print(f"  index_budget={args.index_budget} ({int(args.index_budget * n)} vectors)")
    print(f"  hist_pct={args.hist_pct}, num_threads={args.num_threads}")

    # Build SIEVE index
    print("\nBuilding SIEVE index...")
    build_start = time.time()
    index = hnswlib.HierarchicalIndexFloat(
        data_file,
        base_filter_file,
        historical_filters,
        n,
        dim,
        args.M,
        args.ef_construction,
        int(args.index_budget * n),
        args.bitvector_cutoff,
        args.workload_window_size,
        True,   # enable_heterogeneous_indexing
        True,   # enable_heterogeneous_search
        args.num_threads,
    )
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f}s")

    # Estimate index size: vectors + graph structure
    # vectors: index_budget * n * dim * 4 bytes
    # graph: index_budget * n * M * 2 * 8 bytes (bidirectional links)
    n_indexed = int(args.index_budget * n)
    index_size_bytes = n_indexed * (dim * 4 + args.M * 2 * 8)
    index_size_mb = index_size_bytes / (1024 * 1024)
    print(f"Index size (est.): {index_size_mb:.1f} MB")

    # Load queries
    queries, nq, qd = read_bin(query_file)
    assert qd == dim, f"Query dim {qd} != dataset dim {dim}"
    nq = min(nq, len(all_filters), len(gt))
    print(f"Queries: {nq}")

    # Search with varying ef_search
    ef_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]

    # Output CSV
    tag = f"M{args.M}_b{args.index_budget}_h{args.hist_pct}"
    csv_path = os.path.join(result_dir, f'sieve_{args.dataset}_{args.scenario}_{tag}.csv')

    print(f"\nSearching (results -> {csv_path})...")
    per_query_csv_env = os.environ.get('PER_QUERY_CSV', '')
    save_per_query = bool(per_query_csv_env)
    if save_per_query:
        print(f"[per-query] PER_QUERY_CSV={per_query_csv_env}, will dump per-query latency+recall")
    with open(csv_path, 'w') as csv_f:
        csv_f.write('dataset,scenario,M,ef_construction,index_budget,hist_pct,'
                    'num_threads,ef_search,build_time_s,index_size_mb,search_time_s,qps,recall_at_10\n')

        for ef in ef_values:
            print(f"  ef={ef}...", end=' ', flush=True)
            # Force single thread when per-query timing requested (for accurate timing)
            threads = 1 if save_per_query else args.num_threads
            search_start = time.time()
            results, times, cardinalities = index.batch_filter_search(
                queries[:nq],
                all_filters[:nq],
                nq,
                10,       # k=10
                ef,
                threads,
            )
            search_time = time.time() - search_start

            recall = compute_recall(results, gt[:nq], k=10)
            qps = nq / search_time

            print(f"recall={recall:.4f}, QPS={qps:.0f}")

            csv_f.write(f'{args.dataset},{args.scenario},{args.M},'
                        f'{args.ef_construction},{args.index_budget},'
                        f'{args.hist_pct},{args.num_threads},{ef},{build_time:.2f},'
                        f'{index_size_mb:.1f},{search_time:.3f},{qps:.1f},{recall:.6f}\n')
            csv_f.flush()  # Flush after each line for resume capability

            if save_per_query:
                # Per-query CSV: query_id, latency_us, recall_at_10
                pq_path = (f'{per_query_csv_env}_M{args.M}_b{args.index_budget}_'
                           f'h{args.hist_pct}_ef{ef}_{args.scenario}.csv')
                gt_use = gt[:nq]
                with open(pq_path, 'w') as pq_f:
                    pq_f.write('query_id,latency_us,recall_at_10\n')
                    for qi in range(nq):
                        # per-query recall
                        if hasattr(results[qi], '__iter__'):
                            res_set = set(int(x) for x in results[qi])
                        else:
                            res_set = set()
                        gt_set = set(int(x) for x in gt_use[qi]) if qi < len(gt_use) else set()
                        if gt_set:
                            rq = len(res_set & gt_set) / min(10, len(gt_set))
                        else:
                            rq = 0.0
                        # times[qi] is per-query latency in seconds (chrono duration<float>
                        # from bindings.cpp line 220). Convert to microseconds.
                        lat_us = float(times[qi]) * 1e6
                        pq_f.write(f'{qi},{lat_us:.3f},{rq:.6f}\n')
                print(f"    [per-query] saved {pq_path}", flush=True)

    print(f"\nResults saved to {csv_path}")

    # Cleanup to free memory
    del index
    gc.collect()
    print("Index cleaned up")


def main():
    parser = argparse.ArgumentParser(description='Run single SIEVE experiment')
    parser.add_argument('--dataset', required=True,
                        choices=['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video',
                                 'synth200', 'arxiv_fanns_real',
                                 'lid50', 'lid80', 'lid100', 'lid120', 'lid150',
                                 'synth5', 'synth30', 'synth100', 'hm21',
                                 'synth_192d', 'synth_512d', 'synth_768d_hc', 'yahoo800k', 'dbpedia560k'])
    parser.add_argument('--scenario', required=True,
                        choices=['original_eq', 'original_and', 'original_or', 'fixed_eq'])
    parser.add_argument('--M', type=int, default=32,
                        help='HNSW bidirectional links per element')
    parser.add_argument('--ef_construction', type=int, default=40)
    parser.add_argument('--index_budget', type=float, default=2.0,
                        help='Index budget as fraction of dataset size')
    parser.add_argument('--hist_pct', type=float, default=0.25,
                        help='Fraction of queries used as historical workload')
    parser.add_argument('--bitvector_cutoff', type=int, default=1000)
    parser.add_argument('--workload_window_size', type=int, default=100000)
    parser.add_argument('--num_threads', type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--benchmark_dir', default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument('--validate_gt', action='store_true',
                        help='Validate ground truth labels (original scenarios only)')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
