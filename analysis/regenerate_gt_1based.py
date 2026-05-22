#!/usr/bin/env python3
"""
用 1-based 标签重新生成 V2 数据集的 Ground Truth

原 GT 是用 0-based 标签生成的，但 UNG 需要 1-based 标签（label 0 是哨兵）。
这个脚本用 1-based 标签重新计算 GT，确保所有方法用统一的标签和 GT。

用法:
    python analysis/regenerate_gt_1based.py [dataset]
    python analysis/regenerate_gt_1based.py synth_192d
    python analysis/regenerate_gt_1based.py all
"""
import os
import sys
import struct
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

DATA_DIR = os.path.expanduser("~/benchmarks/datasets/discrete")
V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS = ["and", "or", "equal"]
K = 10


def read_fvecs(path):
    """Read fvecs format: each vector prefixed with dimension (int32)"""
    # Read first 4 bytes as int32 to get dimension
    with open(path, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]

    # Efficient numpy reading
    # Each record: 1 int32 (dim) + d float32 values = (1 + d) float32
    data = np.fromfile(path, dtype=np.float32)
    # Reshape to (n, 1+d), then discard first column (dimension prefix)
    data = data.reshape(-1, d + 1)[:, 1:].copy()
    return data


def read_bin(path):
    """Read UNG bin format: N, D header (uint32) + N*D float32"""
    with open(path, 'rb') as f:
        n, d = struct.unpack('II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, d)
    return data


def read_labels_1based(path):
    """Read 1-based label file, return list of sets"""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(set(int(x) for x in line.split(',') if x))
            else:
                labels.append(set())
    return labels


def check_constraint(base_labels, query_labels, scenario):
    """Check if base labels satisfy query constraint"""
    if scenario == 'and':
        # Containment: base labels must contain all query labels
        return query_labels.issubset(base_labels)
    elif scenario == 'or':
        # Overlap: base and query must share at least one label
        return len(base_labels & query_labels) > 0
    elif scenario == 'equal':
        # Equality: base labels must equal query labels exactly
        return base_labels == query_labels
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# Global variables for multiprocessing workers
_base_vecs = None
_base_labels = None
_scenario = None
_k = None


def _init_worker(base_vecs, base_labels, scenario, k):
    """Initialize worker process with shared data"""
    global _base_vecs, _base_labels, _scenario, _k
    _base_vecs = base_vecs
    _base_labels = base_labels
    _scenario = scenario
    _k = k


def _process_query(args):
    """Process single query (for parallel execution)"""
    qi, q_vec, q_labels = args

    # Compute distances to all base vectors
    dists = np.sum((_base_vecs - q_vec) ** 2, axis=1)

    # Sort by distance
    sorted_indices = np.argsort(dists)

    # Find K nearest that satisfy constraint
    neighbors = []
    neighbor_dists = []
    for idx in sorted_indices:
        if check_constraint(_base_labels[idx], q_labels, _scenario):
            neighbors.append(idx)
            neighbor_dists.append(float(dists[idx]))
            if len(neighbors) >= _k:
                break

    # Pad with -1/inf if not enough neighbors found
    while len(neighbors) < _k:
        neighbors.append(-1)
        neighbor_dists.append(float('inf'))

    return qi, neighbors, neighbor_dists


def compute_gt(base_vecs, query_vecs, base_labels, query_labels, scenario, k=10):
    """Compute ground truth: K nearest neighbors satisfying label constraint

    Returns:
        gt_ids: list of lists of neighbor IDs
        gt_dists: list of lists of distances
    """
    n_queries = len(query_vecs)
    n_workers = min(cpu_count(), 16)

    print(f"  Using {n_workers} workers...")

    # Prepare arguments
    args = [(i, query_vecs[i], query_labels[i]) for i in range(n_queries)]

    # Initialize global data
    _init_worker(base_vecs, base_labels, scenario, k)

    # Process in parallel (using fork to share numpy arrays)
    gt_ids = [None] * n_queries
    gt_dists = [None] * n_queries
    with Pool(n_workers, initializer=_init_worker,
              initargs=(base_vecs, base_labels, scenario, k)) as pool:
        for i, (qi, neighbors, dists) in enumerate(pool.imap_unordered(_process_query, args, chunksize=10)):
            gt_ids[qi] = neighbors
            gt_dists[qi] = dists
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_queries} queries...", end='\r')

    print(f"  Processed {n_queries} queries" + " " * 20)
    return gt_ids, gt_dists


def save_gt_txt(gt, path):
    """Save GT as text file (space-separated IDs per line)"""
    with open(path, 'w') as f:
        for neighbors in gt:
            f.write(' '.join(str(n) for n in neighbors) + '\n')


def save_gt_bin(gt_ids, gt_dists, path, k=10):
    """Save GT as UNG binary format: NO header, (int32 ID, float32 dist) pairs

    UNG expects: num_queries * K pairs of (int32, float32), interleaved
    Total size: N * K * 8 bytes
    """
    with open(path, 'wb') as f:
        for neighbors, dists in zip(gt_ids, gt_dists):
            for nid, dist in zip(neighbors[:k], dists[:k]):
                # Write (int32 ID, float32 distance) pair
                f.write(struct.pack('if', nid, dist))


def process_dataset(dataset):
    """Process one dataset: regenerate GT for all scenarios"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print(f"{'='*60}")

    ds_dir = f"{DATA_DIR}/{dataset}"

    # Check if 1-based label files exist
    base_label_file = f"{ds_dir}/label_base_1based.txt"
    if not os.path.exists(base_label_file):
        print(f"  ERROR: 1-based label file not found: {base_label_file}")
        return False

    # Load base vectors
    fvecs_file = f"{ds_dir}/{dataset}_base.fvecs"
    bin_file = f"{ds_dir}/{dataset}_base.bin"

    if os.path.exists(fvecs_file):
        print(f"  Loading base vectors from fvecs...")
        base_vecs = read_fvecs(fvecs_file)
    elif os.path.exists(bin_file):
        print(f"  Loading base vectors from bin...")
        base_vecs = read_bin(bin_file)
    else:
        print(f"  ERROR: No base vector file found")
        return False

    print(f"  Base: {base_vecs.shape[0]} vectors, dim={base_vecs.shape[1]}")

    # Load 1-based base labels
    print(f"  Loading 1-based base labels...")
    base_labels = read_labels_1based(base_label_file)
    print(f"  Loaded {len(base_labels)} label sets")

    for scenario in SCENARIOS:
        suffix = scenario if scenario != 'and' else 'and'

        # Load query vectors
        query_fvecs = f"{ds_dir}/{dataset}_query_{suffix}.fvecs"
        query_bin = f"{ds_dir}/{dataset}_query_{suffix}.bin"

        if os.path.exists(query_fvecs):
            query_vecs = read_fvecs(query_fvecs)
        elif os.path.exists(query_bin):
            query_vecs = read_bin(query_bin)
        else:
            print(f"  [{scenario.upper()}] Query file not found, skipping")
            continue

        # Load 1-based query labels
        query_label_file = f"{ds_dir}/{dataset}_query_{suffix}_1based.txt"
        if not os.path.exists(query_label_file):
            print(f"  [{scenario.upper()}] 1-based query labels not found, skipping")
            continue

        query_labels = read_labels_1based(query_label_file)

        print(f"\n  [{scenario.upper()}] {len(query_vecs)} queries")

        # Compute GT
        gt_ids, gt_dists = compute_gt(base_vecs, query_vecs, base_labels, query_labels, scenario, K)

        # Count valid neighbors
        valid_count = sum(1 for row in gt_ids for n in row if n >= 0)
        total_count = len(gt_ids) * K
        print(f"  Valid neighbors: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")

        # Backup old GT files
        old_txt = f"{ds_dir}/{dataset}_gt_{suffix}.txt"
        old_bin = f"{ds_dir}/{dataset}_gt_{suffix}.bin"

        if os.path.exists(old_txt):
            backup = f"{ds_dir}/{dataset}_gt_{suffix}_0based.txt"
            if not os.path.exists(backup):
                os.rename(old_txt, backup)
                print(f"  Backed up: {os.path.basename(backup)}")

        if os.path.exists(old_bin):
            backup = f"{ds_dir}/{dataset}_gt_{suffix}_0based.bin"
            if not os.path.exists(backup):
                os.rename(old_bin, backup)
                print(f"  Backed up: {os.path.basename(backup)}")

        # Save new GT
        save_gt_txt(gt_ids, old_txt)
        save_gt_bin(gt_ids, gt_dists, old_bin)
        print(f"  Saved: {dataset}_gt_{suffix}.txt/bin")

    return True


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'all':
            datasets = V2_DATASETS
        elif arg in V2_DATASETS:
            datasets = [arg]
        else:
            print(f"Unknown dataset: {arg}")
            print(f"Available: {V2_DATASETS} or 'all'")
            sys.exit(1)
    else:
        datasets = V2_DATASETS

    print("="*60)
    print("Regenerating GT with 1-based labels")
    print("="*60)
    print(f"Datasets: {datasets}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"K: {K}")

    for dataset in datasets:
        process_dataset(dataset)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print("\n旧 GT 已备份为 *_0based.txt/bin")
    print("新 GT 使用 1-based 标签，与 UNG 兼容")


if __name__ == "__main__":
    main()
