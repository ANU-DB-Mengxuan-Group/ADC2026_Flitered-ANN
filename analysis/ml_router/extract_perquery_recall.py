#!/usr/bin/env python3
"""
Extract per-query recall@10 from existing search result files.

Processes:
  1. UNG          - result CSV files (contain both GT and Result per query)
  2. FilteredVamana (DiskANN) - binary result files + GT text/bin files
  3. StitchedVamana (DiskANN) - same as FilteredVamana
  4. Pre-filter   - binary result files (prefilter_bruteforce_idx_uint32.bin)
  5. Post-filter  - binary result files (M=*_efc=*_ef=*_idx_uint32.bin)

For each method, finds the best parameter config (highest avg recall) per
(dataset, scenario) and extracts per-query recall for that config.

Output: CSV with columns: query_id, dataset, scenario, method, recall_at_10, params

Usage:
    cd ~/benchmarks/discrete
    python analysis/extract_perquery_recall.py
    python analysis/extract_perquery_recall.py --dataset arxiv
    python analysis/extract_perquery_recall.py --data-dir /path/to/datasets
"""

import argparse
import csv
import os
import struct
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
VALIDATION_DATASETS = ['synth200', 'arxiv_fanns_real',
                       'lid50', 'lid80', 'lid100', 'lid120', 'lid150',
                       'synth5', 'synth30', 'synth100', 'hm21']

# UNG uses containment/overlap/equality; we map to standard and/or/equal
SCENARIO_MAP = {'containment': 'and', 'overlap': 'or', 'equality': 'equal'}
SCENARIO_MAP_REV = {v: k for k, v in SCENARIO_MAP.items()}

K = 10


# ==================== Per-query recall ====================

def compute_perquery_recall(gt_list, res_list, k=10):
    """Compute per-query recall@k.

    Args:
        gt_list: list of lists, GT IDs per query
        res_list: list of lists, result IDs per query

    Returns:
        list of float recall values (0.0 to 1.0)
    """
    recalls = []
    for gt_ids, res_ids in zip(gt_list, res_list):
        gt_set = set(gt_ids[:k])
        res_set = set(res_ids[:k])
        # Remove invalid entries
        gt_set.discard(-1)
        gt_set.discard(4294967295)  # uint32 max = -1 in unsigned
        res_set.discard(-1)
        res_set.discard(4294967295)
        if len(gt_set) > 0:
            recall = len(gt_set & res_set) / min(k, len(gt_set))
        else:
            recall = 0.0
        recalls.append(recall)
    return recalls


# ==================== GT loaders ====================

def load_gt_txt(filepath, k=10):
    """Load ground truth from text file (space-separated IDs per line)."""
    gt = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_gt_bin(filepath, k=10):
    """Load GT from UNG binary format: (id:uint32, dist:float32) pairs, no header."""
    file_size = os.path.getsize(filepath)
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


def load_gt(data_dir, dataset, scenario_std):
    """Load GT for a dataset/scenario. Try text first, then binary."""
    gt_txt = data_dir / f"{dataset}/{dataset}_gt_{scenario_std}.txt"
    gt_bin = data_dir / f"{dataset}/{dataset}_gt_{scenario_std}.bin"

    if gt_txt.exists():
        return load_gt_txt(gt_txt, K)
    elif gt_bin.exists():
        return load_gt_bin(gt_bin, K)
    else:
        return None


# ==================== UNG ====================

def find_best_ung_config(dataset):
    """Find best UNG config per scenario from summary.csv.

    Returns: dict of scenario_ung -> (avg_recall_pct, max_degree, Lbuild, Lsearch)
    For validation datasets without summary.csv, detects configs from directory listing.
    """
    summary_path = BASE_DIR / f"UNG-dev/results_original/{dataset}/summary.csv"
    if summary_path.exists():
        best = {}
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'success':
                    continue
                scenario = row['scenario']
                recall = float(row['recall@10'])  # 0-100 scale
                config = (int(row['max_degree']), int(row['Lbuild']), int(row['Lsearch']))

                if scenario not in best or recall > best[scenario][0]:
                    best[scenario] = (recall, *config)
        return best

    # Fallback for validation datasets: no summary.csv
    # Detect available scenarios from directory listing
    ds_dir = BASE_DIR / f"UNG-dev/results_original/{dataset}"
    if not ds_dir.exists():
        return {}

    best = {}
    # Validation datasets use and/or/equal directory names
    for subdir in ds_dir.iterdir():
        if not subdir.is_dir():
            continue
        dirname = subdir.name
        # Map directory name to UNG scenario name
        if dirname in SCENARIO_MAP:
            scenario_ung = dirname  # containment/overlap/equality
        elif dirname in SCENARIO_MAP_REV:
            scenario_ung = dirname  # and/or/equal (use as-is, handle in extract)
        else:
            continue
        # Default config for validation datasets
        best[scenario_ung] = (0.0, 32, 100, 100)

    return best


def _find_ung_result_file(dataset, scenario_dir, max_degree, Lbuild, Lsearch, is_equality=False):
    """Find UNG result CSV file, handling both original and validation naming conventions."""
    base = BASE_DIR / f"UNG-dev/results_original/{dataset}/{scenario_dir}"

    # For equality, prefer _eqresult files (built with --scenario equality)
    eq_suffix = "_eq" if is_equality else ""

    # Try original naming: index_M=32_L=100_Ls=100result_L100.csv
    f = base / f"index_M={max_degree}_L={Lbuild}_Ls={Lsearch}{eq_suffix}result_L{Lsearch}.csv"
    if f.exists():
        return f

    # Try validation naming: M32_L100_Ls100result_L100.csv
    f = base / f"M{max_degree}_L{Lbuild}_Ls{Lsearch}{eq_suffix}result_L{Lsearch}.csv"
    if f.exists():
        return f

    # Try without eq suffix as fallback
    if is_equality:
        f = base / f"index_M={max_degree}_L={Lbuild}_Ls={Lsearch}result_L{Lsearch}.csv"
        if f.exists():
            return f
        f = base / f"M{max_degree}_L{Lbuild}_Ls{Lsearch}result_L{Lsearch}.csv"
        if f.exists():
            return f

    return None


def extract_ung_perquery(dataset, scenario_dir, max_degree, Lbuild, Lsearch, is_equality=False):
    """Extract per-query recall from UNG result CSV file.

    File format: CSV with header GT,Result
    Each row: space-separated GT IDs , space-separated Result IDs
    """
    result_file = _find_ung_result_file(
        dataset, scenario_dir, max_degree, Lbuild, Lsearch, is_equality)

    if result_file is None:
        print(f"  WARNING: UNG result file not found for {dataset}/{scenario_dir}")
        return None

    print(f"    file: {result_file.name}")
    gt_list = []
    res_list = []

    with open(result_file) as f:
        header = f.readline()  # Skip header "GT,Result"
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on comma - GT is before comma, Result is after
            parts = line.split(',')
            if len(parts) < 2:
                continue
            gt_ids = [int(x) for x in parts[0].strip().split() if x]
            res_ids = [int(x) for x in parts[1].strip().split() if x]
            gt_list.append(gt_ids)
            res_list.append(res_ids)

    return compute_perquery_recall(gt_list, res_list, K)


def process_ung(dataset, all_rows):
    """Process all UNG results for a dataset."""
    best = find_best_ung_config(dataset)

    for scenario_key, (recall_pct, M, Lb, Ls) in best.items():
        # Determine standard scenario name and directory name
        if scenario_key in SCENARIO_MAP:
            scenario_std = SCENARIO_MAP[scenario_key]
            scenario_dir = scenario_key
        else:
            scenario_std = scenario_key  # already standard (and/or/equal)
            scenario_dir = scenario_key

        is_equality = (scenario_std == 'equal')
        print(f"  UNG [{scenario_std}]: M={M}, Lb={Lb}, Ls={Ls}")

        recalls = extract_ung_perquery(dataset, scenario_dir, M, Lb, Ls, is_equality)
        if recalls:
            for qid, r in enumerate(recalls):
                all_rows.append({
                    'query_id': qid,
                    'dataset': dataset,
                    'scenario': scenario_std,
                    'method': 'UNG',
                    'recall_at_10': round(r, 6),
                    'params': 'M={}_Lb={}_Ls={}'.format(M, Lb, Ls),
                })
            avg_r = sum(recalls) / len(recalls)
            print(f"    -> {len(recalls)} queries, avg per-query recall={avg_r:.4f}")


# ==================== DiskANN (FilteredVamana / StitchedVamana) ====================

def load_diskann_result_bin(filepath):
    """Load DiskANN result from binary: (nq uint32, k uint32, then nq*k uint32 IDs)."""
    with open(filepath, 'rb') as f:
        nq = struct.unpack('<I', f.read(4))[0]
        k = struct.unpack('<I', f.read(4))[0]
        results = []
        for _ in range(nq):
            ids = list(struct.unpack(f'<{k}I', f.read(4 * k)))
            results.append(ids)
    return results, nq, k


def find_best_diskann_config(dataset, variant='original'):
    """Find best DiskANN config per scenario.

    variant: 'original' for FilteredVamana, 'stitched_original' for StitchedVamana

    Returns: dict of scenario_diskann -> (avg_recall, R, Lsearch[, stitched_R])
    """
    if variant == 'original':
        summary_path = BASE_DIR / f"DiskANN/data_original/results/{dataset}/summary.csv"
    else:
        summary_path = BASE_DIR / f"DiskANN/data_stitched_original/results/{dataset}/summary.csv"

    if not summary_path.exists():
        return {}

    best = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'success':
                continue
            scenario = row['scenario']
            recall = float(row['recall@10'])

            if variant == 'stitched_original':
                config = (int(row['R']), int(row['Lsearch']), int(row.get('stitched_R', 32)))
            else:
                config = (int(row['R']), int(row['Lsearch']))

            if scenario not in best or recall > best[scenario][0]:
                best[scenario] = (recall, *config)

    return best


def extract_diskann_perquery(dataset, scenario_diskann, R, Lsearch, data_dir,
                              variant='original', stitched_R=None):
    """Extract per-query recall from DiskANN binary result file."""
    scenario_std = SCENARIO_MAP[scenario_diskann]

    # Build result file path
    if variant == 'original':
        result_file = (BASE_DIR /
            f"DiskANN/data_original/results/{dataset}/"
            f"index_R={R}_{scenario_diskann}_Ls={Lsearch}_{Lsearch}_idx_uint32.bin")
    else:
        sR = stitched_R or 32
        result_file = (BASE_DIR /
            f"DiskANN/data_stitched_original/results/{dataset}/"
            f"stitched_R={R}_sR={sR}_{scenario_diskann}_Ls={Lsearch}_{Lsearch}_idx_uint32.bin")

    if not result_file.exists():
        print(f"  WARNING: result file not found: {result_file}")
        return None

    # Load GT
    gt = load_gt(data_dir, dataset, scenario_std)
    if gt is None:
        print(f"  WARNING: GT not found for {dataset}/{scenario_std}")
        return None

    # Load results
    results, nq, k = load_diskann_result_bin(result_file)

    # Align lengths
    n = min(len(gt), len(results))
    if n == 0:
        return None

    return compute_perquery_recall(gt[:n], results[:n], K)


def process_diskann(dataset, data_dir, all_rows, variant='original'):
    """Process FilteredVamana or StitchedVamana results."""
    method_name = 'FilteredVamana' if variant == 'original' else 'StitchedVamana'
    best = find_best_diskann_config(dataset, variant)

    for scenario_diskann, values in best.items():
        scenario_std = SCENARIO_MAP[scenario_diskann]

        if variant == 'stitched_original':
            recall, R, Ls, sR = values
            param_str = f'R={R}_sR={sR}_Ls={Ls}'
            print(f"  {method_name} [{scenario_std}]: best R={R}, sR={sR}, Ls={Ls} (avg recall={recall:.4f})")
        else:
            recall, R, Ls = values
            sR = None
            param_str = f'R={R}_Ls={Ls}'
            print(f"  {method_name} [{scenario_std}]: best R={R}, Ls={Ls} (avg recall={recall:.4f})")

        recalls = extract_diskann_perquery(
            dataset, scenario_diskann, R, Ls, data_dir,
            variant=variant, stitched_R=sR)

        if recalls:
            for qid, r in enumerate(recalls):
                all_rows.append({
                    'query_id': qid,
                    'dataset': dataset,
                    'scenario': scenario_std,
                    'method': method_name,
                    'recall_at_10': round(r, 6),
                    'params': param_str,
                })
            avg_r = sum(recalls) / len(recalls)
            print(f"    -> {len(recalls)} queries, avg per-query recall={avg_r:.4f}")


# ==================== Pre-filter ====================

def process_prefilter(dataset, data_dir, all_rows):
    """Process Pre-filter results (brute-force, no index params)."""
    faiss_dir = BASE_DIR / "faiss"

    for scenario_std in ['and', 'or', 'equal']:
        result_file = faiss_dir / "results_prefilter" / dataset / scenario_std / "prefilter_bruteforce_idx_uint32.bin"
        if not result_file.exists():
            continue

        gt = load_gt(data_dir, dataset, scenario_std)
        if gt is None:
            print(f"  WARNING: GT not found for {dataset}/{scenario_std}")
            continue

        results, nq, k = load_diskann_result_bin(result_file)
        n = min(len(gt), len(results))
        if n == 0:
            continue

        recalls = compute_perquery_recall(gt[:n], results[:n], K)
        if recalls:
            for qid, r in enumerate(recalls):
                all_rows.append({
                    'query_id': qid,
                    'dataset': dataset,
                    'scenario': scenario_std,
                    'method': 'Pre-filter',
                    'recall_at_10': round(r, 6),
                    'params': 'bruteforce',
                })
            avg_r = sum(recalls) / len(recalls)
            print(f"  Pre-filter [{scenario_std}]: {len(recalls)} queries, avg recall={avg_r:.4f}")


# ==================== Post-filter ====================

def find_best_postfilter_config(dataset):
    """Find best (M, efc) per scenario from summary.csv. Returns dict of scenario -> config."""
    summary_path = BASE_DIR / "faiss" / "results_postfilter" / dataset / "summary.csv"
    if summary_path.exists():
        best = {}
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'success':
                    continue
                scenario = row.get('scenario', '')
                recall = float(row.get('recall@10', 0))
                M = int(row.get('M', 0))
                efc = int(row.get('efc', 0))
                if scenario not in best or recall > best[scenario]['recall']:
                    best[scenario] = {'recall': recall, 'M': M, 'efc': efc}
        return best

    # Fallback for validation datasets: no summary.csv, default to M=32, efc=100
    ds_dir = BASE_DIR / "faiss" / "results_postfilter" / dataset
    if not ds_dir.exists():
        return {}

    best = {}
    for scenario_std in ['and', 'or', 'equal']:
        scenario_dir = ds_dir / scenario_std
        if scenario_dir.exists():
            best[scenario_std] = {'recall': 0.0, 'M': 32, 'efc': 100}
    return best


def process_postfilter(dataset, data_dir, all_rows):
    """Process Post-filter results. Finds best ef_search from binary result files."""
    import glob as globmod

    faiss_dir = BASE_DIR / "faiss"
    best_configs = find_best_postfilter_config(dataset)

    for scenario_std in ['and', 'or', 'equal']:
        if scenario_std not in best_configs:
            continue

        cfg = best_configs[scenario_std]
        M, efc = cfg['M'], cfg['efc']

        # Find all binary result files for this (M, efc)
        result_dir = faiss_dir / "results_postfilter" / dataset / scenario_std
        pattern = str(result_dir / f"M={M}_efc={efc}_ef=*_idx_uint32.bin")
        bin_files = sorted(globmod.glob(pattern))

        if not bin_files:
            continue

        gt = load_gt(data_dir, dataset, scenario_std)
        if gt is None:
            print(f"  WARNING: GT not found for {dataset}/{scenario_std}")
            continue

        # Find the ef_search with highest avg recall
        best_ef = None
        best_avg = -1
        best_recalls = None

        for bf in bin_files:
            results, nq, k = load_diskann_result_bin(bf)
            n = min(len(gt), len(results))
            if n == 0:
                continue
            recalls = compute_perquery_recall(gt[:n], results[:n], K)
            avg_r = sum(recalls) / len(recalls) if recalls else 0
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                # Extract ef from filename
                fname = Path(bf).name
                ef_str = fname.split('_ef=')[1].split('_')[0]
                best_ef = int(ef_str)

        if best_recalls and best_ef is not None:
            param_str = f'M={M}_efc={efc}_ef={best_ef}'
            for qid, r in enumerate(best_recalls):
                all_rows.append({
                    'query_id': qid,
                    'dataset': dataset,
                    'scenario': scenario_std,
                    'method': 'Post-filter',
                    'recall_at_10': round(r, 6),
                    'params': param_str,
                })
            print(f"  Post-filter [{scenario_std}]: M={M}, efc={efc}, ef={best_ef}, "
                  f"{len(best_recalls)} queries, avg recall={best_avg:.4f}")


# ==================== ACORN ====================

def find_best_acorn_config(dataset):
    """Find best (M, M_beta, gamma) per scenario from summary.csv files.

    Scans all param_search_* directories (including _gamma1).
    Returns: dict of scenario -> {recall, M, M_beta, gamma}
    """
    import glob as globmod

    best = {}
    acorn_dir = BASE_DIR / "ACORN"
    for summary in sorted(globmod.glob(
            str(acorn_dir / "data" / f"param_search_*" / "results" / "**" / "summary.csv"),
            recursive=True)):
        # Infer dataset from path
        parts = Path(summary).parts
        ds_in_path = None
        for p in parts:
            if p in DATASETS:
                ds_in_path = p
                break
        if ds_in_path != dataset:
            continue

        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'success':
                    continue
                scenario = row.get('scenario', '')
                recall = float(row.get('recall@10', row.get('recall', 0)))
                M = int(row.get('M', 0))
                M_beta = int(row.get('M_beta', 0))
                gamma = int(row.get('gamma', 0))
                if scenario not in best or recall > best[scenario]['recall']:
                    best[scenario] = {'recall': recall, 'M': M, 'M_beta': M_beta, 'gamma': gamma}
    return best


def process_acorn(dataset, data_dir, all_rows):
    """Process ACORN results. Finds best (M, M_beta, gamma), then best ef from binary files."""
    import glob as globmod

    acorn_dir = BASE_DIR / "ACORN"
    best_configs = find_best_acorn_config(dataset)

    for scenario_std in ['and', 'or', 'equal']:
        if scenario_std not in best_configs:
            continue

        cfg = best_configs[scenario_std]
        M, Mb, g = cfg['M'], cfg['M_beta'], cfg['gamma']

        # Find binary result files across all param_search dirs
        bin_files = []
        for d in sorted(globmod.glob(str(acorn_dir / "data" / f"param_search_{dataset}*"))):
            pattern = str(Path(d) / "results" / dataset / scenario_std /
                         f"M={M}_M_beta={Mb}_gamma={g}_ef=*_idx_uint32.bin")
            bin_files.extend(sorted(globmod.glob(pattern)))

        if not bin_files:
            continue

        gt = load_gt(data_dir, dataset, scenario_std)
        if gt is None:
            print(f"  WARNING: GT not found for {dataset}/{scenario_std}")
            continue

        # Find the ef with highest avg recall
        best_ef = None
        best_avg = -1
        best_recalls = None

        for bf in bin_files:
            results, nq, k = load_diskann_result_bin(bf)
            n = min(len(gt), len(results))
            if n == 0:
                continue
            recalls = compute_perquery_recall(gt[:n], results[:n], K)
            avg_r = sum(recalls) / len(recalls) if recalls else 0
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                # Extract ef from filename
                fname = Path(bf).name
                ef_str = fname.split('_ef=')[1].split('_')[0]
                best_ef = int(ef_str)

        if best_recalls and best_ef is not None:
            param_str = f'M={M}_Mb={Mb}_g={g}_ef={best_ef}'
            for qid, r in enumerate(best_recalls):
                all_rows.append({
                    'query_id': qid,
                    'dataset': dataset,
                    'scenario': scenario_std,
                    'method': 'ACORN',
                    'recall_at_10': round(r, 6),
                    'params': param_str,
                })
            print(f"  ACORN [{scenario_std}]: M={M}, Mb={Mb}, g={g}, ef={best_ef}, "
                  f"{len(best_recalls)} queries, avg recall={best_avg:.4f}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Extract per-query recall@10 from existing search result files')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='Datasets to process (default: original 6)')
    parser.add_argument('--include-validation', action='store_true',
                        help='Include validation datasets (synth200, arxiv_fanns_real, lid*)')
    parser.add_argument('--validation-only', action='store_true',
                        help='Process only validation datasets')
    parser.add_argument('--data-dir', default=None,
                        help='Path to dataset directory (default: ~/benchmarks/datasets/discrete)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path')
    args = parser.parse_args()

    if args.dataset:
        pass  # use explicitly provided datasets
    elif args.validation_only:
        args.dataset = VALIDATION_DATASETS
    elif args.include_validation:
        args.dataset = DATASETS + VALIDATION_DATASETS
    else:
        args.dataset = DATASETS

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(os.path.expanduser("~/benchmarks/datasets/discrete"))

    output_dir = BASE_DIR / "analysis" / "perquery_recall"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dataset in args.dataset:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # UNG
        process_ung(dataset, all_rows)

        # FilteredVamana
        process_diskann(dataset, data_dir, all_rows, variant='original')

        # StitchedVamana
        process_diskann(dataset, data_dir, all_rows, variant='stitched_original')

        # Pre-filter
        process_prefilter(dataset, data_dir, all_rows)

        # Post-filter
        process_postfilter(dataset, data_dir, all_rows)

        # ACORN
        process_acorn(dataset, data_dir, all_rows)

    # Write output
    output_file = args.output or str(output_dir / "perquery_recall.csv")
    print(f"\nWriting {len(all_rows)} rows to {output_file}")

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f,
            fieldnames=['query_id', 'dataset', 'scenario', 'method', 'recall_at_10', 'params'])
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print(f"\nSummary:")
    counts = Counter((r['dataset'], r['scenario'], r['method']) for r in all_rows)
    for (ds, sc, method), count in sorted(counts.items()):
        recalls = [r['recall_at_10'] for r in all_rows
                   if r['dataset'] == ds and r['scenario'] == sc and r['method'] == method]
        avg = sum(recalls) / len(recalls) if recalls else 0
        print(f"  {ds:12s} {sc:6s} {method:18s}: {count:6d} queries, avg recall={avg:.4f}")

    print(f"\nTotal: {len(all_rows)} per-query records")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    main()
