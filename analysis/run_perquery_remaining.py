#!/usr/bin/env python3
"""
Run Post-filter, Pre-filter, and ACORN with best configs to generate per-query result files.

Rebuilds indexes as needed (they were cleaned up after original experiments).
Only runs the best (M, efc) or (M, M_beta, gamma) per dataset — not all combos.

Usage:
    cd ~/benchmarks/discrete
    python analysis/run_perquery_remaining.py --method postfilter
    python analysis/run_perquery_remaining.py --method prefilter
    python analysis/run_perquery_remaining.py --method acorn
    python analysis/run_perquery_remaining.py --method all
    python analysis/run_perquery_remaining.py --method postfilter --dataset arxiv
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('DATA_DIR',
    os.path.expanduser("~/benchmarks/datasets/discrete")))

FAISS_DIR = BASE_DIR / "faiss"
ACORN_DIR = BASE_DIR / "ACORN"

# Use the cmake-built binaries (not _static)
POSTFILTER_SEARCH_BIN = FAISS_DIR / "build" / "tutorial" / "cpp" / "search_HNSW_index"
POSTFILTER_BUILD_BIN = FAISS_DIR / "build" / "tutorial" / "cpp" / "build_HNSW_index"
PREFILTER_SEARCH_BIN = FAISS_DIR / "build" / "tutorial" / "cpp" / "search_prefilter_bruteforce"
ACORN_SEARCH_BIN = ACORN_DIR / "build" / "demos" / "search_acorn_index"
ACORN_BUILD_BIN = ACORN_DIR / "build" / "demos" / "build_acorn_index"

DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "tripclick": {"N": 1055976, "D": 768},
    "LAION1M": {"N": 1000448, "D": 512},
    "yfcc": {"N": 1000000, "D": 192},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
}

SCENARIOS = ["and", "or", "equal"]
K = 10


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_best_postfilter_configs():
    """Find best (M, efc) per (dataset, scenario) from existing summary.csv."""
    best = {}
    for dataset in DATASETS_CONFIG:
        summary = FAISS_DIR / "results_postfilter" / dataset / "summary.csv"
        if not summary.exists():
            continue
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario = row.get('scenario', '')
                status = row.get('status', 'success')
                if status != 'success':
                    continue
                recall = float(row.get('recall@10', row.get('recall', row.get('Recall', 0))))
                key = (dataset, scenario)
                if key not in best or recall > best[key]['recall']:
                    best[key] = {
                        'recall': recall,
                        'M': int(row.get('M', 0)),
                        'efc': int(row.get('efc', row.get('efConstruction', 0))),
                    }
    return best


def find_best_acorn_configs():
    """Find best (M, M_beta, gamma) per (dataset, scenario) from existing summary.csv."""
    best = {}
    import glob as globmod
    for summary in sorted(globmod.glob(str(ACORN_DIR / "data" / "param_search_*" / "**" / "summary.csv"), recursive=True)):
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get('dataset', '')
                # Infer dataset from path if not in CSV
                if not dataset:
                    parts = Path(summary).parts
                    for p in parts:
                        if p in DATASETS_CONFIG:
                            dataset = p
                            break
                if dataset not in DATASETS_CONFIG:
                    continue
                scenario = row.get('scenario', '')
                status = row.get('status', 'success')
                if status != 'success':
                    continue
                recall = float(row.get('recall@10', row.get('recall', row.get('Recall', 0))))
                key = (dataset, scenario)
                if key not in best or recall > best[key]['recall']:
                    best[key] = {
                        'recall': recall,
                        'M': int(row.get('M', 0)),
                        'M_beta': int(row.get('M_beta', 0)),
                        'gamma': int(row.get('gamma', 0)),
                    }
    return best


# ==================== Post-filter ====================

def run_postfilter(datasets, scenarios):
    best = find_best_postfilter_configs()
    log(f"Post-filter: found {len(best)} best configs")
    for (ds, sc), cfg in sorted(best.items()):
        log(f"  {ds}/{sc}: M={cfg['M']}, efc={cfg['efc']}, recall={cfg['recall']:.4f}")

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        N = config['N']

        # Collect unique (M, efc) combos needed for this dataset
        needed_indices = set()
        for scenario in scenarios:
            key = (dataset, scenario)
            if key in best:
                b = best[key]
                needed_indices.add((b['M'], b['efc']))

        for M, efc in sorted(needed_indices):
            index_dir = FAISS_DIR / "data" / "index_files" / "hnsw"
            index_path = index_dir / dataset / f"M={M}_efc={efc}.json"

            # Build index if needed
            if not index_path.exists():
                log(f"Building Post-filter index: {dataset} M={M} efc={efc}")
                base_file = DATA_DIR / dataset / f"{dataset}_base.fvecs"
                cmd = [
                    str(POSTFILTER_BUILD_BIN),
                    str(base_file), str(M), str(efc),
                    str(index_dir), dataset
                ]
                start = time.time()
                result = subprocess.run(cmd, capture_output=True, timeout=7200)
                elapsed = time.time() - start
                if index_path.exists():
                    log(f"  Built in {elapsed:.1f}s")
                else:
                    log(f"  BUILD FAILED! stderr: {result.stderr.decode()[:200]}")
                    continue

            # Search each scenario
            for scenario in scenarios:
                key = (dataset, scenario)
                if key not in best:
                    continue
                b = best[key]
                if b['M'] != M or b['efc'] != efc:
                    continue

                result_dir = FAISS_DIR / "results_postfilter" / dataset / scenario
                result_dir.mkdir(parents=True, exist_ok=True)

                log(f"Searching Post-filter: {dataset}/{scenario} M={M} efc={efc}")
                cmd = [
                    str(POSTFILTER_SEARCH_BIN),
                    dataset, str(M), str(efc),
                    str(index_dir), scenario, str(result_dir),
                    str(DATA_DIR / dataset / f"{dataset}_base.fvecs"),
                    str(DATA_DIR / dataset / "label_base.txt"),
                    str(DATA_DIR / dataset / f"{dataset}_query_{scenario}.fvecs"),
                    str(DATA_DIR / dataset / f"{dataset}_query_{scenario}.txt"),
                    str(DATA_DIR / dataset / f"{dataset}_gt_{scenario}.txt"),
                    str(K), str(N)
                ]

                log_file = result_dir / f"M={M}_efc={efc}_perquery.log"
                start = time.time()
                try:
                    with open(log_file, 'w') as f:
                        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=3600, check=True)
                    elapsed = time.time() - start
                    log(f"  Done in {elapsed:.1f}s")
                except Exception as e:
                    log(f"  FAILED: {e}")

            # Cleanup index
            if index_path.exists():
                os.remove(index_path)
                log(f"  Cleaned index: {index_path.name}")


# ==================== Pre-filter ====================

def run_prefilter(datasets, scenarios):
    for dataset in datasets:
        for scenario in scenarios:
            result_dir = FAISS_DIR / "results_prefilter" / dataset / scenario
            result_dir.mkdir(parents=True, exist_ok=True)

            log(f"Searching Pre-filter: {dataset}/{scenario}")
            cmd = [
                str(PREFILTER_SEARCH_BIN),
                dataset, scenario, str(result_dir),
                str(DATA_DIR / dataset / f"{dataset}_base.fvecs"),
                str(DATA_DIR / dataset / "label_base.txt"),
                str(DATA_DIR / dataset / f"{dataset}_query_{scenario}.fvecs"),
                str(DATA_DIR / dataset / f"{dataset}_query_{scenario}.txt"),
                str(DATA_DIR / dataset / f"{dataset}_gt_{scenario}.txt"),
                str(K)
            ]

            log_file = result_dir / "prefilter_bruteforce_perquery.log"
            start = time.time()
            try:
                with open(log_file, 'w') as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=7200, check=True)
                elapsed = time.time() - start
                log(f"  Done in {elapsed:.1f}s")
            except Exception as e:
                log(f"  FAILED: {e}")


# ==================== ACORN ====================

def run_acorn(datasets, scenarios):
    best = find_best_acorn_configs()
    log(f"ACORN: found {len(best)} best configs")
    for (ds, sc), cfg in sorted(best.items()):
        log(f"  {ds}/{sc}: M={cfg['M']}, Mb={cfg['M_beta']}, g={cfg['gamma']}, recall={cfg['recall']:.4f}")

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        N = config['N']
        data_dir = DATA_DIR / dataset

        # Collect unique (M, M_beta, gamma) combos needed
        needed_indices = set()
        for scenario in scenarios:
            key = (dataset, scenario)
            if key in best:
                b = best[key]
                needed_indices.add((b['M'], b['M_beta'], b['gamma']))

        for M, M_beta, gamma in sorted(needed_indices):
            acorn_base = ACORN_DIR / "data" / f"param_search_{dataset}"
            # Build/search binaries append dataset/ internally, so pass parent
            index_parent = acorn_base / "indices"
            index_dir = index_parent / dataset
            index_path = index_dir / f"hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

            # Build index if needed
            if not index_path.exists():
                log(f"Building ACORN index: {dataset} M={M} M_beta={M_beta} gamma={gamma}")
                index_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    str(ACORN_BUILD_BIN),
                    str(N), str(gamma),
                    str(data_dir / f"{dataset}_base.fvecs"),
                    str(M), str(M_beta), str(index_parent), dataset
                ]
                env = os.environ.copy()
                env['debugSearchFlag'] = '0'

                build_log = index_dir / f"M={M}_Mb={M_beta}_gamma={gamma}_build.log"
                start = time.time()
                try:
                    with open(build_log, 'w') as f:
                        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=7200)
                    elapsed = time.time() - start
                    if index_path.exists():
                        log(f"  Built in {elapsed:.1f}s")
                    else:
                        log(f"  BUILD FAILED (no index file)")
                        continue
                except Exception as e:
                    log(f"  BUILD FAILED: {e}")
                    continue

            # Search each scenario that uses this index
            for scenario in scenarios:
                key = (dataset, scenario)
                if key not in best:
                    continue
                b = best[key]
                if b['M'] != M or b['M_beta'] != M_beta or b['gamma'] != gamma:
                    continue

                results_dir = acorn_base / "results" / dataset / scenario
                results_dir.mkdir(parents=True, exist_ok=True)

                log(f"Searching ACORN: {dataset}/{scenario} M={M} Mb={M_beta} g={gamma}")
                cmd = [
                    str(ACORN_SEARCH_BIN),
                    str(N), str(gamma), dataset,
                    str(M), str(M_beta),
                    str(index_parent), scenario, str(results_dir),
                    str(data_dir / f"{dataset}_base.fvecs"),
                    str(data_dir / "label_base.txt"),
                    str(data_dir / f"{dataset}_query_{scenario}.fvecs"),
                    str(data_dir / f"{dataset}_query_{scenario}.txt"),
                    str(data_dir / f"{dataset}_gt_{scenario}.txt"),
                    str(K)
                ]
                env = os.environ.copy()
                env['debugSearchFlag'] = '0'

                log_file = results_dir / f"M={M}_Mb={M_beta}_gamma={gamma}_perquery.log"
                start = time.time()
                try:
                    with open(log_file, 'w') as f:
                        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                      timeout=3600, check=True)
                    elapsed = time.time() - start
                    log(f"  Done in {elapsed:.1f}s")
                except Exception as e:
                    log(f"  FAILED: {e}")

            # Cleanup index (build creates both gamma and gamma=1 files)
            gamma1_path = index_dir / f"hybrid_M={M}_Mb={M_beta}_gamma=1.json"
            for p in [index_path, gamma1_path]:
                if p.exists():
                    os.remove(p)
                    log(f"  Cleaned index: {p.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True,
                       choices=['postfilter', 'prefilter', 'acorn', 'all'])
    parser.add_argument('--dataset', nargs='+', default=list(DATASETS_CONFIG.keys()))
    parser.add_argument('--scenario', nargs='+', default=SCENARIOS)
    args = parser.parse_args()

    log(f"Methods: {args.method}, Datasets: {args.dataset}, Scenarios: {args.scenario}")

    if args.method in ('prefilter', 'all'):
        log("\n" + "=" * 60)
        log("PRE-FILTER (brute-force)")
        log("=" * 60)
        run_prefilter(args.dataset, args.scenario)

    if args.method in ('postfilter', 'all'):
        log("\n" + "=" * 60)
        log("POST-FILTER (HNSW)")
        log("=" * 60)
        run_postfilter(args.dataset, args.scenario)

    if args.method in ('acorn', 'all'):
        log("\n" + "=" * 60)
        log("ACORN")
        log("=" * 60)
        run_acorn(args.dataset, args.scenario)

    log("\nAll done!")


if __name__ == '__main__':
    main()
