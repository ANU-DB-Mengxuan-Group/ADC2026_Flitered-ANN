#!/usr/bin/env python3
"""
Pre-filter brute-force baseline experiment script

Usage:
  cd ~/benchmarks/discrete/faiss
  python bash/auto_prefilter_bruteforce.py                    # Run all datasets
  python bash/auto_prefilter_bruteforce.py arxiv              # Run specific dataset
  python bash/auto_prefilter_bruteforce.py arxiv and          # Specific dataset and scenario
"""

import os
import sys
import time
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== Config ====================

FAISS_DIR = "/home/remote/u7905817/benchmarks/discrete/faiss"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"

# Dataset config
DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "tripclick": {"N": 1055976, "D": 768},
    "LAION1M": {"N": 1000448, "D": 512},
    "yfcc": {"N": 1000000, "D": 192},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
    # V2 datasets (合成 / V2 real)
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
}

SCENARIOS = ["and", "or", "equal"]
K = 10

# ==================== Helpers ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"completed": [], "failed": []}

def save_progress(progress_file, progress):
    Path(os.path.dirname(progress_file)).mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def init_summary_file(summary_file):
    if not os.path.exists(summary_file):
        Path(os.path.dirname(summary_file)).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'scenario', 'method',
                'qps', 'recall@10', 'avg_candidates',
                'filter_time', 'search_time', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, scenario, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, scenario, 'prefilter_bruteforce',
            metrics.get('qps', 0),
            metrics.get('recall', 0),
            metrics.get('avg_candidates', 0),
            metrics.get('filter_time', 0),
            metrics.get('search_time', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def parse_result(output_csv):
    """Parse result CSV"""
    try:
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {
                    'qps': float(row.get('QPS', 0)),
                    'recall': float(row.get('Recall', 0)),
                    'avg_candidates': float(row.get('AvgCandidates', 0)),
                    'filter_time': float(row.get('FilterTime', 0)),
                    'search_time': float(row.get('SearchTime', 0)),
                }
    except Exception as e:
        log(f"Warning: parse result failed: {e}")
    return None

# ==================== Main ====================

def run_search(dataset, scenario, config, output_base):
    """Run pre-filter brute-force search"""
    result_dir = f"{output_base}/results_prefilter/{dataset}/{scenario}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{result_dir}/prefilter_bruteforce.log"
    output_csv = f"{result_dir}/prefilter_bruteforce_result.csv"

    cmd = [
        f"{FAISS_DIR}/tutorial/cpp/search_prefilter_bruteforce_static",
        dataset,
        scenario,
        result_dir,
        f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs",
        f"{DATA_DIR}/{dataset}/label_base.txt",
        f"{DATA_DIR}/{dataset}/{dataset}_query_{scenario}.fvecs",
        f"{DATA_DIR}/{dataset}/{dataset}_query_{scenario}.txt",
        f"{DATA_DIR}/{dataset}/{dataset}_gt_{scenario}.txt",
        str(K)
    ]

    log(f"Running pre-filter [{scenario}]")

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=7200, check=True)

        metrics = parse_result(output_csv)
        if metrics and metrics['recall'] > 0:
            log(f"  Success: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}, AvgCandidates={metrics['avg_candidates']:.0f}")
            return True, metrics
        else:
            log(f"  Done but no valid results")
            return True, {'qps': 0, 'recall': 0, 'avg_candidates': 0}
    except subprocess.TimeoutExpired:
        log(f"  Timeout (2h)")
        return False, None
    except subprocess.CalledProcessError as e:
        log(f"  Failed: exit code {e.returncode}")
        return False, None
    except Exception as e:
        log(f"  Failed: {e}")
        return False, None

def run_dataset(dataset, scenarios_to_run, config, output_base):
    """Run all scenarios for a dataset"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()} (Pre-filter Brute-force)")
    log(f"N = {config['N']:,}, D = {config['D']}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_prefilter_{dataset}.json"
    summary_file = f"{output_base}/results_prefilter/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for scenario in scenarios_to_run:
        task_key = f"{dataset}_{scenario}"

        if task_key in progress.get("completed", []):
            log(f"Skip (done): {task_key}")
            continue

        if task_key in progress.get("failed", []):
            log(f"Skip (failed): {task_key}")
            continue

        search_ok, metrics = run_search(dataset, scenario, config, output_base)

        if not search_ok:
            progress.setdefault("failed", []).append(task_key)
            save_progress(progress_file, progress)
            append_to_summary(summary_file, dataset, scenario, {
                'qps': 0, 'recall': 0, 'avg_candidates': 0,
                'filter_time': 0, 'search_time': 0, 'status': 'failed'
            })
            continue

        progress.setdefault("completed", []).append(task_key)
        save_progress(progress_file, progress)

        metrics = metrics or {}
        metrics['status'] = 'success'
        append_to_summary(summary_file, dataset, scenario, metrics)

    log("")
    log(f"Dataset {dataset} done!")

# ==================== Entry ====================

def main():
    datasets = list(DATASETS_CONFIG.keys())
    scenarios = SCENARIOS

    if len(sys.argv) > 1:
        if sys.argv[1] in DATASETS_CONFIG:
            datasets = [sys.argv[1]]
        else:
            log(f"Invalid dataset: {sys.argv[1]}")
            log(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)

    if len(sys.argv) > 2:
        if sys.argv[2] in SCENARIOS:
            scenarios = [sys.argv[2]]
        else:
            log(f"Invalid scenario: {sys.argv[2]}")
            log(f"Available: {SCENARIOS}")
            sys.exit(1)

    log("=" * 60)
    log("Pre-filter Brute-force Baseline Experiment")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log(f"Scenarios: {', '.join(scenarios)}")
    log("")

    # Check executable
    exe = f"{FAISS_DIR}/tutorial/cpp/search_prefilter_bruteforce_static"
    if not os.path.exists(exe):
        log(f"Error: Executable not found: {exe}")
        log("Please compile first:")
        log("  cd ~/benchmarks/discrete/faiss/tutorial/cpp")
        log("  g++ -O3 -std=c++17 -fopenmp -I../../ -I../../build search_prefilter_bruteforce.cpp \\")
        log("      ../../build/faiss/libfaiss.a -L$CONDA_PREFIX/lib \\")
        log("      -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl \\")
        log("      -Wl,--no-as-needed -o search_prefilter_bruteforce_static")
        sys.exit(1)

    start_time = time.time()

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        try:
            run_dataset(dataset, scenarios, config, FAISS_DIR)
        except KeyboardInterrupt:
            log("User interrupted")
            sys.exit(0)
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    log("")
    log("=" * 60)
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log("=" * 60)

if __name__ == "__main__":
    main()
