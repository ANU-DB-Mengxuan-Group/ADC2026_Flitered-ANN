#!/usr/bin/env python3
"""V2 数据集 UNG 实验 - 固定配置版

固定参数（来自论文 5.1 节）:
- max_degree = 96
- Lbuild = 200
- Lsearch = 500
"""
import os
import sys
import subprocess
import time

# 固定配置
M = 96
Lb = 200
Ls = 500
K = 10
T = 16

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k", "synth_10k", "synth_4k", "synth_100k"]
SCENARIOS = ["containment", "overlap", "equality"]

DATA_DIR = os.path.expanduser("~/benchmarks/datasets/discrete")
UNG_DIR = os.path.expanduser("~/benchmarks/discrete/UNG-dev")
BUILD_BIN = f"{UNG_DIR}/build/apps/build_UNG_index"
SEARCH_BIN = f"{UNG_DIR}/build/apps/search_UNG_index"

DATASET_CONFIG = {
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
    "synth_10k": {"N": 800000, "D": 192},
    "synth_4k": {"N": 800000, "D": 768},
    "synth_100k": {"N": 800000, "D": 192},
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def get_label_file(path):
    """V2 用 1-based 标签"""
    base, ext = os.path.splitext(path)
    onebased = f"{base}_1based{ext}"
    if os.path.exists(onebased):
        return onebased
    return path

def run_cmd(cmd, timeout=3600):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def build_index(dataset, scenario):
    """Build UNG index"""
    # File suffix mapping: containment->and, overlap->or, equality->equal
    suffix_map = {"containment": "and", "overlap": "or", "equality": "equal"}
    suffix = suffix_map[scenario]

    # UNG scenario mapping
    if scenario == "containment":
        build_scenario = "general"  # containment/overlap use general
    elif scenario == "overlap":
        build_scenario = "general"
    else:
        build_scenario = "equality"

    index_dir = f"{UNG_DIR}/indices_v2_fixed/{dataset}/{build_scenario}"
    os.makedirs(index_dir, exist_ok=True)
    index_prefix = f"{index_dir}/index_M={M}_L={Lb}"

    # Check if already built
    if os.path.exists(f"{index_prefix}.data"):
        log(f"  Index exists, skipping build")
        return index_prefix

    base_label = get_label_file(f"{DATA_DIR}/{dataset}/label_base.txt")

    cmd = [
        BUILD_BIN,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--base_label_file", base_label,
        "--index_path_prefix", index_prefix,
        "--scenario", build_scenario,
        "--max_degree", str(M),
        "--Lbuild", str(Lb),
        "--num_cross_edges", "6",
        "--num_threads", str(T),
    ]

    log(f"  Building: M={M}, Lb={Lb}, scenario={build_scenario}")
    start = time.time()
    success, output = run_cmd(cmd)
    elapsed = time.time() - start

    if success:
        log(f"  Build success: {elapsed:.1f}s")
        return index_prefix
    else:
        log(f"  Build FAILED: {output[:200]}")
        return None

def search_index(dataset, scenario, index_prefix):
    """Search UNG index"""
    # File suffix mapping: containment->and, overlap->or, equality->equal
    suffix_map = {"containment": "and", "overlap": "or", "equality": "equal"}
    suffix = suffix_map[scenario]

    # Search scenario mapping
    search_scenario = "containment" if scenario == "containment" else scenario

    result_dir = f"{UNG_DIR}/results_v2_fixed"
    os.makedirs(result_dir, exist_ok=True)
    result_file = f"{result_dir}/{dataset}_{scenario}_M={M}_Lb={Lb}_Ls={Ls}.txt"

    # Check if already done
    if os.path.exists(result_file):
        with open(result_file) as f:
            content = f.read()
        if "Recall=" in content:
            log(f"  Already done, skipping")
            return

    base_label = get_label_file(f"{DATA_DIR}/{dataset}/label_base.txt")
    query_label = get_label_file(f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.txt")
    gt_file = f"{DATA_DIR}/{dataset}/{dataset}_gt_{suffix}.bin"

    result_prefix = f"{result_dir}/{dataset}_{scenario}_M={M}_Lb={Lb}_Ls={Ls}"

    cmd = [
        SEARCH_BIN,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--query_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.bin",
        "--base_label_file", base_label,
        "--query_label_file", query_label,
        "--gt_file", gt_file,
        "--K", str(K),
        "--index_path_prefix", index_prefix,
        "--scenario", search_scenario,
        "--Lsearch", str(Ls),
        "--num_threads", str(T),
        "--result_path_prefix", result_prefix,
    ]

    log(f"  Searching: Ls={Ls}")
    start = time.time()
    success, output = run_cmd(cmd)
    elapsed = time.time() - start

    # Save output
    with open(result_file, 'w') as f:
        f.write(output)

    if success:
        # Extract recall
        for line in output.split('\n'):
            if 'Recall' in line:
                log(f"  {line.strip()}")
                break
    else:
        log(f"  Search FAILED: {output[:200]}")

def main():
    log("="*60)
    log("V2 UNG Fixed Config: M=96, Lb=200, Ls=500")
    log("="*60)

    datasets = sys.argv[1:] if len(sys.argv) > 1 else V2_DATASETS

    for dataset in datasets:
        if dataset not in DATASET_CONFIG:
            log(f"Unknown dataset: {dataset}")
            continue

        for scenario in SCENARIOS:
            log(f"\n>>> {dataset} / {scenario}")

            # Build
            index_prefix = build_index(dataset, scenario)
            if not index_prefix:
                continue

            # Search
            search_index(dataset, scenario, index_prefix)

    log("\n" + "="*60)
    log("Done!")
    log("="*60)

if __name__ == "__main__":
    main()
