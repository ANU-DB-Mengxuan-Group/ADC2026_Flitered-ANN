#!/usr/bin/env python3
"""
NHQ 测试 - 使用合成固定长度标签数据
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

NHQ_DIR = "/home/remote/u7905817/benchmarks/discrete/NHQ/NHQ/NHQ-NPG_kgraph"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"
LABEL_DIR = "/home/remote/u7905817/benchmarks/discrete/NHQ/synthetic_labels"
OUTPUT_BASE = "/home/remote/u7905817/benchmarks/discrete/NHQ/data_synthetic"

DATASETS = {
    "arxiv": {"N": 132687, "query_file": "arxiv_query.fvecs"},
    "yfcc": {"N": 1000000, "query_file": "yfcc_query_equal.fvecs"},
}

PARAM_SETS = [
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 100, "RANGE": 20, "PL": 150, "B": 0.4, "M": 1},
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_param_str(params):
    return f"K={params['K']}_L={params['L']}_iter={params['iter']}_S={params['S']}_R={params['R']}_RANGE={params['RANGE']}_PL={params['PL']}_B={params['B']}_M={params['M']}"

def build_index(dataset, params, output_base):
    param_str = get_param_str(params)
    index_dir = f"{output_base}/indices/{dataset}/{param_str}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    label_path = f"{LABEL_DIR}/{dataset}/label_NHQ_base.txt"
    
    cmd = [
        f"{NHQ_DIR}/tests/test_dng_index",
        data_path, label_path, graph_path, attr_path,
        str(params['K']), str(params['L']), str(params['iter']),
        str(params['S']), str(params['R']), str(params['RANGE']),
        str(params['PL']), str(params['B']), str(params['M']),
    ]
    
    log_file = f"{index_dir}/build.log"
    start = time.time()
    
    log(f"Building index for {dataset}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200, check=True)
        build_time = time.time() - start
        log(f"  Build completed in {build_time:.1f}s")
        return True, index_dir
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, None

def search_index(dataset, params, index_dir, output_base):
    param_str = get_param_str(params)
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    query_path = f"{DATA_DIR}/{dataset}/{DATASETS[dataset]['query_file']}"
    query_label = f"{LABEL_DIR}/{dataset}/label_NHQ_query.txt"
    gt_path = f"{LABEL_DIR}/{dataset}/gt_NHQ.txt"
    result_csv = f"{result_dir}/{param_str}_results.csv"
    
    cmd = [
        f"{NHQ_DIR}/tests/test_dng_optimized_search",
        graph_path, attr_path, data_path, query_path, query_label, gt_path, result_csv,
    ]
    
    log_file = f"{result_dir}/{param_str}_search.log"
    
    log(f"Searching {dataset}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800, check=True)
        log(f"  Search completed, results in {result_csv}")
        
        # 打印结果
        if os.path.exists(result_csv):
            with open(result_csv, 'r') as f:
                print(f.read())
        return True
    except Exception as e:
        log(f"  Search failed: {e}")
        # 打印日志
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                print("Log:", f.read()[-2000:])
        return False

def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "arxiv"
    
    log(f"=== NHQ Synthetic Test: {dataset} ===")
    
    for params in PARAM_SETS:
        success, index_dir = build_index(dataset, params, OUTPUT_BASE)
        if success:
            search_index(dataset, params, index_dir, OUTPUT_BASE)

if __name__ == '__main__':
    main()
