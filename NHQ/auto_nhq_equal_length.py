#!/usr/bin/env python3
"""
NHQ 自动参数搜索 - 使用等长格式数据
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================

NHQ_DIR = "/home/remote/u7905817/benchmarks/discrete/NHQ/NHQ/NHQ-NPG_kgraph"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"
EQUAL_LEN_DIR = "/home/remote/u7905817/benchmarks/discrete/equal_length_experiment"
OUTPUT_BASE = "/home/remote/u7905817/benchmarks/discrete/NHQ/data_equal_length"

DATASETS = {
    "arxiv": {"N": 132687},
    "LAION1M": {"N": 1000448},
    "tripclick": {"N": 1055976},
    # "yfcc": {"N": 1000000},  # max_labels=1332 可能太大
    # "ytb_audio": {"N": 5000000},
    # "ytb_video": {"N": 1000000},
}

# NHQ 参数 - 先用一组简单参数测试
PARAM_SETS = [
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 100, "RANGE": 20, "PL": 150, "B": 0.4, "M": 1},
]

# ==================== 辅助函数 ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_param_str(params):
    return f"K={params['K']}_L={params['L']}_iter={params['iter']}_S={params['S']}_R={params['R']}_RANGE={params['RANGE']}_PL={params['PL']}_B={params['B']}_M={params['M']}"

def build_index(dataset, params, output_base):
    """构建 NHQ 索引"""
    param_str = get_param_str(params)
    index_dir = f"{output_base}/indices/{dataset}/{param_str}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    label_path = f"{EQUAL_LEN_DIR}/{dataset}/label_NHQ_base.txt"
    
    cmd = [
        f"{NHQ_DIR}/tests/test_dng_index",
        data_path,
        label_path,
        graph_path,
        attr_path,
        str(params['K']),
        str(params['L']),
        str(params['iter']),
        str(params['S']),
        str(params['R']),
        str(params['RANGE']),
        str(params['PL']),
        str(params['B']),
        str(params['M']),
    ]
    
    log_file = f"{index_dir}/build.log"
    start = time.time()
    
    log(f"Building index for {dataset} with {param_str}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, 
                          timeout=7200, check=True)
        build_time = time.time() - start
        log(f"  Build completed in {build_time:.1f}s")
        return True, build_time, index_dir
    except subprocess.TimeoutExpired:
        log(f"  Build timeout")
        return False, 0, None
    except subprocess.CalledProcessError as e:
        log(f"  Build failed: {e}")
        return False, 0, None

def search_index(dataset, params, index_dir, output_base):
    """搜索测试"""
    param_str = get_param_str(params)
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    query_path = f"{DATA_DIR}/{dataset}/{dataset}_query_equal.fvecs"
    query_label = f"{EQUAL_LEN_DIR}/{dataset}/label_NHQ_query.txt"
    gt_path = f"{DATA_DIR}/{dataset}/{dataset}_gt_equal.txt"
    
    result_csv = f"{result_dir}/{param_str}_results.csv"
    
    cmd = [
        f"{NHQ_DIR}/tests/test_dng_optimized_search",
        graph_path,
        attr_path,
        data_path,
        query_path,
        query_label,
        gt_path,
        result_csv,
    ]
    
    log_file = f"{result_dir}/{param_str}_search.log"
    
    log(f"Searching {dataset} with {param_str}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=1800, check=True)
        log(f"  Search completed, results in {result_csv}")
        return True
    except Exception as e:
        log(f"  Search failed: {e}")
        return False

def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "arxiv"
    
    log(f"=== NHQ Equal Length Test: {dataset} ===")
    
    for params in PARAM_SETS:
        success, build_time, index_dir = build_index(dataset, params, OUTPUT_BASE)
        if success:
            search_index(dataset, params, index_dir, OUTPUT_BASE)

if __name__ == '__main__':
    main()
