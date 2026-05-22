#!/usr/bin/env python3
"""
NHQ 完整参数搜索 - arxiv 和 yfcc
记录 build_time, index_size, QPS, Recall
"""

import os
import sys
import time
import csv
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

# 基于NHQ论文的参数范围
PARAM_SETS = [
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 100, "RANGE": 20, "PL": 150, "B": 0.4, "M": 1},
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 200, "RANGE": 20, "PL": 250, "B": 0.4, "M": 1},
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 300, "RANGE": 20, "PL": 350, "B": 0.4, "M": 1},
    {"K": 200, "L": 200, "iter": 12, "S": 15, "R": 200, "RANGE": 40, "PL": 150, "B": 0.6, "M": 1},
    {"K": 200, "L": 200, "iter": 12, "S": 15, "R": 300, "RANGE": 40, "PL": 250, "B": 0.6, "M": 1},
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_param_str(params):
    return f"K={params['K']}_L={params['L']}_iter={params['iter']}_S={params['S']}_R={params['R']}_RANGE={params['RANGE']}_PL={params['PL']}_B={params['B']}_M={params['M']}"

def get_index_size(index_dir):
    """获取索引目录大小（MB）"""
    total_size = 0
    try:
        for f in ['graph.bin', 'attr.txt']:
            fp = os.path.join(index_dir, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    except:
        return 0.0

def build_index(dataset, params, output_base):
    param_str = get_param_str(params)
    index_dir = f"{output_base}/indices/{dataset}/{param_str}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    # 如果已经构建过，跳过但返回索引大小
    if os.path.exists(graph_path) and os.path.getsize(graph_path) > 0:
        index_size = get_index_size(index_dir)
        log(f"  Index exists ({index_size:.1f}MB), skipping build")
        return True, 0, index_size, index_dir
    
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
    
    log(f"Building {param_str}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200, check=True)
        build_time = time.time() - start
        index_size = get_index_size(index_dir)
        log(f"  Build completed: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size, index_dir
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, 0, 0, None

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
    
    log(f"Searching {param_str}...")
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800, check=True)
        log(f"  Done -> {result_csv}")
        return True, result_csv
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

def write_summary(dataset, results, output_base):
    """写入汇总CSV"""
    summary_file = f"{output_base}/results/{dataset}/summary.csv"
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['params', 'build_time_s', 'index_size_mb', 'L', 'QPS', 'Recall@10'])
        
        for r in results:
            if r.get('search_csv') and os.path.exists(r['search_csv']):
                # 读取搜索结果
                with open(r['search_csv'], 'r') as sf:
                    reader = csv.DictReader(sf)
                    for row in reader:
                        writer.writerow([
                            r['params'],
                            f"{r['build_time']:.2f}",
                            f"{r['index_size']:.2f}",
                            row['L'],
                            row['QPS'],
                            row['Recall']
                        ])
    
    log(f"Summary written to {summary_file}")

def run_dataset(dataset):
    log(f"\n{'='*60}")
    log(f"Dataset: {dataset}")
    log(f"{'='*60}")
    
    results = []
    
    for i, params in enumerate(PARAM_SETS):
        log(f"\nParam set {i+1}/{len(PARAM_SETS)}")
        param_str = get_param_str(params)
        
        success, build_time, index_size, index_dir = build_index(dataset, params, OUTPUT_BASE)
        
        result = {
            'params': param_str,
            'build_time': build_time,
            'index_size': index_size,
            'search_csv': None
        }
        
        if success:
            search_ok, search_csv = search_index(dataset, params, index_dir, OUTPUT_BASE)
            if search_ok:
                result['search_csv'] = search_csv
        
        results.append(result)
    
    # 写入汇总
    write_summary(dataset, results, OUTPUT_BASE)

def main():
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ['arxiv', 'yfcc']
    
    for dataset in datasets:
        if dataset in DATASETS:
            run_dataset(dataset)
        else:
            log(f"Unknown dataset: {dataset}")
    
    log("\n=== All done ===")

if __name__ == '__main__':
    main()
