#!/usr/bin/env python3
"""
NHQ 自动参数搜索 - 所有数据集
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
OUTPUT_BASE = "/home/remote/u7905817/benchmarks/discrete/NHQ/data"

DATASETS = {
    "arxiv": {"N": 132687},
    "yfcc": {"N": 1000000},
    "LAION1M": {"N": 1000448},
    "tripclick": {"N": 1055976},
    "ytb_audio": {"N": 5000000},
    "ytb_video": {"N": 1000000},
    # --- V2 validation datasets ---
    "synth_192d": {"N": 800000},
    "synth_512d": {"N": 800000},
    "synth_768d_hc": {"N": 800000},
    "yahoo800k": {"N": 800000},
    "dbpedia560k": {"N": 560000},
}

# NHQ 参数 - 基于论文推荐值
PARAM_SETS = [
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 100, "RANGE": 20, "PL": 150, "B": 0.4, "M": 1},
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 200, "RANGE": 20, "PL": 250, "B": 0.4, "M": 1},
    {"K": 100, "L": 100, "iter": 12, "S": 10, "R": 300, "RANGE": 20, "PL": 350, "B": 0.4, "M": 1},
    {"K": 200, "L": 200, "iter": 12, "S": 15, "R": 200, "RANGE": 40, "PL": 150, "B": 0.6, "M": 1},
    {"K": 200, "L": 200, "iter": 12, "S": 15, "R": 300, "RANGE": 40, "PL": 250, "B": 0.6, "M": 1},
]

# ==================== 辅助函数 ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_param_str(params):
    return f"K={params['K']}_L={params['L']}_iter={params['iter']}_S={params['S']}_R={params['R']}_RANGE={params['RANGE']}_PL={params['PL']}_B={params['B']}_M={params['M']}"

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress_file, progress):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def build_index(dataset, params, output_base):
    """构建 NHQ 索引"""
    param_str = get_param_str(params)
    index_dir = f"{output_base}/indices/{dataset}/{param_str}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    label_path = f"{DATA_DIR}/{dataset}/label_NHQ_base.txt"
    
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
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, 
                          timeout=7200, check=True)  # 2小时超时
        build_time = time.time() - start
        
        # 计算索引大小
        index_size = 0
        if os.path.exists(graph_path):
            index_size += os.path.getsize(graph_path)
        if os.path.exists(attr_path):
            index_size += os.path.getsize(attr_path)
        index_size_mb = index_size / (1024 * 1024)
        
        return True, build_time, index_size_mb, index_dir
    except subprocess.TimeoutExpired:
        log(f"Build timeout")
        return False, 0, 0, None
    except subprocess.CalledProcessError as e:
        log(f"Build failed: {e}")
        return False, 0, 0, None

def search_index(dataset, params, index_dir, output_base):
    """搜索测试"""
    param_str = get_param_str(params)
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = f"{index_dir}/graph.bin"
    attr_path = f"{index_dir}/attr.txt"
    
    data_path = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"
    query_path = f"{DATA_DIR}/{dataset}/{dataset}_query_equal.fvecs"
    query_label = f"{DATA_DIR}/{dataset}/label_NHQ_query.txt"
    gt_path = f"{DATA_DIR}/{dataset}/{dataset}_gt_equal.txt"
    
    # 检查 query label 文件
    if not os.path.exists(query_label):
        # 尝试其他格式
        query_label = f"{DATA_DIR}/{dataset}/{dataset}_query_equal.txt"
    
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
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=1800, check=True)  # 30分钟超时
        
        # 解析结果
        if os.path.exists(result_csv):
            with open(result_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # 取最佳 recall 的结果
                    best_recall = 0
                    best_qps = 0
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            recall = float(parts[1])
                            qps = float(parts[2])
                            if recall > best_recall:
                                best_recall = recall
                                best_qps = qps
                    return True, best_recall, best_qps
        return False, 0, 0
    except subprocess.TimeoutExpired:
        log(f"Search timeout")
        return False, 0, 0
    except subprocess.CalledProcessError as e:
        log(f"Search failed: {e}")
        return False, 0, 0

def run_dataset(dataset):
    """运行单个数据集的所有参数组合"""
    log(f"\n{'='*60}")
    log(f"Dataset: {dataset.upper()}")
    log(f"N = {DATASETS[dataset]['N']:,}")
    log(f"{'='*60}")
    
    progress_file = f"{OUTPUT_BASE}/progress_{dataset}.json"
    summary_file = f"{OUTPUT_BASE}/results/{dataset}/summary.csv"
    
    Path(f"{OUTPUT_BASE}/results/{dataset}").mkdir(parents=True, exist_ok=True)
    
    progress = load_progress(progress_file)
    
    # 写 summary header
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f:
            f.write("params,build_time_s,index_size_mb,recall@10,qps,status,timestamp\n")
    
    for params in PARAM_SETS:
        param_str = get_param_str(params)
        task_id = f"{dataset}_{param_str}"
        
        if task_id in progress['completed']:
            log(f"Skip (done): {param_str}")
            continue
        if task_id in progress['failed']:
            log(f"Skip (failed): {param_str}")
            continue
        
        log(f"\n--- {param_str} ---")
        
        # 构建索引
        log(f"Building index...")
        success, build_time, index_size, index_dir = build_index(dataset, params, OUTPUT_BASE)
        
        if not success:
            log(f"Build failed")
            progress['failed'].append(task_id)
            save_progress(progress_file, progress)
            with open(summary_file, 'a') as f:
                f.write(f"{param_str},0,0,0,0,build_failed,{datetime.now().isoformat()}\n")
            continue
        
        log(f"Build done: {build_time:.1f}s, {index_size:.1f}MB")
        
        # 搜索测试
        log(f"Searching...")
        success, recall, qps = search_index(dataset, params, index_dir, OUTPUT_BASE)
        
        if not success:
            log(f"Search failed")
            progress['failed'].append(task_id)
            save_progress(progress_file, progress)
            with open(summary_file, 'a') as f:
                f.write(f"{param_str},{build_time:.2f},{index_size:.2f},0,0,search_failed,{datetime.now().isoformat()}\n")
            continue
        
        log(f"Result: Recall@10={recall:.2%}, QPS={qps:.0f}")
        
        # 记录结果
        progress['completed'].append(task_id)
        save_progress(progress_file, progress)
        with open(summary_file, 'a') as f:
            f.write(f"{param_str},{build_time:.2f},{index_size:.2f},{recall},{qps},success,{datetime.now().isoformat()}\n")
    
    log(f"\nDataset {dataset} done!")

def main():
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = list(DATASETS.keys())
    
    log(f"Running NHQ on datasets: {datasets}")
    
    start = time.time()
    for dataset in datasets:
        if dataset not in DATASETS:
            log(f"Unknown dataset: {dataset}")
            continue
        run_dataset(dataset)
    
    total_time = time.time() - start
    log(f"\n{'='*60}")
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
