#!/usr/bin/env python3
"""
DiskANN Fixed-EQ 实验脚本
使用合成标签进行Fixed-Length Equality实验

用法:
  cd ~/benchmarks/discrete/DiskANN
  python scripts/auto_diskann_fixed_eq.py                # 运行所有数据集
  python scripts/auto_diskann_fixed_eq.py arxiv          # 只运行指定数据集
"""

import os
import sys
import time
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================

DISKANN_DIR = "/home/remote/u7905817/benchmarks/discrete/DiskANN"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"
# UNG目录（复用相同的合成标签和转换后的数据）
UNG_DIR = "/home/remote/u7905817/benchmarks/discrete/UNG-dev"
LABEL_DIR = f"{UNG_DIR}/synthetic_labels"
UNG_DATA_DIR = f"{UNG_DIR}/data_bin"  # UNG转换后的bin格式数据

# DiskANN参数范围
R_VALUES = [32, 64, 128]  # max out-degree (simplified for grid search)
L_BUILD_VALUES = [100]  # build queue size (200 shows <0.3% recall improvement)
# Note: EQUALITY scenario requires larger L due to post-filter approach
# (search with OVERLAP, then filter results for EQUALITY)
L_SEARCH_VALUES = [100, 200, 500, 1000, 2000]  # search queue size
ALPHA_VALUES = [1.2]  # graph density (1.0/1.2/1.4 show <0.1% recall difference)
# FilteredLbuild: build complexity specifically for filtered points
# Higher values = better quality filtered graphs, 0 means use regular Lbuild
FILTERED_LBUILD_VALUES = [0]  # 0=disabled (100/200 show <0.2% recall improvement)
T = 16  # threads

DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "yfcc": {"N": 1000000, "D": 192},
    "LAION1M": {"N": 1000448, "D": 512},
    "tripclick": {"N": 1055976, "D": 768},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
}

K = 10

# ==================== 辅助函数 ====================

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
                'dataset', 'R', 'Lbuild', 'alpha', 'FilteredLbuild', 'Lsearch',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'latency_us', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, R, Lbuild, alpha, filtered_lbuild, Lsearch, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, R, Lbuild, alpha, filtered_lbuild, Lsearch,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('latency', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_prefix):
    """计算索引总大小"""
    import glob
    total_size = 0
    for f in glob.glob(f"{index_prefix}*"):
        if os.path.isfile(f):
            total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)

def cleanup_index(index_prefix):
    """删除索引文件以节省磁盘空间"""
    import glob
    files = glob.glob(f"{index_prefix}*")
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    if files:
        log(f"  Cleaned up {len(files)} index files")

def convert_labels_to_internal(original_labels_file, index_labels_file, query_labels_file, output_file):
    """转换查询标签到DiskANN内部ID"""
    from collections import defaultdict

    # Load both label files
    original = []
    with open(original_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(set(int(x) for x in line.split(',')))

    index = []
    with open(index_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                index.append(set(int(x) for x in line.split(',')))

    # Build mapping by finding co-occurrences
    orig_to_internal = {}
    orig_label_to_points = defaultdict(set)
    internal_label_to_points = defaultdict(set)

    for i, (orig_set, idx_set) in enumerate(zip(original, index)):
        for label in orig_set:
            orig_label_to_points[label].add(i)
        for label in idx_set:
            internal_label_to_points[label].add(i)

    for orig_label, orig_points in orig_label_to_points.items():
        for internal_label, internal_points in internal_label_to_points.items():
            if orig_points == internal_points:
                if orig_label not in orig_to_internal:
                    orig_to_internal[orig_label] = internal_label
                break

    # Convert query labels
    converted = []
    with open(query_labels_file) as f:
        for line in f:
            line = line.strip()
            if line:
                q_labels = [int(x) for x in line.split(',')]
                new_labels = [orig_to_internal.get(l, l) for l in q_labels]
                converted.append(','.join(map(str, new_labels)))

    with open(output_file, 'w') as f:
        for line in converted:
            f.write(line + '\n')

    return len(orig_to_internal)

def convert_gt_format(ung_gt_file, output_file, K=10):
    """转换UNG GT格式到DiskANN格式"""
    import numpy as np
    import struct
    import os

    # 从GT文件大小推算query数量
    gt_size = os.path.getsize(ung_gt_file)
    num_queries = gt_size // (K * 8)  # 每个query有K个(id, dist)对，每对8字节

    with open(ung_gt_file, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8),
                                dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
    gt_data = gt_data.reshape(num_queries, K)

    ids = np.ascontiguousarray(gt_data['idx']).astype(np.uint32)
    dists = np.ascontiguousarray(gt_data['dist']).astype(np.float32)

    with open(output_file, 'wb') as f:
        f.write(struct.pack('<ii', num_queries, K))
        ids.tofile(f)
        dists.tofile(f)

    return num_queries

# ==================== 主流程 ====================

def build_index(dataset, R, Lbuild, alpha, filtered_lbuild, output_base):
    """构建索引"""
    index_dir = f"{output_base}/indices/{dataset}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # 索引命名包含所有构建参数
    index_name = f"index_R={R}_L={Lbuild}_a={alpha}_fL={filtered_lbuild}"
    log_file = f"{index_dir}/{index_name}_build.log"
    index_prefix = f"{index_dir}/{index_name}"

    # 检查索引是否已存在
    if os.path.exists(f"{index_prefix}.data"):
        log(f"Index already exists: {index_prefix}")
        return True, 0, get_index_size(index_prefix), index_prefix

    cmd = [
        f"{DISKANN_DIR}/build/apps/build_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--index_path_prefix", index_prefix,
        "--label_file", f"{LABEL_DIR}/{dataset}/label_base.txt",
        "-R", str(R),
        "-L", str(Lbuild),
        "--alpha", str(alpha),
        "-T", str(T),
    ]

    # FilteredLbuild: 专门为filtered points设计的构建复杂度
    if filtered_lbuild > 0:
        cmd.extend(["--FilteredLbuild", str(filtered_lbuild)])

    log(f"Building: R={R}, L={Lbuild}, alpha={alpha}, FilteredL={filtered_lbuild}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=7200, check=True)
        build_time = time.time() - start
        index_size = get_index_size(index_prefix)
        log(f"  Build success: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size, index_prefix
    except subprocess.TimeoutExpired:
        log(f"  Build timeout (2h)")
        return False, 0, 0, None
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, 0, 0, None

def search_index(dataset, index_prefix, Lsearch, output_base):
    """搜索测试"""
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    idx_name = os.path.basename(index_prefix)
    log_file = f"{result_dir}/{idx_name}_Ls={Lsearch}_search.log"
    result_prefix = f"{result_dir}/{idx_name}_Ls={Lsearch}"

    # 转换后的查询标签文件
    converted_labels = f"{output_base}/converted_labels/{dataset}/query_labels.txt"
    gt_file = f"{output_base}/converted_labels/{dataset}/gt_diskann.bin"

    cmd = [
        f"{DISKANN_DIR}/build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", index_prefix,
        "--query_file", f"{DATA_DIR}/{dataset}/{dataset}_query_equal.bin",  # 使用与GT匹配的query_equal文件
        "--query_filters_file", converted_labels,
        "--gt_file", gt_file,
        "--label_type", "uint",
        "--recall_at", str(K),
        "--result_path", result_prefix,
        "--num_threads", str(T),
        "--filter_scenario", "equality",
        "-L", str(Lsearch),
    ]

    log(f"Searching: Lsearch={Lsearch}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # 解析输出获取metrics
        metrics = parse_diskann_output(result.stdout + result.stderr)
        if metrics:
            log(f"  Search success: Recall={metrics.get('recall', 0):.4f}, QPS={metrics.get('qps', 0):.2f}")
            return True, metrics
        else:
            log(f"  Search done but no valid results")
            return True, {'recall': 0, 'qps': 0, 'latency': 0}

    except subprocess.TimeoutExpired:
        log(f"  Search timeout (30min)")
        return False, None
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

def parse_diskann_output(output):
    """解析DiskANN搜索输出"""
    # DiskANN实际输出格式:
    #   Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
    #  100   119801.15              5.30               73.42         445.05        0.90
    metrics = {}
    for line in output.split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            try:
                L = int(parts[0])
                qps = float(parts[1])
                # Recall在最后一列 (索引5)
                recall = float(parts[5])
                latency = float(parts[3])  # Mean Latency

                # DiskANN有时输出百分比(16.60=16.60%)，有时输出小数(0.166)
                # 如果recall > 1，说明是百分比格式，需要除以100
                if recall > 1.0:
                    recall = recall / 100.0

                if recall >= 0:
                    metrics = {
                        'L': L,
                        'qps': qps,
                        'recall': min(recall, 1.0),  # Cap at 1.0 (处理浮点误差)
                        'latency': latency,
                    }
            except (ValueError, IndexError):
                continue
    return metrics

def prepare_data(dataset, output_base):
    """准备数据：转换标签和GT格式"""
    converted_dir = f"{output_base}/converted_labels/{dataset}"
    Path(converted_dir).mkdir(parents=True, exist_ok=True)

    # 先构建一个临时索引来获取标签映射
    tmp_index = f"{output_base}/tmp_index/{dataset}/tmp"
    Path(os.path.dirname(tmp_index)).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(f"{tmp_index}_labels.txt"):
        log(f"Building temporary index to get label mapping...")
        cmd = [
            f"{DISKANN_DIR}/build/apps/build_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
            "--index_path_prefix", tmp_index,
            "--label_file", f"{LABEL_DIR}/{dataset}/label_base.txt",
            "-R", "32", "-L", "50", "--alpha", "1.2", "-T", str(T),
        ]
        subprocess.run(cmd, capture_output=True, timeout=3600)

    # 转换查询标签
    converted_labels = f"{converted_dir}/query_labels.txt"
    if not os.path.exists(converted_labels):
        log(f"Converting query labels...")
        n_mapped = convert_labels_to_internal(
            f"{LABEL_DIR}/{dataset}/label_base.txt",
            f"{tmp_index}_labels.txt",
            f"{LABEL_DIR}/{dataset}/label_query.txt",
            converted_labels
        )
        log(f"  Mapped {n_mapped} labels")

    # 转换GT格式
    gt_file = f"{converted_dir}/gt_diskann.bin"
    if not os.path.exists(gt_file):
        log(f"Converting GT format...")
        n_queries = convert_gt_format(
            f"{LABEL_DIR}/{dataset}/gt.bin",
            gt_file,
            K
        )
        log(f"  Converted GT for {n_queries} queries")

def run_experiments(dataset, output_base):
    """运行所有实验"""
    summary_file = f"{output_base}/results/{dataset}/summary.csv"
    progress_file = f"{output_base}/progress/{dataset}.json"

    init_summary_file(summary_file)
    progress = load_progress(progress_file)

    # 准备数据
    prepare_data(dataset, output_base)

    # 计算总实验数
    total_builds = len(R_VALUES) * len(L_BUILD_VALUES) * len(ALPHA_VALUES) * len(FILTERED_LBUILD_VALUES)
    total_searches = total_builds * len(L_SEARCH_VALUES)
    log(f"Total configurations: {total_builds} builds, {total_searches} search experiments")

    for R in R_VALUES:
        for Lbuild in L_BUILD_VALUES:
            for alpha in ALPHA_VALUES:
                for filtered_lbuild in FILTERED_LBUILD_VALUES:
                    # 构建索引
                    task_key = f"build_{R}_{Lbuild}_{alpha}_{filtered_lbuild}"
                    index_name = f"index_R={R}_L={Lbuild}_a={alpha}_fL={filtered_lbuild}"
                    index_prefix = f"{output_base}/indices/{dataset}/{index_name}"

                    if task_key in progress["completed"]:
                        build_time = 0
                        index_size = get_index_size(index_prefix)
                    else:
                        success, build_time, index_size, index_prefix = build_index(
                            dataset, R, Lbuild, alpha, filtered_lbuild, output_base)
                        if not success:
                            progress["failed"].append(task_key)
                            save_progress(progress_file, progress)
                            continue
                        progress["completed"].append(task_key)
                        save_progress(progress_file, progress)

                    # 搜索测试
                    for Lsearch in L_SEARCH_VALUES:
                        if Lsearch < K:
                            continue

                        task_key = f"search_{R}_{Lbuild}_{alpha}_{filtered_lbuild}_{Lsearch}"
                        if task_key in progress["completed"]:
                            continue

                        success, metrics = search_index(dataset, index_prefix, Lsearch, output_base)
                        if success and metrics:
                            metrics['build_time'] = build_time
                            metrics['index_size'] = index_size
                            metrics['status'] = 'success'
                            append_to_summary(summary_file, dataset, R, Lbuild, alpha, filtered_lbuild, Lsearch, metrics)
                            progress["completed"].append(task_key)
                        else:
                            progress["failed"].append(task_key)

                        save_progress(progress_file, progress)

                    # 搜索完成后清理索引文件，节省磁盘空间
                    cleanup_index(index_prefix)

def main():
    output_base = f"{DISKANN_DIR}/data_fixed_eq"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        datasets = [sys.argv[1]]
    else:
        datasets = list(DATASETS_CONFIG.keys())

    for dataset in datasets:
        if dataset not in DATASETS_CONFIG:
            log(f"Unknown dataset: {dataset}")
            continue

        log(f"=" * 50)
        log(f"Starting experiments for {dataset}")
        log(f"=" * 50)

        try:
            run_experiments(dataset, output_base)
        except Exception as e:
            log(f"Error in {dataset}: {e}")
            import traceback
            traceback.print_exc()

    log("All experiments completed!")

if __name__ == '__main__':
    main()
