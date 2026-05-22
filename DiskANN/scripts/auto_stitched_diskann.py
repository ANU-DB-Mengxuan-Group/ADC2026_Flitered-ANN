#!/usr/bin/env python3
"""
StitchedDiskANN 实验脚本
使用合成标签进行Fixed-Length Equality实验

StitchedDiskANN vs FilterDiskANN:
- FilterDiskANN: 单个共享图 + 标签过滤
- StitchedDiskANN: 每个标签单独建图，然后拼接

用法:
  cd ~/benchmarks/discrete/DiskANN
  python scripts/auto_stitched_diskann.py                # 运行所有数据集
  python scripts/auto_stitched_diskann.py arxiv          # 只运行指定数据集
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

# StitchedDiskANN参数范围
# R: 每个per-label子图的max degree
# stitched_R: 拼接后最终图的degree
R_VALUES = [32, 64, 128]  # per-label graph max degree
STITCHED_R_VALUES = [32, 64, 96]  # final stitched graph degree (should be <= R)
L_BUILD_VALUES = [100]  # build queue size (200 shows <0.3% recall improvement, not worth 2x build time)
L_SEARCH_VALUES = [100, 200, 500, 1000, 2000]  # search queue size
ALPHA_VALUES = [1.2]  # graph density (1.2 is default, less variation needed for stitched)
T = 16  # threads

DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "yfcc": {"N": 1000000, "D": 192},
    "LAION1M": {"N": 1000448, "D": 512},
    "tripclick": {"N": 1055976, "D": 768},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
    # --- V2 validation datasets ---
    # V2 datasets use 1-based labels (label 0 is sentinel in DiskANN)
    "synth_192d": {"N": 800000, "D": 192,
        "label_base": f"{DATA_DIR}/synth_192d/label_base_1based.txt",
        "label_query": f"{DATA_DIR}/synth_192d/synth_192d_query_equal_1based.txt",
        "gt_bin": f"{DATA_DIR}/synth_192d/synth_192d_gt_equal.bin"},
    "synth_512d": {"N": 800000, "D": 512,
        "label_base": f"{DATA_DIR}/synth_512d/label_base_1based.txt",
        "label_query": f"{DATA_DIR}/synth_512d/synth_512d_query_equal_1based.txt",
        "gt_bin": f"{DATA_DIR}/synth_512d/synth_512d_gt_equal.bin"},
    "synth_768d_hc": {"N": 800000, "D": 768,
        "label_base": f"{DATA_DIR}/synth_768d_hc/label_base_1based.txt",
        "label_query": f"{DATA_DIR}/synth_768d_hc/synth_768d_hc_query_equal_1based.txt",
        "gt_bin": f"{DATA_DIR}/synth_768d_hc/synth_768d_hc_gt_equal.bin"},
    "yahoo800k": {"N": 800000, "D": 768,
        "label_base": f"{DATA_DIR}/yahoo800k/label_base_1based.txt",
        "label_query": f"{DATA_DIR}/yahoo800k/yahoo800k_query_equal_1based.txt",
        "gt_bin": f"{DATA_DIR}/yahoo800k/yahoo800k_gt_equal.bin"},
    "dbpedia560k": {"N": 560000, "D": 768,
        "label_base": f"{DATA_DIR}/dbpedia560k/label_base_1based.txt",
        "label_query": f"{DATA_DIR}/dbpedia560k/dbpedia560k_query_equal_1based.txt",
        "gt_bin": f"{DATA_DIR}/dbpedia560k/dbpedia560k_gt_equal.bin"},
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
                'dataset', 'R', 'stitched_R', 'Lbuild', 'alpha', 'Lsearch',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'latency_us', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, R, stitched_R, Lbuild, alpha, Lsearch, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, R, stitched_R, Lbuild, alpha, Lsearch,
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

def build_index(dataset, R, stitched_R, Lbuild, alpha, output_base):
    """构建Stitched索引"""
    index_dir = f"{output_base}/indices/{dataset}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # 索引命名包含所有构建参数
    index_name = f"stitched_R={R}_sR={stitched_R}_L={Lbuild}_a={alpha}"
    log_file = f"{index_dir}/{index_name}_build.log"
    index_prefix = f"{index_dir}/{index_name}"

    # 检查索引是否已存在
    if os.path.exists(f"{index_prefix}.data"):
        log(f"Index already exists: {index_prefix}")
        return True, 0, get_index_size(index_prefix), index_prefix

    config = DATASETS_CONFIG.get(dataset, {})
    label_file = config.get("label_base", f"{LABEL_DIR}/{dataset}/label_base.txt")

    cmd = [
        f"{DISKANN_DIR}/build/apps/build_stitched_index",
        "--data_type", "float",
        "--data_path", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--index_path_prefix", index_prefix,
        "--label_file", label_file,
        "-R", str(R),
        "-L", str(Lbuild),
        "--alpha", str(alpha),
        "--stitched_R", str(stitched_R),
        "-T", str(T),
    ]

    log(f"Building Stitched: R={R}, stitched_R={stitched_R}, L={Lbuild}, alpha={alpha}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=14400, check=True)  # 4 hour timeout for stitched (longer)
        build_time = time.time() - start
        index_size = get_index_size(index_prefix)
        log(f"  Build success: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size, index_prefix
    except subprocess.TimeoutExpired:
        log(f"  Build timeout (4h)")
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

    # 获取 per-dataset 标签路径（V2 数据集使用 DATA_DIR 下的原始标签）
    config = DATASETS_CONFIG.get(dataset, {})
    label_base_file = config.get("label_base", f"{LABEL_DIR}/{dataset}/label_base.txt")
    label_query_file = config.get("label_query", f"{LABEL_DIR}/{dataset}/label_query.txt")
    gt_bin_file = config.get("gt_bin", f"{LABEL_DIR}/{dataset}/gt.bin")

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
            "--label_file", label_base_file,
            "-R", "32", "-L", "50", "--alpha", "1.2", "-T", str(T),
        ]
        subprocess.run(cmd, capture_output=True, timeout=3600)

    # 转换查询标签
    converted_labels = f"{converted_dir}/query_labels.txt"
    if not os.path.exists(converted_labels):
        log(f"Converting query labels...")
        n_mapped = convert_labels_to_internal(
            label_base_file,
            f"{tmp_index}_labels.txt",
            label_query_file,
            converted_labels
        )
        log(f"  Mapped {n_mapped} labels")

    # 清理临时索引（只保留_labels.txt用于后续标签映射）
    import glob
    for f in glob.glob(f"{tmp_index}*"):
        if not f.endswith("_labels.txt") and os.path.isfile(f):
            os.remove(f)

    # 转换GT格式
    gt_file = f"{converted_dir}/gt_diskann.bin"
    if not os.path.exists(gt_file):
        log(f"Converting GT format...")
        n_queries = convert_gt_format(
            gt_bin_file,
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
    total_builds = len(R_VALUES) * len(STITCHED_R_VALUES) * len(L_BUILD_VALUES) * len(ALPHA_VALUES)
    total_searches = total_builds * len(L_SEARCH_VALUES)
    log(f"Total configurations: {total_builds} builds, {total_searches} search experiments")

    for R in R_VALUES:
        for stitched_R in STITCHED_R_VALUES:
            # stitched_R should be <= R
            if stitched_R > R:
                continue

            for Lbuild in L_BUILD_VALUES:
                for alpha in ALPHA_VALUES:
                    # 构建索引
                    task_key = f"build_{R}_{stitched_R}_{Lbuild}_{alpha}"
                    index_name = f"stitched_R={R}_sR={stitched_R}_L={Lbuild}_a={alpha}"
                    index_prefix = f"{output_base}/indices/{dataset}/{index_name}"

                    if task_key in progress["completed"]:
                        build_time = 0
                        index_size = get_index_size(index_prefix)
                    else:
                        success, build_time, index_size, index_prefix = build_index(
                            dataset, R, stitched_R, Lbuild, alpha, output_base)
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

                        task_key = f"search_{R}_{stitched_R}_{Lbuild}_{alpha}_{Lsearch}"
                        if task_key in progress["completed"]:
                            continue

                        success, metrics = search_index(dataset, index_prefix, Lsearch, output_base)
                        if success and metrics:
                            metrics['build_time'] = build_time
                            metrics['index_size'] = index_size
                            metrics['status'] = 'success'
                            append_to_summary(summary_file, dataset, R, stitched_R, Lbuild, alpha, Lsearch, metrics)
                            progress["completed"].append(task_key)
                        else:
                            progress["failed"].append(task_key)

                        save_progress(progress_file, progress)

                    # 搜索完成后清理索引文件，节省磁盘空间
                    cleanup_index(index_prefix)

def main():
    output_base = f"{DISKANN_DIR}/data_stitched_eq"
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
        log(f"Starting StitchedDiskANN experiments for {dataset}")
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
