#!/usr/bin/env python3
"""
StitchedVamana 原始标签实验脚本
支持三种过滤场景: containment(AND), overlap(OR), equality(EQUAL)

对同一个索引跑三种场景的搜索，避免重复构建。

用法:
  cd ~/benchmarks/discrete/DiskANN
  python scripts/auto_stitched_diskann_original.py                    # 所有数据集所有场景
  python scripts/auto_stitched_diskann_original.py arxiv              # 指定数据集
  python scripts/auto_stitched_diskann_original.py arxiv containment  # 指定数据集和场景
"""

import os
import sys
import time
import json
import csv
import glob
import struct
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ==================== 配置 ====================

DISKANN_DIR = "/home/remote/u7905817/benchmarks/discrete/DiskANN"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"

# StitchedVamana参数（基于Fixed-EQ敏感性分析精简）
R_VALUES = [32, 64, 128]
STITCHED_R_VALUES = [32, 64, 96]  # stitched_R <= R
L_BUILD = 100
ALPHA = 1.2
L_SEARCH_VALUES = [100, 200, 500, 1000, 2000]
T = 16
K = 10

DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "yfcc": {"N": 1000000, "D": 192},
    "LAION1M": {"N": 1000448, "D": 512},
    "tripclick": {"N": 1055976, "D": 768},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
    # --- V2 validation datasets ---
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
}

SCENARIOS = ["containment", "overlap", "equality"]

SCENARIO_SUFFIX = {
    "containment": "and",
    "overlap": "or",
    "equality": "equal",
}

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
                'dataset', 'scenario', 'R', 'stitched_R', 'Lsearch',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'latency_us', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, scenario, R, stitched_R, Lsearch, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, scenario, R, stitched_R, Lsearch,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('latency', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_prefix):
    total_size = 0
    for f in glob.glob(f"{index_prefix}*"):
        if os.path.isfile(f):
            total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)

def cleanup_index(index_prefix):
    files = glob.glob(f"{index_prefix}*")
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    if files:
        log(f"  Cleaned up {len(files)} index files")

def parse_diskann_output(output):
    metrics = {}
    for line in output.split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            try:
                L = int(parts[0])
                qps = float(parts[1])
                recall = float(parts[5])
                latency = float(parts[3])
                if recall > 1.0:
                    recall = recall / 100.0
                if recall >= 0:
                    metrics = {
                        'L': L, 'qps': qps,
                        'recall': min(recall, 1.0),
                        'latency': latency,
                    }
            except (ValueError, IndexError):
                continue
    return metrics

# ==================== 标签/GT转换 ====================

def convert_labels_to_internal(original_labels_file, index_labels_file, query_labels_file, output_file):
    """转换查询标签到DiskANN内部ID"""
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

    log(f"  Mapped {len(orig_to_internal)} labels")
    return len(orig_to_internal)

def convert_gt_format(ung_gt_file, output_file, K=10):
    gt_size = os.path.getsize(ung_gt_file)
    num_queries = gt_size // (K * 8)

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

# ==================== 数据准备 ====================

def prepare_label_mapping(dataset, output_base):
    """构建临时索引获取标签映射（所有场景共用）"""
    tmp_index = f"{output_base}/tmp_index/{dataset}/tmp"
    Path(os.path.dirname(tmp_index)).mkdir(parents=True, exist_ok=True)

    label_file = f"{DATA_DIR}/{dataset}/label_base.txt"

    if not os.path.exists(f"{tmp_index}_labels.txt"):
        log(f"Building temporary index for label mapping...")
        cmd = [
            f"{DISKANN_DIR}/build/apps/build_memory_index",
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
            "--index_path_prefix", tmp_index,
            "--label_file", label_file,
            "-R", "32", "-L", "50", "--alpha", "1.2", "-T", str(T),
        ]
        subprocess.run(cmd, capture_output=True, timeout=3600)

        # 清理临时索引，只保留_labels.txt
        for f in glob.glob(f"{tmp_index}*"):
            if not f.endswith("_labels.txt") and os.path.isfile(f):
                os.remove(f)

    return tmp_index, label_file

def prepare_scenario_data(dataset, scenario, output_base, tmp_index, label_file):
    suffix = SCENARIO_SUFFIX[scenario]
    converted_dir = f"{output_base}/converted_labels/{dataset}/{scenario}"
    Path(converted_dir).mkdir(parents=True, exist_ok=True)

    converted_labels = f"{converted_dir}/query_labels.txt"
    if not os.path.exists(converted_labels):
        log(f"Converting query labels for {scenario}...")
        query_labels_file = f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.txt"
        convert_labels_to_internal(
            label_file,
            f"{tmp_index}_labels.txt",
            query_labels_file,
            converted_labels
        )

    gt_file = f"{converted_dir}/gt_diskann.bin"
    if not os.path.exists(gt_file):
        log(f"Converting GT for {scenario}...")
        ung_gt = f"{DATA_DIR}/{dataset}/{dataset}_gt_{suffix}.bin"
        if os.path.exists(ung_gt):
            n_queries = convert_gt_format(ung_gt, gt_file, K)
            log(f"  Converted GT: {n_queries} queries")
        else:
            log(f"  WARNING: GT file not found: {ung_gt}")
            return False

    return True

# ==================== 构建与搜索 ====================

def build_index(dataset, R, stitched_R, output_base):
    """构建StitchedVamana索引"""
    index_dir = f"{output_base}/indices/{dataset}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    index_name = f"stitched_R={R}_sR={stitched_R}"
    index_prefix = f"{index_dir}/{index_name}"
    log_file = f"{index_dir}/{index_name}_build.log"

    if os.path.exists(f"{index_prefix}.data"):
        log(f"Index already exists: R={R}, sR={stitched_R}")
        return True, 0, get_index_size(index_prefix), index_prefix

    label_file = f"{DATA_DIR}/{dataset}/label_base.txt"

    cmd = [
        f"{DISKANN_DIR}/build/apps/build_stitched_index",
        "--data_type", "float",
        "--data_path", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--index_path_prefix", index_prefix,
        "--label_file", label_file,
        "-R", str(R),
        "-L", str(L_BUILD),
        "--alpha", str(ALPHA),
        "--stitched_R", str(stitched_R),
        "-T", str(T),
    ]

    log(f"Building StitchedVamana: R={R}, sR={stitched_R}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=14400, check=True)
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

def search_index(dataset, scenario, index_prefix, Lsearch, output_base):
    suffix = SCENARIO_SUFFIX[scenario]
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    idx_name = os.path.basename(index_prefix)
    log_file = f"{result_dir}/{idx_name}_{scenario}_Ls={Lsearch}.log"
    result_prefix = f"{result_dir}/{idx_name}_{scenario}_Ls={Lsearch}"

    converted_labels = f"{output_base}/converted_labels/{dataset}/{scenario}/query_labels.txt"
    gt_file = f"{output_base}/converted_labels/{dataset}/{scenario}/gt_diskann.bin"
    query_file = f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.bin"

    cmd = [
        f"{DISKANN_DIR}/build/apps/search_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", index_prefix,
        "--query_file", query_file,
        "--query_filters_file", converted_labels,
        "--gt_file", gt_file,
        "--label_type", "uint",
        "--recall_at", str(K),
        "--result_path", result_prefix,
        "--num_threads", str(T),
        "--filter_scenario", scenario,
        "-L", str(Lsearch),
    ]

    log(f"Searching [{scenario}]: Ls={Lsearch}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        metrics = parse_diskann_output(result.stdout + result.stderr)
        if metrics:
            log(f"  R@10={metrics['recall']:.4f}, QPS={metrics['qps']:.0f}")
            return True, metrics
        else:
            log(f"  No valid results parsed")
            return True, {'recall': 0, 'qps': 0, 'latency': 0}

    except subprocess.TimeoutExpired:
        log(f"  Search timeout")
        return False, None
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

# ==================== 主流程 ====================

def run_experiments(dataset, scenarios, output_base):
    summary_file = f"{output_base}/results/{dataset}/summary.csv"
    progress_file = f"{output_base}/progress/{dataset}.json"

    init_summary_file(summary_file)
    progress = load_progress(progress_file)

    tmp_index, label_file = prepare_label_mapping(dataset, output_base)

    valid_scenarios = []
    for scenario in scenarios:
        if prepare_scenario_data(dataset, scenario, output_base, tmp_index, label_file):
            valid_scenarios.append(scenario)
        else:
            log(f"Skipping {scenario} (missing data)")

    if not valid_scenarios:
        log(f"No valid scenarios for {dataset}")
        return

    # 计算实验量（stitched_R <= R）
    n_builds = sum(1 for R in R_VALUES for sR in STITCHED_R_VALUES if sR <= R)
    total = n_builds * len(valid_scenarios) * len(L_SEARCH_VALUES)
    log(f"Total: {n_builds} builds × {len(valid_scenarios)} scenarios × {len(L_SEARCH_VALUES)} Lsearch = {total} searches")

    for R in R_VALUES:
        for stitched_R in STITCHED_R_VALUES:
            if stitched_R > R:
                continue

            build_key = f"build_R={R}_sR={stitched_R}"
            index_prefix = f"{output_base}/indices/{dataset}/stitched_R={R}_sR={stitched_R}"

            if build_key in progress["completed"]:
                build_time = 0
                index_size = get_index_size(index_prefix)
            else:
                success, build_time, index_size, index_prefix = build_index(
                    dataset, R, stitched_R, output_base)
                if not success:
                    progress["failed"].append(build_key)
                    save_progress(progress_file, progress)
                    continue
                progress["completed"].append(build_key)
                save_progress(progress_file, progress)

            for scenario in valid_scenarios:
                for Lsearch in L_SEARCH_VALUES:
                    task_key = f"search_{scenario}_R={R}_sR={stitched_R}_Ls={Lsearch}"
                    if task_key in progress["completed"]:
                        continue

                    success, metrics = search_index(dataset, scenario, index_prefix, Lsearch, output_base)
                    if success and metrics:
                        metrics['build_time'] = build_time
                        metrics['index_size'] = index_size
                        metrics['status'] = 'success'
                        append_to_summary(summary_file, dataset, scenario, R, stitched_R, Lsearch, metrics)
                        progress["completed"].append(task_key)
                    else:
                        progress["failed"].append(task_key)

                    save_progress(progress_file, progress)

            cleanup_index(index_prefix)

def main():
    output_base = f"{DISKANN_DIR}/data_stitched_original"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    datasets = list(DATASETS_CONFIG.keys())
    scenarios = SCENARIOS

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in DATASETS_CONFIG:
            datasets = [arg]
        else:
            print(f"Unknown dataset: {arg}")
            print(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)

    if len(sys.argv) > 2:
        arg = sys.argv[2]
        if arg in SCENARIOS:
            scenarios = [arg]
        else:
            print(f"Unknown scenario: {arg}")
            print(f"Available: {SCENARIOS}")
            sys.exit(1)

    for dataset in datasets:
        log(f"{'=' * 60}")
        log(f"StitchedVamana (original labels): {dataset}")
        log(f"Scenarios: {scenarios}")
        log(f"{'=' * 60}")

        try:
            run_experiments(dataset, scenarios, output_base)
        except Exception as e:
            log(f"Error in {dataset}: {e}")
            import traceback
            traceback.print_exc()

    log("All experiments completed!")

if __name__ == '__main__':
    main()
