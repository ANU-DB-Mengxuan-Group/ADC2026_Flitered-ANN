#!/usr/bin/env python3
"""
UNG 原始元数据实验脚本
使用原始标签进行 Containment(AND)/Overlap(OR)/Equality 实验

用法:
  cd ~/benchmarks/discrete/UNG-dev
  python bash/auto_ung_original.py                    # 运行所有数据集和场景
  python bash/auto_ung_original.py arxiv              # 只运行指定数据集
  python bash/auto_ung_original.py arxiv containment  # 指定数据集和场景
"""

import os
import sys
import time
import json
import csv
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================

UNG_DIR = "/home/remote/u7905817/benchmarks/discrete/UNG-dev"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"

# 参数范围
MAX_DEGREES = [32, 48, 64, 96]
L_BUILDS = [100, 150, 200]
L_SEARCHES = [100, 200, 300, 500]
NUM_CROSS_EDGES = 6
T = 16

# 场景配置
# build_scenario: 构建时用的scenario (equality或general)
# search_scenario: 搜索时用的scenario
SCENARIOS = {
    "containment": {"build": "general", "search": "containment"},  # AND
    "overlap": {"build": "general", "search": "overlap"},          # OR
    "equality": {"build": "equality", "search": "equality"},       # EQUAL
}

# 数据集配置 (所有6个数据集)
DATASETS_CONFIG = {
    "arxiv": {"N": 132678, "D": 768},
    "tripclick": {"N": 1055976, "D": 768},
    "LAION1M": {"N": 1000448, "D": 512},
    "yfcc": {"N": 1000000, "D": 192},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},  # 注意: 1M不是5M!
    # --- V2 validation datasets ---
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
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
                'dataset', 'scenario', 'max_degree', 'Lbuild', 'Lsearch',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, scenario, max_degree, Lbuild, Lsearch, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, scenario, max_degree, Lbuild, Lsearch,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_prefix):
    total_size = 0
    for f in glob.glob(f"{index_prefix}*"):
        if os.path.isfile(f):
            total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)

def parse_ung_result(result_prefix):
    result_file = f"{result_prefix}result.csv"
    try:
        with open(result_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                recall = float(row.get('Recall', 0))
                qps = float(row.get('QPS', 0))
                return {'recall': recall, 'qps': qps}
    except Exception as e:
        log(f"Warning: parse result failed: {e}")
    return None

def get_scenario_suffix(scenario):
    """获取场景对应的文件后缀"""
    mapping = {
        "containment": "and",
        "overlap": "or",
        "equality": "equal",
    }
    return mapping.get(scenario, scenario)

def get_label_file(path):
    """V2 数据集标签从 0 开始，UNG 用 0 做哨兵会 segfault。
    如果存在 _1based 版本就自动使用。训练数据集没有该文件，回退到原始路径。"""
    base, ext = os.path.splitext(path)
    onebased = f"{base}_1based{ext}"
    if os.path.exists(onebased):
        log(f"  Using 1-based label file: {os.path.basename(onebased)}")
        return onebased
    return path

# ==================== 主流程 ====================

def build_index(dataset, scenario, max_degree, Lbuild, config, output_base):
    """构建索引"""
    # containment和overlap共用general索引
    build_scenario = SCENARIOS[scenario]["build"]
    index_dir = f"{output_base}/indices_original/{dataset}/{build_scenario}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{index_dir}/M={max_degree}_L={Lbuild}_build.log"
    index_prefix = f"{index_dir}/index_M={max_degree}_L={Lbuild}"

    cmd = [
        f"{UNG_DIR}/build/apps/build_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--base_label_file", get_label_file(f"{DATA_DIR}/{dataset}/label_base.txt"),
        "--index_path_prefix", index_prefix,
        "--scenario", build_scenario,
        "--max_degree", str(max_degree),
        "--Lbuild", str(Lbuild),
        "--num_cross_edges", str(NUM_CROSS_EDGES),
        "--num_threads", str(T),
    ]

    log(f"Building [{scenario}]: max_degree={max_degree}, Lbuild={Lbuild}")
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

def search_index(dataset, scenario, index_prefix, Lsearch, config, output_base):
    """搜索测试"""
    result_dir = f"{output_base}/results_original/{dataset}/{scenario}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    idx_name = os.path.basename(index_prefix)
    log_file = f"{result_dir}/{idx_name}_Ls={Lsearch}_search.log"
    result_prefix = f"{result_dir}/{idx_name}_Ls={Lsearch}"

    # 场景对应的文件后缀和搜索scenario
    suffix = get_scenario_suffix(scenario)
    search_scenario = SCENARIOS[scenario]["search"]

    cmd = [
        f"{UNG_DIR}/build/apps/search_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_base.bin",
        "--query_bin_file", f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.bin",
        "--base_label_file", get_label_file(f"{DATA_DIR}/{dataset}/label_base.txt"),
        "--query_label_file", get_label_file(f"{DATA_DIR}/{dataset}/{dataset}_query_{suffix}.txt"),
        "--gt_file", f"{DATA_DIR}/{dataset}/{dataset}_gt_{suffix}.bin",
        "--K", str(K),
        "--index_path_prefix", index_prefix,
        "--scenario", search_scenario,
        "--Lsearch", str(Lsearch),
        "--num_threads", str(T),
        "--result_path_prefix", result_prefix,
    ]

    log(f"Searching [{scenario}]: Lsearch={Lsearch}")

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=1800, check=True)

        metrics = parse_ung_result(result_prefix)
        if metrics and metrics['recall'] > 0:
            log(f"  Search success: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")
            return True, metrics
        else:
            log(f"  Search done but no valid results")
            return True, {'recall': 0, 'qps': 0}

    except subprocess.TimeoutExpired:
        log(f"  Search timeout (30min)")
        return False, None
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

def cleanup_index(index_prefix):
    for f in glob.glob(f"{index_prefix}*"):
        try:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
            log(f"  Cleaned: {os.path.basename(f)}")
        except Exception as e:
            log(f"  Cleanup failed: {e}")

def run_scenario(dataset, scenario, config, output_base):
    """运行单个场景"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()}, Scenario: {scenario}")
    log(f"N = {config['N']:,}, D = {config['D']}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_{dataset}_{scenario}.json"
    summary_file = f"{output_base}/results_original/{dataset}/summary.csv"
    build_scenario = SCENARIOS[scenario]["build"]

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for max_degree in MAX_DEGREES:
        for Lbuild in L_BUILDS:
            # 构建任务key用build_scenario，因为general索引可复用
            task_key_build = f"{dataset}_{build_scenario}_M={max_degree}_Lb={Lbuild}_build"
            index_dir = f"{output_base}/indices_original/{dataset}/{build_scenario}"
            index_prefix = f"{index_dir}/index_M={max_degree}_L={Lbuild}"

            build_time = 0
            index_size = 0

            # 检查索引是否已存在（可能被其他场景构建过）
            index_exists = os.path.exists(f"{index_prefix}graph") or os.path.exists(f"{index_prefix}meta")

            if task_key_build not in progress.get("completed", []) and not index_exists:
                build_ok, build_time, index_size, index_prefix = build_index(
                    dataset, scenario, max_degree, Lbuild, config, output_base)

                if not build_ok:
                    progress.setdefault("failed", []).append(task_key_build)
                    save_progress(progress_file, progress)
                    for Lsearch in L_SEARCHES:
                        append_to_summary(summary_file, dataset, scenario, max_degree, Lbuild, Lsearch, {
                            'build_time': 0, 'index_size': 0,
                            'recall': 0, 'qps': 0, 'status': 'build_failed'
                        })
                    continue

                progress.setdefault("completed", []).append(task_key_build)
                save_progress(progress_file, progress)
            else:
                log(f"Skip build (exists): M={max_degree}, Lb={Lbuild}")
                index_size = get_index_size(index_prefix)

            for Lsearch in L_SEARCHES:
                task_key = f"{dataset}_{scenario}_M={max_degree}_Lb={Lbuild}_Ls={Lsearch}"

                if task_key in progress.get("completed", []):
                    log(f"Skip (done): {task_key}")
                    continue

                if task_key in progress.get("failed", []):
                    log(f"Skip (failed): {task_key}")
                    continue

                log("")
                log(f"--- {task_key} ---")

                search_ok, metrics = search_index(
                    dataset, scenario, index_prefix, Lsearch, config, output_base)

                if not search_ok:
                    progress.setdefault("failed", []).append(task_key)
                    save_progress(progress_file, progress)
                    append_to_summary(summary_file, dataset, scenario, max_degree, Lbuild, Lsearch, {
                        'build_time': build_time, 'index_size': index_size,
                        'recall': 0, 'qps': 0, 'status': 'search_failed'
                    })
                    continue

                progress.setdefault("completed", []).append(task_key)
                save_progress(progress_file, progress)

                metrics = metrics or {'recall': 0, 'qps': 0}
                metrics['build_time'] = build_time
                metrics['index_size'] = index_size
                metrics['status'] = 'success'
                append_to_summary(summary_file, dataset, scenario, max_degree, Lbuild, Lsearch, metrics)

            # 只有 equality 场景需要立即清理（独占索引）
            # general 索引 (containment/overlap共用) 在最后统一清理
            if index_prefix and build_scenario == "equality":
                cleanup_index(index_prefix)

    log("")
    log(f"Scenario {scenario} done!")

# ==================== 主入口 ====================

def main():
    datasets = list(DATASETS_CONFIG.keys())
    scenarios = list(SCENARIOS.keys())

    # 解析命令行参数
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
            log(f"Available: {list(SCENARIOS.keys())}")
            sys.exit(1)

    log("=" * 60)
    log("UNG Original Metadata Experiment")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log(f"Scenarios: {', '.join(scenarios)}")
    log("")

    # 检查可执行文件
    build_exe = f"{UNG_DIR}/build/apps/build_UNG_index"
    if not os.path.exists(build_exe):
        log(f"Error: UNG not compiled at {build_exe}")
        sys.exit(1)

    start_time = time.time()

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        for scenario in scenarios:
            try:
                run_scenario(dataset, scenario, config, UNG_DIR)
            except KeyboardInterrupt:
                log("User interrupted")
                sys.exit(0)
            except Exception as e:
                log(f"Error: {e}")
                import traceback
                traceback.print_exc()

        # 清理该数据集的 general 索引目录（containment/overlap共用）
        general_index_dir = f"{UNG_DIR}/indices_original/{dataset}/general"
        if os.path.exists(general_index_dir):
            log(f"\nCleaning up general indices for {dataset}...")
            try:
                shutil.rmtree(general_index_dir)
                log(f"  Removed: {general_index_dir}")
            except Exception as e:
                log(f"  Cleanup failed: {e}")

    total_time = time.time() - start_time
    log("")
    log("=" * 60)
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log("=" * 60)

if __name__ == "__main__":
    main()
