#!/usr/bin/env python3
"""V2 数据集 UNG 实验 - 网格搜索版

参数搜索范围（基于 UNG 论文推荐）:
- max_degree (M): 32, 64, 96
- Lbuild (Lb): 100 (论文默认)
- Lsearch (Ls): 100, 400, 800
- cross_edges: 6 (containment/equality), 2 (overlap) - 论文建议

数据集（6个）:
- synth_192d, synth_512d, synth_768d_hc, synth_100k, yahoo800k, dbpedia560k
"""
import os
import sys
import subprocess
import time
import csv
from itertools import product

# 网格搜索参数（基于 UNG 论文）
M_VALUES = [32, 64, 96]
LB_VALUES = [100]  # 只用论文默认值，Lb=200 在部分数据集上 segfault
LS_VALUES = [100, 400, 800]

# cross_edges: overlap 用 2，其他用 6（论文建议）
CROSS_EDGES = {
    'containment': 6,
    'overlap': 2,
    'equality': 6,
}

K = 10
T = 16

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "synth_100k", "yahoo800k", "dbpedia560k"]
SCENARIOS = ["containment", "overlap", "equality"]

DATA_DIR = os.path.expanduser("~/benchmarks/datasets/discrete")
UNG_DIR = os.path.expanduser("~/benchmarks/discrete/UNG-dev")
BUILD_BIN = f"{UNG_DIR}/build/apps/build_UNG_index"
SEARCH_BIN = f"{UNG_DIR}/build/apps/search_UNG_index"
RESULTS_DIR = f"{UNG_DIR}/results_v2_grid"

DATASET_CONFIG = {
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "synth_100k": {"N": 800000, "D": 192},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
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

def run_cmd(cmd, desc=""):
    """运行命令并返回是否成功"""
    log(f"Running: {desc or cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR: {result.stderr[:500]}")
        return False
    return True

def build_index(dataset, M, Lb, scenario):
    """构建 UNG 索引"""
    ds_dir = f"{DATA_DIR}/{dataset}"

    base_file = f"{ds_dir}/{dataset}_base.bin"
    label_file = get_label_file(f"{ds_dir}/label_base.txt")

    # UNG scenario: general for containment/overlap, equality for equality
    build_scenario = "equality" if scenario == "equality" else "general"

    # 索引输出目录
    idx_dir = f"{RESULTS_DIR}/{dataset}/{build_scenario}"
    os.makedirs(idx_dir, exist_ok=True)

    # cross_edges 根据场景调整
    cross_edges = CROSS_EDGES.get(scenario, 6)
    idx_prefix = f"{idx_dir}/index_M={M}_L={Lb}_ce={cross_edges}"

    # 先清理旧的索引文件（防止崩溃后残留）
    import glob
    old_files = glob.glob(f"{idx_prefix}.*")
    if old_files:
        for f in old_files:
            os.remove(f)
        log(f"Cleaned {len(old_files)} old index files before build")

    # 构建命令
    cmd = (f"{BUILD_BIN} "
           f"--data_type float "
           f"--dist_fn L2 "
           f"--base_bin_file {base_file} "
           f"--base_label_file {label_file} "
           f"--index_path_prefix {idx_prefix} "
           f"--scenario {build_scenario} "
           f"--max_degree {M} "
           f"--Lbuild {Lb} "
           f"--num_cross_edges {cross_edges} "
           f"--num_threads {T} ")

    if run_cmd(cmd, f"Build {dataset} M={M} Lb={Lb} {scenario}"):
        return idx_prefix
    return None

def search_index(dataset, M, Lb, Ls, scenario, idx_prefix):
    """搜索并计算 recall"""
    ds_dir = f"{DATA_DIR}/{dataset}"

    # File suffix mapping: containment->and, overlap->or, equality->equal
    suffix_map = {"containment": "and", "overlap": "or", "equality": "equal"}
    suffix = suffix_map[scenario]

    base_file = f"{ds_dir}/{dataset}_base.bin"
    base_label = get_label_file(f"{ds_dir}/label_base.txt")
    query_file = f"{ds_dir}/{dataset}_query_{suffix}.bin"
    query_label_file = get_label_file(f"{ds_dir}/{dataset}_query_{suffix}.txt")
    gt_file = f"{ds_dir}/{dataset}_gt_{suffix}.bin"

    # 结果输出
    result_dir = f"{RESULTS_DIR}/{dataset}/{scenario}"
    os.makedirs(result_dir, exist_ok=True)
    cross_edges = CROSS_EDGES.get(scenario, 6)
    result_prefix = f"{result_dir}/M={M}_Lb={Lb}_Ls={Ls}_ce={cross_edges}"

    # Search scenario: containment 保持，其他不变
    search_scenario = "containment" if scenario == "containment" else scenario

    # 搜索命令
    cmd = (f"{SEARCH_BIN} "
           f"--data_type float "
           f"--dist_fn L2 "
           f"--base_bin_file {base_file} "
           f"--query_bin_file {query_file} "
           f"--base_label_file {base_label} "
           f"--query_label_file {query_label_file} "
           f"--gt_file {gt_file} "
           f"--K {K} "
           f"--index_path_prefix {idx_prefix} "
           f"--scenario {search_scenario} "
           f"--Lsearch {Ls} "
           f"--num_threads {T} "
           f"--result_path_prefix {result_prefix}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Search failed: {result.stderr[:200]}")
        return None

    # UNG 将结果写入 {result_prefix}result.csv
    # 格式: L,Cmps,QPS,Recall
    result_csv = f"{result_prefix}result.csv"
    recall, qps = None, None

    if os.path.exists(result_csv):
        with open(result_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # 解析第二行: L,Cmps,QPS,Recall
                parts = lines[1].strip().split(',')
                if len(parts) >= 4:
                    try:
                        qps = float(parts[2])
                        recall = float(parts[3]) / 100.0  # 转为小数
                    except ValueError:
                        pass

    if recall is None:
        log(f"  Warning: Could not parse {result_csv}")

    return {'recall': recall, 'qps': qps}

def run_grid_search(datasets=None, scenarios=None, cleanup=False):
    """运行网格搜索"""
    if datasets is None:
        datasets = V2_DATASETS
    if scenarios is None:
        scenarios = SCENARIOS

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 结果 CSV
    csv_path = f"{RESULTS_DIR}/summary.csv"
    csv_exists = os.path.exists(csv_path)

    # 加载已完成的配置
    completed = set()
    if csv_exists:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get('status', '')
                # 跳过 success 或 skip_* 状态
                if (status == 'success' and row.get('recall')) or status.startswith('skip'):
                    key = (row['dataset'], row['scenario'], int(row['M']), int(row['Lb']), int(row['Ls']))
                    completed.add(key)
        log(f"Found {len(completed)} completed configs in summary.csv")

    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'dataset', 'scenario', 'M', 'Lb', 'Ls', 'cross_edges', 'recall', 'qps', 'status', 'timestamp'
    ])
    if not csv_exists:
        writer.writeheader()

    total_configs = len(M_VALUES) * len(LB_VALUES) * len(LS_VALUES)
    total_runs = len(datasets) * len(scenarios) * total_configs

    log(f"Grid search: {len(datasets)} datasets × {len(scenarios)} scenarios × {total_configs} configs = {total_runs} runs")

    run_count = 0
    for dataset in datasets:
        for scenario in scenarios:
            log(f"\n{'='*60}")
            log(f"Dataset: {dataset}, Scenario: {scenario}")
            log(f"{'='*60}")

            cross_edges = CROSS_EDGES.get(scenario, 6)

            # 对每个 (M, Lb) 构建一次索引，不同 Ls 复用
            for M, Lb in product(M_VALUES, LB_VALUES):
                # 检查是否所有 Ls 都已完成，是则跳过构建
                all_done = all((dataset, scenario, M, Lb, Ls) in completed for Ls in LS_VALUES)
                if all_done:
                    log(f"All Ls done for M={M}, Lb={Lb}, skipping build")
                    run_count += len(LS_VALUES)
                    continue

                idx_prefix = build_index(dataset, M, Lb, scenario)
                if idx_prefix is None:
                    log(f"Build failed for M={M}, Lb={Lb}")
                    for Ls in LS_VALUES:
                        writer.writerow({
                            'dataset': dataset, 'scenario': scenario,
                            'M': M, 'Lb': Lb, 'Ls': Ls, 'cross_edges': cross_edges,
                            'recall': None, 'qps': None,
                            'status': 'build_failed',
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    csv_file.flush()
                    continue

                # 不同 Ls 搜索
                for Ls in LS_VALUES:
                    run_count += 1

                    # 跳过已完成的配置
                    key = (dataset, scenario, M, Lb, Ls)
                    if key in completed:
                        log(f"[{run_count}/{total_runs}] M={M}, Lb={Lb}, Ls={Ls} - SKIP (already done)")
                        continue

                    log(f"[{run_count}/{total_runs}] M={M}, Lb={Lb}, Ls={Ls}, ce={cross_edges}")

                    result = search_index(dataset, M, Lb, Ls, scenario, idx_prefix)

                    row = {
                        'dataset': dataset, 'scenario': scenario,
                        'M': M, 'Lb': Lb, 'Ls': Ls, 'cross_edges': cross_edges,
                        'recall': result['recall'] if result else None,
                        'qps': result['qps'] if result else None,
                        'status': 'success' if result else 'search_failed',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    writer.writerow(row)
                    csv_file.flush()

                    if result and result['recall'] is not None:
                        qps_str = f"{result['qps']:.1f}" if result['qps'] else "N/A"
                        log(f"  -> recall={result['recall']:.4f}, QPS={qps_str}")

                # 每个索引用完立即删除
                if cleanup:
                    import glob
                    deleted = 0
                    for f in glob.glob(f"{idx_prefix}.*"):
                        os.remove(f)
                        deleted += 1
                    if deleted:
                        log(f"  Cleaned up {deleted} index files")

    csv_file.close()
    log(f"\nResults saved to {csv_path}")

def cleanup_indices(datasets):
    """删除索引文件，只保留结果 CSV"""
    import shutil
    for dataset in datasets:
        for scenario_dir in ['general', 'equality']:
            idx_dir = f"{RESULTS_DIR}/{dataset}/{scenario_dir}"
            if os.path.exists(idx_dir):
                # 删除索引文件 (*.data, *.labels 等)
                for f in os.listdir(idx_dir):
                    fpath = os.path.join(idx_dir, f)
                    if os.path.isfile(fpath) and f.startswith('index_'):
                        os.remove(fpath)
                        log(f"Deleted: {fpath}")
    log("Cleanup complete. Summary CSV preserved.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='UNG V2 Grid Search')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='Datasets to run (default: all 6)')
    parser.add_argument('--scenario', nargs='+', default=None,
                        help='Scenarios to run (default: all 3)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Delete indices after completion to save space')
    args = parser.parse_args()

    datasets = args.dataset if args.dataset else V2_DATASETS
    scenarios = args.scenario if args.scenario else SCENARIOS

    # 验证数据集名称
    for ds in datasets:
        if ds not in DATASET_CONFIG:
            print(f"Unknown dataset: {ds}")
            print(f"Available: {list(DATASET_CONFIG.keys())}")
            sys.exit(1)

    run_grid_search(datasets, scenarios, cleanup=args.cleanup)

if __name__ == '__main__':
    main()
