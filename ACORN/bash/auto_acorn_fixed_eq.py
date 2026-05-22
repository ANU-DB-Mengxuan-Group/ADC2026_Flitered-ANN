#!/usr/bin/env python3
"""
ACORN Fixed-EQ 实验脚本
使用合成标签（4属性×3值）进行Fixed-Length Equality实验

用法:
  cd ~/benchmarks/discrete/ACORN
  python bash/auto_acorn_fixed_eq.py                # 运行所有数据集
  python bash/auto_acorn_fixed_eq.py arxiv          # 只运行指定数据集
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

ACORN_DIR = "/home/remote/u7905817/benchmarks/discrete/ACORN"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"
SYNTHETIC_LABELS = f"{ACORN_DIR}/synthetic_labels"
OUTPUT_BASE = f"{ACORN_DIR}/data_fixed_eq"

# 参数范围
Ms = [32, 48, 64]
M_betas = [48, 64, 96, 128]
gammas = [4, 8, 12, 24]

DATASETS_CONFIG = {
    "arxiv": {"N": 132687},
    "yfcc": {"N": 1000000},
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
                'dataset', 'M', 'M_beta', 'gamma',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, M, M_beta, gamma, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, M, M_beta, gamma,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_path):
    try:
        return os.path.getsize(index_path) / (1024 * 1024)
    except:
        return 0.0

def parse_result_csv(csv_file):
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            best_recall = 0.0
            best_qps = 0.0
            for row in reader:
                recall = float(row.get('Recall', row.get('recall', 0)))
                qps = float(row.get('QPS', row.get('qps', 0)))
                if recall > best_recall:
                    best_recall = recall
                    best_qps = qps
            return {'recall': best_recall, 'qps': best_qps}
    except Exception as e:
        log(f"Warning: parse CSV failed: {e}")
    return None

# ==================== 主流程 ====================

def build_index(dataset, M, M_beta, gamma, config, output_base):
    """构建索引"""
    # 注意：ACORN会自动在index_dir下创建{dataset}子目录
    index_dir = f"{output_base}/indices"
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{index_dir}/{dataset}").mkdir(parents=True, exist_ok=True)

    log_file = f"{index_dir}/{dataset}/M={M}_Mb={M_beta}_gamma={gamma}_build.log"
    N = config['N']

    # Build参数顺序: N, gamma, data_file, M, M_beta, output_path, dataset
    cmd = [
        f"{ACORN_DIR}/build/demos/build_acorn_index",
        str(N), str(gamma),
        f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs",
        str(M), str(M_beta),
        index_dir, dataset
    ]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'

    log(f"Building: M={M}, M_beta={M_beta}, gamma={gamma}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                          timeout=3600, check=True)
        build_time = time.time() - start
        index_path = f"{index_dir}/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        index_size = get_index_size(index_path)
        log(f"  Build success: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size
    except subprocess.TimeoutExpired:
        log(f"  Build timeout (1h)")
        return False, 0, 0
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, 0, 0

def search_index(dataset, M, M_beta, gamma, config, output_base):
    """搜索测试"""
    index_dir = f"{output_base}/indices"
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{result_dir}/M={M}_Mb={M_beta}_gamma={gamma}_search.log"
    N = config['N']

    # Search参数顺序: N, gamma, dataset, M, M_beta, index_dir, scenario, result_dir,
    #                 base_fvecs, label_base, query_fvecs, query_labels, gt, K
    # 使用合成标签进行Fixed-EQ实验
    cmd = [
        f"{ACORN_DIR}/build/demos/search_acorn_index",
        str(N), str(gamma), dataset,
        str(M), str(M_beta), index_dir, "equal", result_dir,
        f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs",
        f"{SYNTHETIC_LABELS}/{dataset}/label_base_synthetic.txt",
        f"{DATA_DIR}/{dataset}/{dataset}_query_equal.fvecs",
        f"{SYNTHETIC_LABELS}/{dataset}/label_query_synthetic.txt",
        f"{SYNTHETIC_LABELS}/{dataset}/gt_synthetic.txt",
        str(K)
    ]

    env = os.environ.copy()
    env['debugSearchFlag'] = '0'

    log(f"Searching: M={M}, M_beta={M_beta}, gamma={gamma}")

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                          timeout=1800, check=True)

        csv_file = f"{result_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
        metrics = parse_result_csv(csv_file)
        if metrics:
            log(f"  Search success: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")
            return True, metrics
        else:
            log(f"  Search done but no results")
            return True, {'recall': 0, 'qps': 0}

    except subprocess.TimeoutExpired:
        log(f"  Search timeout (30min)")
        return False, None
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

def cleanup_index(dataset, M, M_beta, gamma, output_base):
    """删除索引文件节省空间"""
    index_dir = f"{output_base}/indices/{dataset}"
    # ACORN会生成gamma和gamma=1两个索引
    patterns = [
        f"hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json",
        f"hybrid_M={M}_Mb={M_beta}_gamma=1.json",
    ]
    for pattern in patterns:
        index_path = f"{index_dir}/{pattern}"
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
                log(f"  Cleaned: {pattern}")
            except Exception as e:
                log(f"  Cleanup failed: {e}")

def run_dataset(dataset, config, output_base):
    """运行单个数据集"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()} (Fixed-EQ)")
    log(f"N = {config['N']:,}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_{dataset}.json"
    summary_file = f"{output_base}/results/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for M in Ms:
        for M_beta in M_betas:
            # 检查参数约束
            # 1. M_beta=128 与 M=32 组合会SIGABRT
            if M == 32 and M_beta >= 128:
                continue
            # 2. M_beta < 2*M*gamma
            if M_beta >= 2 * M * max(gammas):
                continue

            for gamma in gammas:
                if M_beta >= 2 * M * gamma:
                    continue

                task_key = f"{dataset}_M={M}_Mb={M_beta}_g={gamma}"

                if task_key in progress.get("completed", []):
                    log(f"Skip (done): {task_key}")
                    continue

                if task_key in progress.get("failed", []):
                    log(f"Skip (failed): {task_key}")
                    continue

                log("")
                log(f"--- {task_key} ---")

                # 构建
                build_ok, build_time, index_size = build_index(
                    dataset, M, M_beta, gamma, config, output_base)

                if not build_ok:
                    progress.setdefault("failed", []).append(task_key)
                    save_progress(progress_file, progress)
                    append_to_summary(summary_file, dataset, M, M_beta, gamma, {
                        'build_time': 0, 'index_size': 0,
                        'recall': 0, 'qps': 0, 'status': 'build_failed'
                    })
                    continue

                # 搜索
                search_ok, metrics = search_index(
                    dataset, M, M_beta, gamma, config, output_base)

                # 搜索完成后清理索引
                cleanup_index(dataset, M, M_beta, gamma, output_base)

                if not search_ok:
                    progress.setdefault("failed", []).append(task_key)
                    save_progress(progress_file, progress)
                    append_to_summary(summary_file, dataset, M, M_beta, gamma, {
                        'build_time': build_time, 'index_size': index_size,
                        'recall': 0, 'qps': 0, 'status': 'search_failed'
                    })
                    continue

                # 成功
                progress.setdefault("completed", []).append(task_key)
                save_progress(progress_file, progress)

                metrics = metrics or {'recall': 0, 'qps': 0}
                metrics['build_time'] = build_time
                metrics['index_size'] = index_size
                metrics['status'] = 'success'
                append_to_summary(summary_file, dataset, M, M_beta, gamma, metrics)

    log("")
    log(f"Dataset {dataset} done!")
    log(f"Results: {summary_file}")

# ==================== 主入口 ====================

def main():
    if len(sys.argv) > 1:
        datasets = [ds for ds in sys.argv[1:] if ds in DATASETS_CONFIG]
        if not datasets:
            log(f"Invalid datasets: {sys.argv[1:]}")
            log(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)
    else:
        datasets = list(DATASETS_CONFIG.keys())

    log("=" * 60)
    log("ACORN Fixed-EQ Experiment")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log("")

    # 检查可执行文件
    if not os.path.exists(f"{ACORN_DIR}/build/demos/build_acorn_index"):
        log(f"Error: ACORN not compiled")
        log("Please compile: cd ACORN && mkdir build && cd build && cmake .. && make")
        sys.exit(1)

    start_time = time.time()

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        try:
            run_dataset(dataset, config, OUTPUT_BASE)
        except KeyboardInterrupt:
            log("User interrupted")
            break
        except Exception as e:
            log(f"Dataset {dataset} error: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    log("")
    log("=" * 60)
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log("=" * 60)

if __name__ == "__main__":
    main()
