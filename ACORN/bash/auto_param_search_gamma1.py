#!/usr/bin/env python3
"""
ACORN参数自动搜索 - gamma=1 模式
这是 auto_param_search_v2.py 的变体，固定 gamma=1

用法:
  python auto_param_search_gamma1.py                    # 运行所有数据集
  python auto_param_search_gamma1.py yfcc               # 只运行指定数据集
  python auto_param_search_gamma1.py yfcc LAION1M       # 运行多个数据集

建议用screen或nohup运行:
  screen -S acorn_gamma1
  python -u auto_param_search_gamma1.py 2>&1 | tee gamma1_search.log
  # 断开: Ctrl+A, D
  # 重连: screen -r acorn_gamma1
"""

import os
import sys
import time
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 数据集配置 ====================

# gamma=1 模式的参数范围
GAMMA1_PARAMS = {
    "Ms": [32, 48, 64],
    "M_betas": [48, 64, 96],
    "gammas": [1],  # 固定 gamma=1
}

DATASETS_CONFIG = {
    "yfcc": {
        "N": 1000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
    "LAION1M": {
        "N": 1000448,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
    "tripclick": {
        "N": 1055976,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
    "ytb_audio": {
        "N": 5000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
    "ytb_video": {
        "N": 1000000,  # 1M向量（1024维），不是5M！
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
    "arxiv": {
        "N": 132000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv",
        "scenarios": ["equal", "or", "and"],
        **GAMMA1_PARAMS,
    },
}

# 全局常量
K = 10  # Recall@10
MIN_INDEX_SIZE_MB = 50  # 最小合法索引大小（用于检测损坏）
AUTO_CLEANUP = True  # 搜索完成后自动删除索引文件

# ==================== 辅助函数 ====================

def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_progress(progress_file):
    """加载进度"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                if 'progress' in data:
                    progress_data = data['progress']
                else:
                    progress_data = data
                return {tuple(json.loads(k)): v for k, v in progress_data.items()}
        except Exception as e:
            log(f"Warning: 加载进度文件失败: {e}")
    return {}

def save_progress(progress_file, progress):
    """保存进度"""
    output_dir = os.path.dirname(progress_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    serializable = {json.dumps(k): v for k, v in progress.items()}

    data = {
        'progress': serializable,
        'last_update': datetime.now().isoformat(),
        'total_tasks': len(progress),
        'completed_builds': sum(1 for v in progress.values() if v.get('build_done')),
        'completed_searches': sum(1 for v in progress.values() if v.get('search_done'))
    }

    for attempt in range(3):
        try:
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
            return
        except Exception as e:
            if attempt < 2:
                log(f"Warning: 保存进度失败（尝试 {attempt+1}/3）: {e}")
                time.sleep(1)
            else:
                log(f"Error: 保存进度失败（已尝试3次）: {e}")

def init_summary_file(summary_file):
    """初始化汇总CSV文件"""
    if not os.path.exists(summary_file):
        output_dir = os.path.dirname(summary_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'M', 'M_beta', 'gamma', 'scenario',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'qps_no_filter',
                'status', 'timestamp'
            ])

def append_to_summary(summary_file, M, M_beta, gamma, scenario, metrics):
    """追加结果到CSV"""
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            M, M_beta, gamma, scenario,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('qps_no_filter', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_path):
    """获取索引文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(index_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def parse_search_csv(csv_file):
    """从CSV文件提取最佳性能指标"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            best_recall = 0.0
            best_qps = 0.0
            best_qps_no_filter = 0.0

            for row in reader:
                recall = float(row['Recall'])
                if recall > best_recall:
                    best_recall = recall
                    best_qps = float(row['QPS'])
                    best_qps_no_filter = float(row['QPS_no_filter'])

            if best_recall > 0:
                return {
                    'recall': best_recall,
                    'qps': best_qps,
                    'qps_no_filter': best_qps_no_filter
                }
    except Exception as e:
        log(f"Warning: 解析CSV失败 {csv_file}: {e}")
    return None

def cleanup_index(index_path):
    """删除索引文件（节省空间）"""
    if AUTO_CLEANUP and os.path.exists(index_path):
        try:
            size_mb = get_index_size(index_path)
            os.remove(index_path)
            log(f"Cleaned up index file ({size_mb:.1f}MB)")
        except Exception as e:
            log(f"Warning: 删除索引失败: {e}")

# ==================== 主流程 ====================

def build_index(dataset, M, M_beta, gamma, config, output_base, progress, progress_file):
    """构建索引"""
    key = (M, M_beta, gamma, None)

    indices_base = f"{output_base}/indices"
    dataset_dir = f"{indices_base}/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)

    index_path = f"{dataset_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

    # 检查是否已完成
    if key in progress:
        if progress[key].get('build_done'):
            index_size = progress[key].get('index_size', 0)
            build_time = progress[key].get('build_time', 0)
            log(f"Skip build (done): M={M}, M_beta={M_beta}, gamma={gamma}, {index_size:.1f}MB")
            return True, build_time, index_size
        elif progress[key].get('build_done') == False:
            failure_reason = progress[key].get('failure_reason', '')
            if failure_reason in ['timeout', 'error']:
                log(f"Skip build (failed: {failure_reason}): M={M}, M_beta={M_beta}, gamma={gamma}")
                return False, 0, 0

    log(f"Building index: M={M}, M_beta={M_beta}, gamma={gamma}")

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma),
        f"{data_dir}/{dataset}_base.fvecs",
        str(M), str(M_beta),
        indices_base, dataset
    ]

    log_file = f"{dataset_dir}/M={M}_Mb={M_beta}_gamma={gamma}_build.log"

    start = time.time()
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=10800)  # 3小时

        build_time = time.time() - start
        index_size = get_index_size(index_path)

        if index_size < MIN_INDEX_SIZE_MB:
            log(f"Build failed: index too small ({index_size:.1f}MB < {MIN_INDEX_SIZE_MB}MB)")
            return False, 0, 0

        log(f"Build success: {build_time:.1f}s, {index_size:.1f}MB")

        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = True
        progress[key]['build_time'] = build_time
        progress[key]['index_size'] = index_size
        save_progress(progress_file, progress)

        return True, build_time, index_size

    except subprocess.TimeoutExpired:
        log(f"Build timeout (3h)")
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
            except:
                pass
        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = False
        progress[key]['failure_reason'] = 'timeout'
        save_progress(progress_file, progress)
        return False, 0, 0
    except Exception as e:
        log(f"Build failed: {e}")
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
            except:
                pass
        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = False
        progress[key]['failure_reason'] = 'error'
        save_progress(progress_file, progress)
        return False, 0, 0

def search_index(dataset, M, M_beta, gamma, scenario, config, output_base, progress, progress_file, summary_file):
    """搜索测试"""
    key = (M, M_beta, gamma, scenario)

    if key in progress:
        if progress[key].get('search_done'):
            recall = progress[key].get('recall', 0)
            log(f"Skip search (done): scenario={scenario}, Recall={recall:.4f}")
            return True
        elif progress[key].get('search_done') == False:
            failure_reason = progress[key].get('failure_reason', '')
            if failure_reason in ['timeout', 'error', 'parse_error']:
                log(f"Skip search (failed: {failure_reason}): scenario={scenario}")
                return False

    indices_base = f"{output_base}/indices"
    dataset_dir = f"{indices_base}/{dataset}"
    index_path = f"{dataset_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

    if not os.path.exists(index_path):
        log(f"Warning: Index file not found, need rebuild")
        return False

    log(f"Searching: scenario={scenario}")

    results_dir = f"{output_base}/results/{dataset}/{scenario}"
    os.makedirs(results_dir, exist_ok=True)

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), dataset,
        str(M), str(M_beta), indices_base, scenario, results_dir,
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt",
        f"{data_dir}/{dataset}_query_{scenario}.fvecs",
        f"{data_dir}/{dataset}_query_{scenario}.txt",
        f"{data_dir}/{dataset}_gt_{scenario}.txt",
        str(K)
    ]

    log_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_search.log"

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1800)

        csv_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
        metrics = parse_search_csv(csv_file)

        if metrics:
            log(f"Search success: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")

            build_key = (M, M_beta, gamma, None)
            if build_key in progress:
                metrics['build_time'] = progress[build_key].get('build_time', 0)
                metrics['index_size'] = progress[build_key].get('index_size', 0)

            metrics['status'] = 'success'
            append_to_summary(summary_file, M, M_beta, gamma, scenario, metrics)

            if key not in progress:
                progress[key] = {}
            progress[key]['search_done'] = True
            progress[key]['recall'] = metrics['recall']
            progress[key]['qps'] = metrics['qps']
            save_progress(progress_file, progress)

            return True
        else:
            log(f"Search failed: cannot parse result")
            if key not in progress:
                progress[key] = {}
            progress[key]['search_done'] = False
            progress[key]['failure_reason'] = 'parse_error'
            save_progress(progress_file, progress)
            return False

    except Exception as e:
        log(f"Search failed: {e}")
        if key not in progress:
            progress[key] = {}
        progress[key]['search_done'] = False
        progress[key]['failure_reason'] = 'error'
        save_progress(progress_file, progress)
        return False

def cleanup_orphaned_indices(indices_dir):
    """启动时清理残留的索引文件（防止之前崩溃留下的文件占用空间）"""
    if not os.path.exists(indices_dir):
        return

    cleaned = 0
    cleaned_size = 0
    for filename in os.listdir(indices_dir):
        if filename.endswith('.json') and filename.startswith('hybrid_'):
            filepath = os.path.join(indices_dir, filename)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                os.remove(filepath)
                cleaned += 1
                cleaned_size += size_mb
                log(f"🗑️  清理残留索引: {filename} ({size_mb:.1f}MB)")
            except Exception as e:
                log(f"⚠️  清理失败: {filename}: {e}")

    if cleaned > 0:
        log(f"✅ 共清理 {cleaned} 个残留索引，释放 {cleaned_size:.1f}MB")

def run_dataset(dataset, config, output_base):
    """运行单个数据集的完整参数搜索"""

    # 启动时清理残留索引文件
    indices_dir = f"{output_base}/indices/{dataset}"
    cleanup_orphaned_indices(indices_dir)

    log("=" * 70)
    log(f"Dataset: {dataset.upper()} (gamma=1 mode)")
    log(f"N={config['N']:,}")
    log(f"Scenarios: {', '.join(config['scenarios'])}")
    log(f"Params: M={config['Ms']}, M_beta={config['M_betas']}, gamma={config['gammas']}")
    log("=" * 70)

    progress_file = f"{output_base}/progress.json"
    summary_file = f"{output_base}/results/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    Ms = config['Ms']
    M_betas = config['M_betas']
    gammas = config['gammas']
    scenarios = config['scenarios']

    total_tasks = len(Ms) * len(M_betas) * len(gammas) * len(scenarios)
    log(f"Total tasks: {total_tasks}")

    completed_builds = 0
    completed_searches = 0

    for M in Ms:
        for M_beta in M_betas:
            if M_beta < M:
                continue

            for gamma in gammas:
                if M_beta > 2 * M * gamma:
                    continue

                log("")
                log(f"=== M={M}, M_beta={M_beta}, gamma={gamma} ===")

                build_success, build_time, index_size = build_index(
                    dataset, M, M_beta, gamma, config, output_base, progress, progress_file
                )

                if build_success:
                    completed_builds += 1
                    index_path = f"{output_base}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

                    for scenario in scenarios:
                        search_success = search_index(
                            dataset, M, M_beta, gamma, scenario, config,
                            output_base, progress, progress_file, summary_file
                        )
                        if search_success:
                            completed_searches += 1

                    cleanup_index(index_path)
                else:
                    log(f"Build failed, skip all searches")

    log("")
    log("=" * 70)
    log(f"Dataset {dataset.upper()} done!")
    log(f"Builds: {completed_builds}, Searches: {completed_searches}/{total_tasks}")
    log(f"Results: {summary_file}")
    log("=" * 70)

# ==================== 主入口 ====================

def main():
    if len(sys.argv) > 1:
        datasets_to_run = [ds for ds in sys.argv[1:] if ds in DATASETS_CONFIG]
        if not datasets_to_run:
            log(f"Invalid datasets: {sys.argv[1:]}")
            log(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)
    else:
        datasets_to_run = list(DATASETS_CONFIG.keys())

    log("=" * 70)
    log("ACORN Parameter Search - gamma=1 mode")
    log("=" * 70)
    log(f"Datasets: {', '.join(datasets_to_run)}")
    log("")

    start_time = time.time()

    for dataset in datasets_to_run:
        config = DATASETS_CONFIG[dataset]
        # 输出到单独的目录，避免覆盖原有数据
        output_base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_{dataset}_gamma1"

        try:
            run_dataset(dataset, config, output_base)
        except KeyboardInterrupt:
            log("User interrupted, saving progress...")
            break
        except Exception as e:
            log(f"Dataset {dataset} failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    log("=" * 70)
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log("=" * 70)

if __name__ == "__main__":
    main()
