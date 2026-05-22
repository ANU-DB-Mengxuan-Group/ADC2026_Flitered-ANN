#!/usr/bin/env python3
"""
UNG Fixed-EQ 实验脚本
使用合成标签（4属性×3值）进行Fixed-Length Equality实验

用法:
  cd ~/benchmarks/discrete/UNG-dev
  python bash/auto_ung_fixed_eq.py                # 运行所有数据集
  python bash/auto_ung_fixed_eq.py arxiv          # 只运行指定数据集
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

# UNG目录（源码+数据都在这里）
UNG_DIR = "/home/remote/u7905817/benchmarks/discrete/UNG-dev"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"

# 参数范围
# UNG主要参数: max_degree, Lbuild, Lsearch, num_cross_edges
MAX_DEGREES = [32, 48, 64, 96]
L_BUILDS = [100, 150, 200]
L_SEARCHES = [100, 200, 300, 500]
NUM_CROSS_EDGES = 6  # 默认值
T = 16  # 线程数

DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "yfcc": {"N": 1000000, "D": 192},
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
                'dataset', 'max_degree', 'Lbuild', 'Lsearch',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, max_degree, Lbuild, Lsearch, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, max_degree, Lbuild, Lsearch,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_prefix):
    """计算索引总大小（UNG生成多个文件）"""
    import glob
    total_size = 0
    for f in glob.glob(f"{index_prefix}*"):
        if os.path.isfile(f):
            total_size += os.path.getsize(f)
    return total_size / (1024 * 1024)

def parse_ung_result(result_prefix):
    """解析UNG结果CSV文件

    UNG输出格式: {result_prefix}result.csv
    内容: L,Cmps,QPS,Recall
    """
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

# ==================== 主流程 ====================

def build_index(dataset, max_degree, Lbuild, config, output_base):
    """构建索引"""
    index_dir = f"{output_base}/indices/{dataset}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{index_dir}/M={max_degree}_L={Lbuild}_build.log"
    index_prefix = f"{index_dir}/index_M={max_degree}_L={Lbuild}"

    # UNG build命令
    cmd = [
        f"{UNG_DIR}/build/apps/build_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{output_base}/data_bin/{dataset}/{dataset}_base.bin",
        "--base_label_file", f"{output_base}/synthetic_labels/{dataset}/label_base.txt",
        "--index_path_prefix", index_prefix,
        "--scenario", "equality",
        "--max_degree", str(max_degree),
        "--Lbuild", str(Lbuild),
        "--num_cross_edges", str(NUM_CROSS_EDGES),
        "--num_threads", str(T),
    ]

    log(f"Building: max_degree={max_degree}, Lbuild={Lbuild}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=7200, check=True)
        build_time = time.time() - start
        # 计算所有索引文件的总大小
        index_size = get_index_size(index_prefix)
        log(f"  Build success: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size, index_prefix
    except subprocess.TimeoutExpired:
        log(f"  Build timeout (2h)")
        return False, 0, 0, None
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, 0, 0, None

def search_index(dataset, index_prefix, Lsearch, config, output_base):
    """搜索测试"""
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # 从index_prefix提取参数用于命名
    idx_name = os.path.basename(index_prefix)
    log_file = f"{result_dir}/{idx_name}_Ls={Lsearch}_search.log"
    result_prefix = f"{result_dir}/{idx_name}_Ls={Lsearch}"

    # UNG search命令
    cmd = [
        f"{UNG_DIR}/build/apps/search_UNG_index",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--base_bin_file", f"{output_base}/data_bin/{dataset}/{dataset}_base.bin",
        "--query_bin_file", f"{output_base}/data_bin/{dataset}/{dataset}_query.bin",
        "--base_label_file", f"{output_base}/synthetic_labels/{dataset}/label_base.txt",
        "--query_label_file", f"{output_base}/synthetic_labels/{dataset}/label_query.txt",
        "--gt_file", f"{output_base}/synthetic_labels/{dataset}/gt.bin",
        "--K", str(K),
        "--index_path_prefix", index_prefix,
        "--scenario", "equality",
        "--Lsearch", str(Lsearch),
        "--num_threads", str(T),
        "--result_path_prefix", result_prefix,
    ]

    log(f"Searching: Lsearch={Lsearch}")

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
    """删除索引文件和目录节省空间"""
    import glob
    import shutil
    for f in glob.glob(f"{index_prefix}*"):
        try:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
            log(f"  Cleaned: {os.path.basename(f)}")
        except Exception as e:
            log(f"  Cleanup failed: {e}")

def run_dataset(dataset, config, output_base):
    """运行单个数据集"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()} (Fixed-EQ)")
    log(f"N = {config['N']:,}, D = {config['D']}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_{dataset}.json"
    summary_file = f"{output_base}/results/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for max_degree in MAX_DEGREES:
        for Lbuild in L_BUILDS:
            # 构建索引
            task_key_build = f"{dataset}_M={max_degree}_Lb={Lbuild}_build"

            index_prefix = None
            build_time = 0
            index_size = 0

            if task_key_build not in progress.get("completed", []):
                build_ok, build_time, index_size, index_prefix = build_index(
                    dataset, max_degree, Lbuild, config, output_base)

                if not build_ok:
                    progress.setdefault("failed", []).append(task_key_build)
                    save_progress(progress_file, progress)
                    # 构建失败，跳过所有该配置的搜索
                    for Lsearch in L_SEARCHES:
                        append_to_summary(summary_file, dataset, max_degree, Lbuild, Lsearch, {
                            'build_time': 0, 'index_size': 0,
                            'recall': 0, 'qps': 0, 'status': 'build_failed'
                        })
                    continue

                progress.setdefault("completed", []).append(task_key_build)
                save_progress(progress_file, progress)
            else:
                log(f"Skip build (done): M={max_degree}, Lb={Lbuild}")
                # 重建index_prefix用于搜索
                index_dir = f"{output_base}/indices/{dataset}"
                index_prefix = f"{index_dir}/index_M={max_degree}_L={Lbuild}"

            # 对该配置测试所有Lsearch值
            for Lsearch in L_SEARCHES:
                task_key = f"{dataset}_M={max_degree}_Lb={Lbuild}_Ls={Lsearch}"

                if task_key in progress.get("completed", []):
                    log(f"Skip (done): {task_key}")
                    continue

                if task_key in progress.get("failed", []):
                    log(f"Skip (failed): {task_key}")
                    continue

                log("")
                log(f"--- {task_key} ---")

                # 搜索
                search_ok, metrics = search_index(
                    dataset, index_prefix, Lsearch, config, output_base)

                if not search_ok:
                    progress.setdefault("failed", []).append(task_key)
                    save_progress(progress_file, progress)
                    append_to_summary(summary_file, dataset, max_degree, Lbuild, Lsearch, {
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
                append_to_summary(summary_file, dataset, max_degree, Lbuild, Lsearch, metrics)

            # 所有Lsearch测试完成后清理该配置的索引
            if index_prefix:
                cleanup_index(index_prefix)

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
    log("UNG Fixed-EQ Experiment")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log("")

    # 检查可执行文件
    build_exe = f"{UNG_DIR}/build/apps/build_UNG_index"
    if not os.path.exists(build_exe):
        log(f"Error: UNG not compiled at {build_exe}")
        log("Please compile UNG:")
        log("  cd UNG-dev")
        log("  mkdir build && cd build")
        log("  cmake -DCMAKE_BUILD_TYPE=Release ../codes/")
        log("  make -j")
        sys.exit(1)

    # 检查数据文件
    for dataset in datasets:
        bin_file = f"{UNG_DIR}/data_bin/{dataset}/{dataset}_base.bin"
        if not os.path.exists(bin_file):
            log(f"Error: Binary data not found: {bin_file}")
            log("Please run: python convert_fvecs_to_bin.py")
            sys.exit(1)

        label_file = f"{UNG_DIR}/synthetic_labels/{dataset}/label_base.txt"
        if not os.path.exists(label_file):
            log(f"Error: Labels not found: {label_file}")
            log("Please run: python generate_ung_synthetic_labels.py")
            sys.exit(1)

        gt_file = f"{UNG_DIR}/synthetic_labels/{dataset}/gt.bin"
        if not os.path.exists(gt_file):
            log(f"Error: Groundtruth not found: {gt_file}")
            log("Please run: python generate_ung_groundtruth.py")
            sys.exit(1)

    start_time = time.time()

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        try:
            run_dataset(dataset, config, UNG_DIR)
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
