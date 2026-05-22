#!/usr/bin/env python3
"""
CAPS 自动参数搜索 - 全数据集版本
支持所有 6 个数据集，带断点续传

用法:
  cd ~/benchmarks/discrete/CAPS
  python bash/auto_caps_all.py                    # 运行所有数据集
  python bash/auto_caps_all.py arxiv              # 只运行指定数据集
  python bash/auto_caps_all.py arxiv yfcc         # 运行多个数据集

建议用 screen 运行:
  screen -S caps_search
  python -u bash/auto_caps_all.py 2>&1 | tee caps_search.log
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

# nb 参数范围（扩大范围以覆盖更多 recall-qps trade-off）
NB_LIST = [64, 128, 256, 512, 1024, 2048, 4096]

# 数据集配置
# 注意：CAPS只支持Fixed-EQ场景，论文Figure 10只测试arxiv和yfcc
# CAPS专用标签文件（逗号分隔）：CAPS/synthetic_labels/{dataset}/label_CAPS_*.txt
CAPS_DIR = "/home/remote/u7905817/benchmarks/discrete/CAPS"
CAPS_LABELS = f"{CAPS_DIR}/synthetic_labels"

DATASETS_CONFIG = {
    "arxiv": {
        "N": 132687,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_base.fvecs",
        # CAPS专用标签：逗号分隔格式
        "label_base": f"{CAPS_LABELS}/arxiv/label_CAPS_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_query_equal.fvecs",
        "query_label": f"{CAPS_LABELS}/arxiv/label_CAPS_query.txt",
        "gt_path": f"{CAPS_LABELS}/arxiv/gt_CAPS.txt",
    },
    "yfcc": {
        "N": 1000000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc/yfcc_base.fvecs",
        # CAPS专用标签：逗号分隔格式
        "label_base": f"{CAPS_LABELS}/yfcc/label_CAPS_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc/yfcc_query_equal.fvecs",
        "query_label": f"{CAPS_LABELS}/yfcc/label_CAPS_query.txt",
        "gt_path": f"{CAPS_LABELS}/yfcc/gt_CAPS.txt",
    },
    # 以下数据集没有合成Fixed-EQ标签，暂时保留配置但不推荐使用
    # 论文Figure 10只测试arxiv和yfcc
    "LAION1M": {
        "N": 1000448,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/LAION1M_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/label_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/LAION1M_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/LAION1M_query_equal.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M/LAION1M_gt_equal.txt",
    },
    "tripclick": {
        "N": 1055976,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/tripclick_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/label_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/tripclick_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/tripclick_query_equal.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick/tripclick_gt_equal.txt",
    },
    "ytb_audio": {
        "N": 5000000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/label_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_query_equal.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio/ytb_audio_gt_equal.txt",
    },
    "ytb_video": {
        "N": 1000000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/label_base.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_query_equal.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video/ytb_video_gt_equal.txt",
    },
    # --- V2 validation datasets ---
    # V2 labels are 0-based; use 1-based to avoid sentinel issues
    "synth_192d": {
        "N": 800000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d/synth_192d_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d/label_base_1based.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d/synth_192d_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d/synth_192d_query_equal_1based.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d/synth_192d_gt_equal.txt",
    },
    "synth_512d": {
        "N": 800000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d/synth_512d_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d/label_base_1based.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d/synth_512d_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d/synth_512d_query_equal_1based.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d/synth_512d_gt_equal.txt",
    },
    "synth_768d_hc": {
        "N": 800000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc/synth_768d_hc_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc/label_base_1based.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc/synth_768d_hc_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc/synth_768d_hc_query_equal_1based.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc/synth_768d_hc_gt_equal.txt",
    },
    "yahoo800k": {
        "N": 800000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k/yahoo800k_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k/label_base_1based.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k/yahoo800k_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k/yahoo800k_query_equal_1based.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k/yahoo800k_gt_equal.txt",
    },
    "dbpedia560k": {
        "N": 560000,
        "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k/dbpedia560k_base.fvecs",
        "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k/label_base_1based.txt",
        "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k/dbpedia560k_query_equal.fvecs",
        "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k/dbpedia560k_query_equal_1based.txt",
        "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k/dbpedia560k_gt_equal.txt",
    },
}

# 路径配置
OUTPUT_BASE = f"{CAPS_DIR}/data"

# ==================== 辅助函数 ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_progress(progress_file):
    """加载进度"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"completed": [], "failed": []}

def save_progress(progress_file, progress):
    """保存进度"""
    Path(os.path.dirname(progress_file)).mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def init_summary_file(summary_file):
    """初始化汇总文件"""
    if not os.path.exists(summary_file):
        Path(os.path.dirname(summary_file)).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'nb', 'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, nb, metrics):
    """追加结果到汇总文件"""
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, nb,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_dir):
    """获取索引目录大小（MB）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(index_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    except:
        return 0.0

def parse_result_csv(csv_file):
    """解析搜索结果 CSV"""
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

def build_index(dataset, nb, config, output_base):
    """构建索引"""
    index_dir = f"{output_base}/indices/{dataset}/nb_{nb}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{index_dir}/build.log"

    cmd = [
        f"{CAPS_DIR}/index",
        config["base_path"],
        config["label_base"],
        index_dir,
        str(nb),
        "kmeans",
        "1"
    ]

    log(f"Building: dataset={dataset}, nb={nb}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=7200, check=True)  # 2小时超时
        build_time = time.time() - start
        index_size = get_index_size(index_dir)
        log(f"Build success: {build_time:.1f}s, {index_size:.1f}MB")
        return True, build_time, index_size
    except subprocess.TimeoutExpired:
        log(f"Build timeout (2h)")
        return False, 0, 0
    except Exception as e:
        log(f"Build failed: {e}")
        return False, 0, 0

def search_index(dataset, nb, config, output_base):
    """搜索测试"""
    index_dir = f"{output_base}/indices/{dataset}/nb_{nb}"
    result_dir = f"{output_base}/results/{dataset}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    result_csv = f"{result_dir}/nb_{nb}_results.csv"
    log_file = f"{result_dir}/nb_{nb}_search.log"

    cmd = [
        f"{CAPS_DIR}/query",
        config["base_path"],
        config["label_base"],
        config["query_path"],
        config["query_label"],
        index_dir,
        config["gt_path"],
        result_csv,
        str(nb),
        "kmeans",
        "1",
        "1000"
    ]

    log(f"Searching: dataset={dataset}, nb={nb}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=3600, check=True)  # 1小时超时
        search_time = time.time() - start

        # 解析结果
        metrics = parse_result_csv(result_csv)
        if metrics:
            log(f"Search success: {search_time:.1f}s, Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")
            return True, search_time, metrics
        else:
            log(f"Search done but no results: {search_time:.1f}s")
            return True, search_time, {'recall': 0, 'qps': 0}

    except subprocess.TimeoutExpired:
        log(f"Search timeout (1h)")
        return False, 0, None
    except Exception as e:
        log(f"Search failed: {e}")
        return False, 0, None

def run_dataset(dataset, config, output_base):
    """运行单个数据集"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()}")
    log(f"N = {config['N']:,}")
    log(f"nb values: {NB_LIST}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_{dataset}.json"
    summary_file = f"{output_base}/results/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for nb in NB_LIST:
        task_key = f"{dataset}_nb_{nb}"

        # 跳过已完成
        if task_key in progress.get("completed", []):
            log(f"Skip (done): nb={nb}")
            continue

        # 跳过已失败
        if task_key in progress.get("failed", []):
            log(f"Skip (failed): nb={nb}")
            continue

        log("")
        log(f"--- nb={nb} ---")

        # 构建
        build_ok, build_time, index_size = build_index(dataset, nb, config, output_base)
        if not build_ok:
            progress.setdefault("failed", []).append(task_key)
            save_progress(progress_file, progress)
            append_to_summary(summary_file, dataset, nb, {
                'build_time': 0, 'index_size': 0,
                'recall': 0, 'qps': 0, 'status': 'build_failed'
            })
            continue

        # 搜索
        search_ok, search_time, metrics = search_index(dataset, nb, config, output_base)
        if not search_ok:
            progress.setdefault("failed", []).append(task_key)
            save_progress(progress_file, progress)
            append_to_summary(summary_file, dataset, nb, {
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
        append_to_summary(summary_file, dataset, nb, metrics)

    log("")
    log(f"Dataset {dataset} done!")
    log(f"Results: {summary_file}")

# ==================== 主入口 ====================

def main():
    # 解析参数
    if len(sys.argv) > 1:
        datasets = [ds for ds in sys.argv[1:] if ds in DATASETS_CONFIG]
        if not datasets:
            log(f"Invalid datasets: {sys.argv[1:]}")
            log(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)
    else:
        datasets = list(DATASETS_CONFIG.keys())

    log("=" * 60)
    log("CAPS Parameter Search")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log(f"nb values: {NB_LIST}")
    log("")

    # 检查可执行文件
    if not os.path.exists(f"{CAPS_DIR}/index"):
        log(f"Error: {CAPS_DIR}/index not found")
        log("Please compile: cd CAPS && make index && make query")
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
