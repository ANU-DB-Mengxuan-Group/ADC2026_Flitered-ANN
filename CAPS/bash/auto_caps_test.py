#!/usr/bin/env python3
"""
CAPS 自动测试脚本 - 小规模测试版
先用 arxiv 数据集测试，验证流程是否正常

用法:
  cd ~/benchmarks/discrete/CAPS
  python bash/auto_caps_test.py

前置条件:
  1. 已编译 CAPS: make index && make query
  2. 数据集已准备好
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 配置 ====================

# 测试用小参数
TEST_NB_LIST = [128, 256, 512]  # 只测试几个 nb 值

# arxiv 数据集配置
DATASET_CONFIG = {
    "name": "arxiv",
    "base_path": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_base.fvecs",
    "label_base": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/label_base.txt",
    "query_path": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_query_equal.fvecs",
    "query_label": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_query_equal.txt",
    "gt_path": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv/arxiv_gt_equal.txt",
}

# 输出目录
INDEX_BASE = "/home/remote/u7905817/benchmarks/discrete/CAPS/indices"
RESULT_BASE = "/home/remote/u7905817/benchmarks/discrete/CAPS/results"

# CAPS 可执行文件路径
CAPS_DIR = "/home/remote/u7905817/benchmarks/discrete/CAPS"

# ==================== 辅助函数 ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_command(cmd, log_file=None, timeout=3600):
    """运行命令"""
    log(f"Running: {' '.join(cmd)}")
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                        timeout=timeout, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout, check=True)
        return True, None
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

# ==================== 主流程 ====================

def build_index(dataset, nb, config):
    """构建 CAPS 索引"""
    index_dir = f"{INDEX_BASE}/{dataset}/nb_{nb}"
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

    log(f"Building index: nb={nb}")
    start = time.time()
    success, error = run_command(cmd, log_file)
    build_time = time.time() - start

    if success:
        log(f"Build success: {build_time:.1f}s")
        return True, build_time
    else:
        log(f"Build failed: {error}")
        return False, 0

def search_index(dataset, nb, config):
    """搜索测试"""
    index_dir = f"{INDEX_BASE}/{dataset}/nb_{nb}"
    result_dir = f"{RESULT_BASE}/{dataset}"
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
        "1000"  # top_k
    ]

    log(f"Searching: nb={nb}")
    start = time.time()
    success, error = run_command(cmd, log_file)
    search_time = time.time() - start

    if success:
        log(f"Search success: {search_time:.1f}s")
        # 尝试读取结果
        if os.path.exists(result_csv):
            try:
                with open(result_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        log(f"Results saved to: {result_csv}")
            except:
                pass
        return True, search_time
    else:
        log(f"Search failed: {error}")
        return False, 0

def main():
    log("=" * 60)
    log("CAPS Test - arxiv dataset")
    log("=" * 60)

    dataset = DATASET_CONFIG["name"]
    config = DATASET_CONFIG

    # 检查可执行文件
    if not os.path.exists(f"{CAPS_DIR}/index"):
        log(f"Error: {CAPS_DIR}/index not found")
        log("Please compile CAPS first: cd CAPS && make index && make query")
        sys.exit(1)

    if not os.path.exists(f"{CAPS_DIR}/query"):
        log(f"Error: {CAPS_DIR}/query not found")
        log("Please compile CAPS first: cd CAPS && make index && make query")
        sys.exit(1)

    log(f"Testing nb values: {TEST_NB_LIST}")
    log("")

    results = []

    for nb in TEST_NB_LIST:
        log("-" * 40)
        log(f"Testing nb={nb}")
        log("-" * 40)

        # 构建索引
        build_ok, build_time = build_index(dataset, nb, config)
        if not build_ok:
            log(f"Skip search due to build failure")
            continue

        # 搜索测试
        search_ok, search_time = search_index(dataset, nb, config)

        results.append({
            "nb": nb,
            "build_ok": build_ok,
            "build_time": build_time,
            "search_ok": search_ok,
            "search_time": search_time
        })

        log("")

    # 打印汇总
    log("=" * 60)
    log("Summary")
    log("=" * 60)
    log(f"{'nb':>6} | {'Build':>8} | {'Search':>8} | {'Total':>8}")
    log("-" * 40)
    for r in results:
        status = "OK" if r["build_ok"] and r["search_ok"] else "FAIL"
        total = r["build_time"] + r["search_time"]
        log(f"{r['nb']:>6} | {r['build_time']:>7.1f}s | {r['search_time']:>7.1f}s | {total:>7.1f}s [{status}]")

    log("")
    log("Test complete!")
    log(f"Results in: {RESULT_BASE}/{dataset}/")

if __name__ == "__main__":
    main()
