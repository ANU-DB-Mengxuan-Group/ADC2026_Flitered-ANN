#!/usr/bin/env python3
"""
汇总 Fixed-EQ 实验结果

将各算法的有效结果整理到统一的目录结构
"""

import os
import csv
import shutil
from pathlib import Path
from datetime import datetime

# 配置
if os.path.exists('/home/remote/u7905817'):
    BASE_DIR = '/home/remote/u7905817/benchmarks/discrete'
else:
    BASE_DIR = '/Users/a1-6/workspace/discrete-filteredANN-benchmark'

OUTPUT_DIR = f"{BASE_DIR}/analysis/fixed_eq_results"
DATASETS = ['arxiv', 'yfcc']

# 各算法的结果路径和过滤条件
ALGORITHM_CONFIG = {
    'NHQ': {
        'result_path': f"{BASE_DIR}/NHQ/data_synthetic/results/{{dataset}}/summary.csv",
        'filter': lambda row: True,  # NHQ结果是干净的
        'recall_col': 'Recall@10',
        'qps_col': 'QPS',
    },
    'CAPS': {
        'result_path': f"{BASE_DIR}/CAPS/data/results/{{dataset}}/summary.csv",
        'filter': lambda row: row.get('status') == 'success' and float(row.get('recall@10', 0)) > 0.5,
        'recall_col': 'recall@10',
        'qps_col': 'qps',
    },
    'ACORN': {
        'result_path': f"{BASE_DIR}/ACORN/data_fixed_eq/results/{{dataset}}/summary.csv",
        'filter': lambda row: row.get('status') == 'success' and float(row.get('recall@10', 0)) > 0.5,
        'recall_col': 'recall@10',
        'qps_col': 'qps',
    },
}

def load_and_filter_results(algo, dataset):
    """加载并过滤算法结果"""
    config = ALGORITHM_CONFIG[algo]
    path = config['result_path'].format(dataset=dataset)

    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return []

    results = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if config['filter'](row):
                results.append(row)

    return results

def extract_best_result(results, config):
    """提取最佳结果（最高recall）"""
    if not results:
        return None

    recall_col = config['recall_col']
    qps_col = config['qps_col']

    best = max(results, key=lambda r: float(r.get(recall_col, 0)))
    return {
        'recall': float(best.get(recall_col, 0)),
        'qps': float(best.get(qps_col, 0)),
        'params': {k: v for k, v in best.items() if k not in [recall_col, qps_col, 'status', 'timestamp']},
    }

def collect_results():
    """汇总所有结果"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    summary = []

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        for algo, config in ALGORITHM_CONFIG.items():
            results = load_and_filter_results(algo, dataset)
            print(f"  {algo}: {len(results)} valid results")

            if results:
                best = extract_best_result(results, config)
                if best:
                    summary.append({
                        'dataset': dataset,
                        'algorithm': algo,
                        'recall@10': best['recall'],
                        'qps': best['qps'],
                    })
                    print(f"    Best: Recall={best['recall']:.4f}, QPS={best['qps']:.2f}")

    # 保存汇总
    summary_file = f"{OUTPUT_DIR}/summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'algorithm', 'recall@10', 'qps'])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n汇总保存到: {summary_file}")

    # 保存完整的清洗后结果
    for dataset in DATASETS:
        for algo, config in ALGORITHM_CONFIG.items():
            results = load_and_filter_results(algo, dataset)
            if results:
                out_path = f"{OUTPUT_DIR}/{dataset}/{algo}.csv"
                Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"  Saved: {out_path}")

def main():
    print("=" * 60)
    print("Fixed-EQ Results Collection")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    collect_results()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == '__main__':
    main()
