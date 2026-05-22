#!/usr/bin/env python3
"""
快速测试所有数据集 - 验证参数范围是否合适

策略：
1. 每个数据集只测试3-4个代表性参数组合
2. 快速发现哪些参数会失败
3. 估算构建时间和Recall范围
4. 为完整搜索提供参数调整建议

预计总时间：1-2小时（全部数据集）
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

# ==================== 测试参数配置 ====================

# 代表性参数组合（基于arxiv经验）
TEST_PARAMS = [
    # (M, M_beta, gamma) - 覆盖小中大参数
    (32, 48, 4),    # 最小参数：快速，低Recall
    (48, 64, 8),    # 中等参数：平衡，arxiv测试显示Recall~0.83
    (64, 96, 12),   # 较大参数：高Recall，较慢
    # (64, 128, 8), # 高M_beta - 先不测试，可能在某些数据集失败
]

DATASETS_CONFIG = {
    "yfcc": {
        "N": 1000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc",
        "scenarios": ["equal"],  # 只测试一个场景
    },
    "LAION1M": {
        "N": 1000448,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M",
        "scenarios": ["and"],
    },
    "tripclick": {
        "N": 1055976,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick",
        "scenarios": ["equal"],
    },
    "ytb_video": {
        "N": 1000000,  # 实际是1M向量，不是5M
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video",
        "scenarios": ["equal"],
    },
}

K = 10

# ==================== 辅助函数 ====================

def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def check_files(dataset, config):
    """检查数据文件"""
    data_dir = config['data_dir']
    required = [
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt",
    ]
    for scenario in config['scenarios']:
        required.extend([
            f"{data_dir}/{dataset}_query_{scenario}.fvecs",
            f"{data_dir}/{dataset}_query_{scenario}.txt",
            f"{data_dir}/{dataset}_gt_{scenario}.txt",
        ])

    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        log(f"❌ 缺少文件:")
        for f in missing:
            log(f"   {f}")
        return False
    return True

def build_index(dataset, M, M_beta, gamma, config, output_dir):
    """构建索引"""
    # C++程序会在output_dir下创建{dataset}子目录，需要确保父目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 同时创建dataset子目录，避免"No such file or directory"错误
    os.makedirs(f"{output_dir}/{dataset}", exist_ok=True)

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma), f"{data_dir}/{dataset}_base.fvecs",
        str(M), str(M_beta), output_dir, dataset
    ]

    log_file = f"{output_dir}/build.log"

    start = time.time()
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        build_time = time.time() - start

        # C++程序生成的索引文件在{output_dir}/{dataset}/子目录下
        index_path = f"{output_dir}/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        index_size = os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0

        return True, build_time, index_size, None
    except subprocess.CalledProcessError as e:
        return False, 0, 0, str(e)
    except subprocess.TimeoutExpired:
        return False, 0, 0, "Timeout (1 hour)"
    except Exception as e:
        return False, 0, 0, str(e)

def search_index(dataset, M, M_beta, gamma, scenario, config, index_dir, output_dir):
    """搜索测试"""
    os.makedirs(output_dir, exist_ok=True)

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), dataset,
        str(M), str(M_beta), index_dir, scenario, output_dir,
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt",
        f"{data_dir}/{dataset}_query_{scenario}.fvecs",
        f"{data_dir}/{dataset}_query_{scenario}.txt",
        f"{data_dir}/{dataset}_gt_{scenario}.txt",
        str(K)
    ]

    log_file = f"{output_dir}/search.log"

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1800)

        # 解析CSV结果
        csv_file = f"{output_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
        if os.path.exists(csv_file):
            import csv as csv_module
            with open(csv_file, 'r') as f:
                reader = csv_module.DictReader(f)
                best_recall = 0.0
                best_qps = 0.0
                for row in reader:
                    recall = float(row['Recall'])
                    if recall > best_recall:
                        best_recall = recall
                        best_qps = float(row['QPS'])
                return True, best_recall, best_qps, None

        return False, 0, 0, "No CSV output"
    except subprocess.CalledProcessError as e:
        return False, 0, 0, str(e)
    except Exception as e:
        return False, 0, 0, str(e)

# ==================== 主测试流程 ====================

def test_dataset(dataset):
    """测试单个数据集"""

    if dataset not in DATASETS_CONFIG:
        log(f"❌ 未知数据集: {dataset}")
        return None

    config = DATASETS_CONFIG[dataset]

    log("="*70)
    log(f"快速测试 - {dataset.upper()}")
    log(f"数据规模: N={config['N']:,}")
    log(f"测试参数: {len(TEST_PARAMS)}组 × {len(config['scenarios'])}场景")
    log("="*70)

    # 检查文件
    if not check_files(dataset, config):
        return None

    output_base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/quick_test_{dataset}"

    results = []

    for M, M_beta, gamma in TEST_PARAMS:
        log(f"\n{'='*70}")
        log(f"测试参数: M={M}, M_beta={M_beta}, gamma={gamma}")
        log(f"{'='*70}")

        # 构建索引
        index_dir = f"{output_base}/indices"
        log("🔨 构建索引...")
        success, build_time, index_size, error = build_index(dataset, M, M_beta, gamma, config, index_dir)

        if not success:
            log(f"   ❌ 构建失败: {error}")
            results.append({
                'M': M, 'M_beta': M_beta, 'gamma': gamma,
                'build_status': 'failed',
                'build_error': error,
            })
            continue

        log(f"   ✅ 构建成功: {build_time:.1f}秒, {index_size:.1f}MB")

        # 测试搜索
        scenario = config['scenarios'][0]
        results_dir = f"{output_base}/results/{scenario}"
        log(f"🔍 测试搜索 ({scenario})...")

        success, recall, qps, error = search_index(dataset, M, M_beta, gamma, scenario, config, index_dir, results_dir)

        if success:
            log(f"   ✅ 搜索成功: Recall@10={recall:.4f}, QPS={qps:.2f}")
            results.append({
                'M': M, 'M_beta': M_beta, 'gamma': gamma,
                'build_status': 'success',
                'build_time_s': build_time,
                'index_size_mb': index_size,
                'recall@10': recall,
                'qps': qps,
            })
        else:
            log(f"   ❌ 搜索失败: {error}")
            results.append({
                'M': M, 'M_beta': M_beta, 'gamma': gamma,
                'build_status': 'success',
                'build_time_s': build_time,
                'index_size_mb': index_size,
                'search_status': 'failed',
                'search_error': error,
            })

    return results

def print_summary(dataset, results):
    """打印结果摘要"""
    log(f"\n{'='*70}")
    log(f"📊 {dataset.upper()} - 测试结果摘要")
    log(f"{'='*70}")

    if not results:
        log("无结果")
        return

    # 统计
    total = len(results)
    build_success = len([r for r in results if r.get('build_status') == 'success'])
    search_success = len([r for r in results if r.get('recall@10', 0) > 0])

    log(f"总测试: {total}")
    log(f"构建成功: {build_success}/{total}")
    log(f"搜索成功: {search_success}/{total}")

    # 成功的结果
    successful = [r for r in results if r.get('recall@10', 0) > 0]
    if successful:
        log(f"\n成功测试详情:")
        log(f"  {'M':>3} {'Mb':>3} {'γ':>2}  {'构建(s)':>8}  {'索引(MB)':>9}  {'Recall':>7}  {'QPS':>8}")
        log(f"  {'-'*60}")
        for r in successful:
            log(f"  {r['M']:>3} {r['M_beta']:>3} {r['gamma']:>2}  "
                f"{r['build_time_s']:>8.1f}  {r['index_size_mb']:>9.1f}  "
                f"{r['recall@10']:>7.4f}  {r['qps']:>8.2f}")

        # Recall范围
        recalls = [r['recall@10'] for r in successful]
        log(f"\nRecall@10 范围: {min(recalls):.4f} - {max(recalls):.4f}")

        # 平均构建时间
        avg_build = sum(r['build_time_s'] for r in successful) / len(successful)
        log(f"平均构建时间: {avg_build:.1f}秒")

    # 失败的结果
    failed = [r for r in results if r.get('build_status') == 'failed']
    if failed:
        log(f"\n⚠️  构建失败的参数:")
        for r in failed:
            log(f"  M={r['M']}, M_beta={r['M_beta']}, gamma={r['gamma']}: {r.get('build_error', 'Unknown')}")

def suggest_params(all_results):
    """根据测试结果建议参数范围"""
    log(f"\n{'='*70}")
    log("💡 参数范围建议")
    log(f"{'='*70}")

    for dataset, results in all_results.items():
        if not results:
            continue

        successful = [r for r in results if r.get('recall@10', 0) > 0]
        failed = [r for r in results if r.get('build_status') == 'failed']

        log(f"\n{dataset.upper()}:")

        if not successful:
            log("  ⚠️  所有测试失败，需要检查数据集或降低参数")
            continue

        # 找出最高Recall
        best = max(successful, key=lambda x: x['recall@10'])
        log(f"  ✅ 最佳Recall: {best['recall@10']:.4f} (M={best['M']}, M_beta={best['M_beta']}, gamma={best['gamma']})")

        # 推荐参数范围
        if best['recall@10'] < 0.80:
            log(f"  ⚠️  Recall偏低，建议增加参数范围（更大的M_beta和gamma）")
            log(f"     推荐: Ms=[32,48,64], M_betas=[64,96,128], gammas=[8,12,24]")
        elif best['recall@10'] > 0.95:
            log(f"  ✅ Recall很高，可以使用较小参数范围节省时间")
            log(f"     推荐: Ms=[32,48], M_betas=[48,64], gammas=[4,8]")
        else:
            log(f"  ✅ Recall合理，使用默认参数范围")
            log(f"     推荐: Ms=[32,48,64], M_betas=[48,64,96], gammas=[4,8,12]")

        if failed:
            log(f"  ⚠️  {len(failed)}个参数组合构建失败，建议从完整搜索中排除")

# ==================== 主函数 ====================

def main():
    log("="*70)
    log("🚀 ACORN快速测试 - 所有数据集")
    log("="*70)
    log(f"测试参数: {TEST_PARAMS}")
    log(f"数据集: {list(DATASETS_CONFIG.keys())}")
    log("")

    all_results = {}

    if len(sys.argv) > 1:
        # 测试指定数据集
        datasets = sys.argv[1:]
    else:
        # 测试所有数据集
        datasets = list(DATASETS_CONFIG.keys())

    start_time = time.time()

    for dataset in datasets:
        results = test_dataset(dataset)
        if results:
            all_results[dataset] = results
            print_summary(dataset, results)

    total_time = time.time() - start_time

    log(f"\n{'='*70}")
    log(f"✅ 所有测试完成！")
    log(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    log(f"{'='*70}")

    # 给出参数建议
    suggest_params(all_results)

if __name__ == "__main__":
    main()
