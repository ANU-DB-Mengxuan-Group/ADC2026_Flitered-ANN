#!/usr/bin/env python3
"""
测试YFCC数据集用大参数的Recall表现
"""
import os
import subprocess
import time

# 配置
DATASET = "yfcc"
N = 1000000
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc"
OUTPUT_BASE = "/home/remote/u7905817/benchmarks/discrete/ACORN/data/yfcc_large_param_test"
SCENARIO = "equal"
K = 10

# 测试参数：使用更大的M和M_beta配合大gamma
TEST_CONFIGS = [
    # (M, M_beta, gamma)
    (64, 96, 12),   # Quick test显示Recall=0.3963
    (64, 96, 24),   # 测试更大的gamma
    (64, 128, 24),  # 测试更大的M_beta
]

def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_index(M, M_beta, gamma):
    """构建索引"""
    # 创建所有需要的目录
    indices_dir = f"{OUTPUT_BASE}/indices"
    dataset_dir = f"{indices_dir}/{DATASET}"
    os.makedirs(dataset_dir, exist_ok=True)

    # 检查索引是否已存在
    index_file = f"{dataset_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
    if os.path.exists(index_file):
        size_mb = os.path.getsize(index_file) / (1024 * 1024)
        # 检查文件大小是否合理（至少100MB）
        if size_mb > 100:
            log(f"⏭️  跳过构建（索引已存在）: M={M}, M_beta={M_beta}, gamma={gamma}, {size_mb:.1f}MB")
            return True, 0, size_mb
        else:
            log(f"⚠️  删除损坏的索引文件（{size_mb:.1f}MB < 100MB）")
            os.remove(index_file)

    log(f"构建索引: M={M}, M_beta={M_beta}, gamma={gamma}")

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma),
        f"{DATA_DIR}/{DATASET}_base.fvecs",
        str(M), str(M_beta),
        indices_dir, DATASET
    ]

    log_file = f"{dataset_dir}/M={M}_Mb={M_beta}_gamma={gamma}_build.log"

    start = time.time()
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=3600)

        build_time = time.time() - start

        # 检查索引文件
        if os.path.exists(index_file):
            size_mb = os.path.getsize(index_file) / (1024 * 1024)
            log(f"✅ 构建成功: {build_time:.1f}秒, {size_mb:.1f}MB")
            return True, build_time, size_mb
        else:
            log(f"❌ 索引文件不存在: {index_file}")
            return False, 0, 0

    except subprocess.TimeoutExpired:
        log(f"❌ 构建超时（1小时）")
        return False, 0, 0
    except Exception as e:
        log(f"❌ 构建失败: {e}")
        return False, 0, 0

def search_index(M, M_beta, gamma):
    """搜索测试"""
    log(f"搜索测试: M={M}, M_beta={M_beta}, gamma={gamma}")

    # 创建所有需要的目录
    indices_dir = f"{OUTPUT_BASE}/indices"
    results_dir = f"{OUTPUT_BASE}/results"
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), DATASET,
        str(M), str(M_beta),
        indices_dir, SCENARIO, results_dir,
        f"{DATA_DIR}/{DATASET}_base.fvecs",
        f"{DATA_DIR}/label_base.txt",
        f"{DATA_DIR}/{DATASET}_query_{SCENARIO}.fvecs",
        f"{DATA_DIR}/{DATASET}_query_{SCENARIO}.txt",
        f"{DATA_DIR}/{DATASET}_gt_{SCENARIO}.txt",
        str(K)
    ]

    log_file = f"{results_dir}/M={M}_Mb={M_beta}_gamma={gamma}_search.log"

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1800)

        # 读取CSV结果
        csv_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
        if os.path.exists(csv_file):
            # 找最高Recall
            max_recall = 0.0
            best_qps = 0.0
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # 跳过header
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            recall = float(parts[3])
                            qps = float(parts[2])
                            if recall > max_recall:
                                max_recall = recall
                                best_qps = qps

            log(f"✅ 搜索成功: Recall={max_recall:.4f}, QPS={best_qps:.2f}")
            return True, max_recall, best_qps
        else:
            log(f"❌ CSV文件不存在: {csv_file}")
            return False, 0, 0

    except Exception as e:
        log(f"❌ 搜索失败: {e}")
        return False, 0, 0

def main():
    log("=" * 70)
    log(f"YFCC大参数测试 - 数据集规模: N={N}")
    log("=" * 70)

    results = []

    for M, M_beta, gamma in TEST_CONFIGS:
        log("")
        log("=" * 70)
        log(f"测试配置: M={M}, M_beta={M_beta}, gamma={gamma}")
        log("=" * 70)

        # 构建
        build_success, build_time, index_size = build_index(M, M_beta, gamma)
        if not build_success:
            log(f"跳过搜索测试（构建失败）")
            continue

        # 搜索
        search_success, recall, qps = search_index(M, M_beta, gamma)

        if search_success:
            results.append({
                'M': M,
                'M_beta': M_beta,
                'gamma': gamma,
                'build_time': build_time,
                'index_size': index_size,
                'recall': recall,
                'qps': qps
            })

    # 打印摘要
    log("")
    log("=" * 70)
    log("测试结果摘要")
    log("=" * 70)
    log(f"{'M':>4}  {'Mb':>4}  {'γ':>3}  {'构建(s)':>10}  {'索引(MB)':>10}  {'Recall':>8}  {'QPS':>10}")
    log("-" * 70)

    for r in results:
        log(f"{r['M']:>4}  {r['M_beta']:>4}  {r['gamma']:>3}  "
            f"{r['build_time']:>10.1f}  {r['index_size']:>10.1f}  "
            f"{r['recall']:>8.4f}  {r['qps']:>10.2f}")

    log("")
    log("=" * 70)
    log("结论:")
    log("=" * 70)

    if results:
        best = max(results, key=lambda x: x['recall'])
        log(f"最佳Recall: {best['recall']:.4f} (M={best['M']}, M_beta={best['M_beta']}, gamma={best['gamma']})")

        if best['recall'] >= 0.80:
            log("✅ 达到目标Recall >= 0.80")
        else:
            log(f"⚠️  未达到目标（差距: {0.80 - best['recall']:.4f}）")
    else:
        log("❌ 所有测试失败")

if __name__ == "__main__":
    main()
