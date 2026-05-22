#!/usr/bin/env python3
"""
ACORN参数测试 - 测试更大的参数以验证Recall
目的：找到能达到Recall>0.80的参数范围
"""

import subprocess
import os
import csv
from datetime import datetime

# ==================== 配置 ====================
DATASET = "arxiv"
N = 132687
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv"
TEST_SCENARIO = "equal"

OUTPUT_DIR = "/home/remote/u7905817/benchmarks/discrete/ACORN/data/test_larger_params"
INDEX_DIR = f"{OUTPUT_DIR}/index"
RESULT_DIR = f"{OUTPUT_DIR}/result"

# 测试几个更大的参数组合
TEST_PARAMS = [
    # (M, M_beta, gamma)
    (32, 64, 1),    # 中等参数
    (48, 96, 2),    # 较大参数
    (48, 96, 1),    # 大参数，gamma=1
]

# ==================== 工具函数 ====================

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

def parse_csv_for_best_recall(csv_file):
    """从CSV中提取最佳Recall值"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            best_recall = 0
            best_config = None
            for row in reader:
                recall = float(row['Recall'])
                if recall > best_recall:
                    best_recall = recall
                    best_config = row
            return best_recall, best_config
    except:
        return None, None

def test_params(M, M_beta, gamma):
    """测试单个参数组合"""
    log(f"\n{'='*70}")
    log(f"测试参数: M={M}, M_beta={M_beta}, gamma={gamma}")
    log(f"{'='*70}")

    # 创建目录
    os.makedirs(f"{INDEX_DIR}/{DATASET}", exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 步骤1: 构建索引
    log("🔨 构建索引...")
    build_cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma),
        f"{DATA_DIR}/arxiv_base.fvecs",
        str(M), str(M_beta),
        INDEX_DIR, DATASET
    ]

    try:
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )
        log(f"   ✅ 构建成功")
    except Exception as e:
        log(f"   ❌ 构建失败: {e}")
        return None

    # 步骤2: 搜索测试
    log("🔍 测试搜索性能...")
    search_cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), DATASET,
        str(M), str(M_beta),
        INDEX_DIR, TEST_SCENARIO, RESULT_DIR,
        f"{DATA_DIR}/arxiv_base.fvecs",
        f"{DATA_DIR}/label_base.txt",
        f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.fvecs",
        f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.txt",
        f"{DATA_DIR}/arxiv_gt_{TEST_SCENARIO}.txt",
        "10"
    ]

    try:
        result = subprocess.run(
            search_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )
        log(f"   ✅ 搜索成功")
    except Exception as e:
        log(f"   ❌ 搜索失败: {e}")
        return None

    # 步骤3: 解析结果
    csv_file = f"{RESULT_DIR}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
    best_recall, config = parse_csv_for_best_recall(csv_file)

    if best_recall:
        log(f"\n   📊 最佳Recall@10: {best_recall:.4f}")
        log(f"      搜索深度L: {config['L']}")
        log(f"      QPS (with filter): {config['QPS']}")

        if best_recall >= 0.80:
            log(f"   ✅ Recall达标 (>= 0.80)！")
            return True
        else:
            log(f"   ⚠️  Recall偏低 (< 0.80)")
            return False
    else:
        log(f"   ❌ 无法解析结果")
        return None

def main():
    log("="*70)
    log("ACORN参数测试 - 寻找合适的参数范围")
    log("="*70)
    log(f"\n将测试 {len(TEST_PARAMS)} 组参数组合")
    log(f"测试参数: {TEST_PARAMS}\n")

    results = []

    for M, M_beta, gamma in TEST_PARAMS:
        success = test_params(M, M_beta, gamma)
        results.append((M, M_beta, gamma, success))

    # 总结
    log(f"\n{'='*70}")
    log("测试总结")
    log(f"{'='*70}")

    passed = []
    failed = []

    for M, M_beta, gamma, success in results:
        if success:
            passed.append((M, M_beta, gamma))
            log(f"✅ M={M}, M_beta={M_beta}, gamma={gamma} - Recall达标")
        elif success is False:
            failed.append((M, M_beta, gamma))
            log(f"⚠️  M={M}, M_beta={M_beta}, gamma={gamma} - Recall偏低")
        else:
            log(f"❌ M={M}, M_beta={M_beta}, gamma={gamma} - 测试失败")

    if passed:
        log(f"\n💡 建议参数搜索范围：")
        min_M = min(p[0] for p in passed)
        min_Mb = min(p[1] for p in passed)
        log(f"   M >= {min_M}")
        log(f"   M_beta >= {min_Mb}")
        log(f"   gamma: {list(set(p[2] for p in passed))}")
    else:
        log(f"\n⚠️  所有测试参数Recall都未达标，需要进一步增大参数！")
        log(f"   建议测试更大的M和M_beta值")

    log(f"\n📁 详细结果保存在: {RESULT_DIR}")
    log("="*70)

if __name__ == '__main__':
    main()
