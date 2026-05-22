#!/usr/bin/env python3
"""
ACORN参数搜索 - 超快速测试版本
用途：验证整个流程能否正常运行（5分钟完成）
"""

import subprocess
import os
import time
from datetime import datetime

# ==================== 测试配置 ====================
DATASET = "arxiv"
N = 132687

# ⚠️ 云端实际数据路径
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv"

# 只测试1个参数组合
TEST_M = 16
TEST_M_BETA = 32
TEST_GAMMA = 1

# 只测试1个场景
TEST_SCENARIO = "equal"

# 输出路径（在ACORN目录下）
OUTPUT_DIR = "/home/remote/u7905817/benchmarks/discrete/ACORN/data/test_search_arxiv"
INDEX_DIR = f"{OUTPUT_DIR}/index"
RESULT_DIR = f"{OUTPUT_DIR}/result"

# ==================== 工具函数 ====================

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

def check_file(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        log(f"✅ {description}: {filepath}")
        return True
    else:
        log(f"❌ {description}不存在: {filepath}")
        return False

# ==================== 测试流程 ====================

def test_full_workflow():
    """测试完整工作流"""

    log("="*70)
    log("ACORN参数搜索 - 超快速测试")
    log("="*70)
    log(f"数据集: {DATASET}")
    log(f"测试参数: M={TEST_M}, M_beta={TEST_M_BETA}, gamma={TEST_GAMMA}")
    log(f"测试场景: {TEST_SCENARIO}")
    log("="*70)

    # ========== 步骤0: 检查数据文件 ==========
    log("\n【步骤0】检查数据文件...")

    required_files = {
        "基础向量": f"{DATA_DIR}/arxiv_base.fvecs",
        "基础标签": f"{DATA_DIR}/label_base.txt",
        f"{TEST_SCENARIO}查询向量": f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.fvecs",
        f"{TEST_SCENARIO}查询标签": f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.txt",
        f"{TEST_SCENARIO}_GT": f"{DATA_DIR}/arxiv_gt_{TEST_SCENARIO}.txt",
    }

    all_exist = True
    for desc, filepath in required_files.items():
        if not check_file(filepath, desc):
            all_exist = False

    if not all_exist:
        log("\n❌ 数据文件缺失！请先确保数据文件在正确位置。")
        log(f"   数据目录应该是: {os.path.abspath(DATA_DIR)}")
        return False

    log("\n✅ 所有数据文件检查通过！")

    # ========== 步骤1: 测试构建索引 ==========
    log("\n【步骤1】测试构建索引...")

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(f"{INDEX_DIR}/{DATASET}", exist_ok=True)  # Create arxiv subdirectory
    build_log = f"{INDEX_DIR}/build.log"

    build_cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(TEST_GAMMA),
        f"{DATA_DIR}/arxiv_base.fvecs",
        str(TEST_M), str(TEST_M_BETA),
        INDEX_DIR, DATASET
    ]

    log(f"   执行命令: {' '.join(build_cmd)}")

    start_time = time.time()

    try:
        with open(build_log, 'w') as f:
            result = subprocess.run(
                build_cmd,
                stdout=f, stderr=subprocess.STDOUT,
                check=True,
                timeout=600  # 10分钟超时
            )

        build_time = time.time() - start_time
        log(f"   ✅ 构建成功！耗时: {build_time:.1f}秒")

        # 检查索引文件是否生成
        index_file = f"{INDEX_DIR}/{DATASET}/hybrid_M={TEST_M}_Mb={TEST_M_BETA}_gamma={TEST_GAMMA}.json"
        if check_file(index_file, "索引文件"):
            size_mb = os.path.getsize(index_file) / (1024 * 1024)
            log(f"   索引大小: {size_mb:.1f} MB")
        else:
            log("   ⚠️  警告：索引文件未找到")
            return False

    except subprocess.TimeoutExpired:
        log("   ❌ 构建超时（10分钟）")
        return False
    except subprocess.CalledProcessError as e:
        log(f"   ❌ 构建失败！")
        log(f"   查看日志: cat {build_log}")
        return False
    except FileNotFoundError:
        log("   ❌ 找不到可执行文件 ../build/demos/build_acorn_index")
        log("   请先编译项目: cd ACORN && mkdir build && cd build && cmake .. && make")
        return False

    # ========== 步骤2: 测试搜索性能 ==========
    log("\n【步骤2】测试搜索性能...")

    os.makedirs(f"{RESULT_DIR}/{DATASET}/{TEST_SCENARIO}", exist_ok=True)
    search_log = f"{RESULT_DIR}/{DATASET}/{TEST_SCENARIO}/search.log"

    search_cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(TEST_GAMMA), DATASET,
        str(TEST_M), str(TEST_M_BETA),
        INDEX_DIR, TEST_SCENARIO, RESULT_DIR,
        f"{DATA_DIR}/arxiv_base.fvecs",
        f"{DATA_DIR}/label_base.txt",
        f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.fvecs",
        f"{DATA_DIR}/arxiv_query_{TEST_SCENARIO}.txt",
        f"{DATA_DIR}/arxiv_gt_{TEST_SCENARIO}.txt",
        "10"  # K=10
    ]

    log(f"   执行命令: {' '.join(search_cmd)}")

    start_time = time.time()

    try:
        with open(search_log, 'w') as f:
            result = subprocess.run(
                search_cmd,
                stdout=f, stderr=subprocess.STDOUT,
                check=True,
                timeout=600  # 10分钟超时
            )

        search_time = time.time() - start_time
        log(f"   ✅ 搜索成功！耗时: {search_time:.1f}秒")

        # 尝试解析结果
        log("\n   尝试解析性能指标...")
        with open(search_log, 'r') as f:
            content = f.read()

            # 查找关键指标
            found_metrics = False
            for line in content.split('\n'):
                if 'Recall' in line or 'QPS' in line or 'Time' in line:
                    log(f"   📊 {line.strip()}")
                    found_metrics = True

            if not found_metrics:
                log("   ⚠️  未找到性能指标，请查看日志文件")
                log(f"   cat {search_log}")

    except subprocess.TimeoutExpired:
        log("   ❌ 搜索超时（10分钟）")
        return False
    except subprocess.CalledProcessError as e:
        log(f"   ❌ 搜索失败！")
        log(f"   查看日志: cat {search_log}")
        return False

    # ========== 步骤3: 测试完成 ==========
    log("\n" + "="*70)
    log("✅ 测试完成！整个流程可以正常运行")
    log("="*70)
    log(f"构建时间: {build_time:.1f}秒")
    log(f"搜索时间: {search_time:.1f}秒")
    log(f"总耗时: {build_time + search_time:.1f}秒")
    log("\n📁 测试输出目录: " + os.path.abspath(OUTPUT_DIR))
    log(f"   - 索引: {INDEX_DIR}")
    log(f"   - 结果: {RESULT_DIR}")
    log("\n💡 下一步:")
    log("   1. 检查结果是否合理（Recall应该>0.80）")
    log("   2. 如果正常，运行小规模测试")
    log("   3. 最后运行完整参数搜索")
    log("="*70)

    return True

if __name__ == '__main__':
    success = test_full_workflow()

    if not success:
        print("\n❌ 测试失败！请根据上面的错误信息排查问题。")
        exit(1)
    else:
        print("\n✅ 可以进行下一步了！")
        exit(0)
