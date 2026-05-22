#!/usr/bin/env python3
"""
ACORN自动参数搜索 - Arxiv数据集
完整流程：构建索引 → 测试性能 → 找到最优参数
"""

import subprocess
import os
import csv
import time
import json
from datetime import datetime
from pathlib import Path

# ==================== 配置 ====================

# 数据集配置
DATASET = "arxiv"
N = 132687  # arxiv数据量

# 数据路径（云端实际路径）
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv"
BASE_FILE = f"{DATA_DIR}/arxiv_base.fvecs"
BASE_LABEL_FILE = f"{DATA_DIR}/label_base.txt"

# Query场景
SCENARIOS = ["equal", "or", "and"]  # 三种查询类型

# 输出路径（在ACORN目录下）
OUTPUT_BASE = "/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_arxiv"
RESULTS_DIR = f"{OUTPUT_BASE}/results"
INDEX_DIR = f"{OUTPUT_BASE}/indices"

# 参数范围 - 基于测试结果优化（gamma需要足够大才能达到Recall>0.80）
Ms = [32, 48, 64]                     # [32, 48, 64] - 去掉过小的值
M_betas = [48, 64, 96, 128]           # [48, 64, 96, 128] - 去掉过小的值
gammas = [4, 8, 12, 24]               # [4, 8, 12, 24] - gamma>=4才能达到目标Recall

# 搜索参数
K = 10  # Top-K

# 进度文件
PROGRESS_FILE = f"{OUTPUT_BASE}/progress.json"
SUMMARY_FILE = f"{OUTPUT_BASE}/summary.csv"

# ==================== 工具函数 ====================

def log(msg):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

def load_progress():
    """加载进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return set(tuple(x) for x in data.get('completed', []))
    return set()

def save_progress(completed_tasks):
    """保存进度"""
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            'completed': list(completed_tasks),
            'last_update': datetime.now().isoformat(),
            'total': len(completed_tasks)
        }, f, indent=2)

def load_build_info():
    """从已有的summary.csv加载构建信息（用于恢复build_time和index_size）"""
    build_info = {}  # {(M, M_beta, gamma): {'build_time': x, 'index_size': y}}

    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(float(row['M'])), int(float(row['M_beta'])), int(float(row['gamma'])))
                    if key not in build_info and row.get('build_time_s'):
                        try:
                            build_info[key] = {
                                'build_time': float(row['build_time_s']),
                                'index_size': float(row['index_size_mb'])
                            }
                        except:
                            pass
        except:
            pass

    return build_info

def init_summary_file():
    """初始化汇总文件"""
    if not os.path.exists(SUMMARY_FILE):
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'M', 'M_beta', 'gamma', 'scenario',
                'build_time_s', 'qps', 'qps_no_filter',
                'recall@10', 'index_size_mb', 'status', 'timestamp'
            ])

def parse_build_log(log_file):
    """解析构建日志，提取构建时间"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Create gamma index in time:' in line:
                    return float(line.split(':')[-1].strip())
    except:
        pass
    return 0.0

def parse_search_csv(csv_file):
    """从CSV文件提取性能指标（最优Recall值）"""
    try:
        import csv
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
                    'recall@10': best_recall,
                    'qps': best_qps,
                    'qps_no_filter': best_qps_no_filter
                }
    except Exception as e:
        pass
    return None

def get_index_size(index_path):
    """获取索引文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(index_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

# ==================== 主流程 ====================

def check_data_files():
    """检查数据文件是否存在"""
    log("检查数据文件...")

    required_files = [BASE_FILE, BASE_LABEL_FILE]

    for scenario in SCENARIOS:
        required_files.extend([
            f"{DATA_DIR}/arxiv_query_{scenario}.fvecs",
            f"{DATA_DIR}/arxiv_query_{scenario}.txt",
            f"{DATA_DIR}/arxiv_gt_{scenario}.txt"
        ])

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        log(f"❌ 缺少以下数据文件：")
        for f in missing:
            log(f"   - {f}")
        log(f"\n请确保数据文件在正确位置！")
        return False

    log("✅ 所有数据文件就绪")
    return True

def build_index(M, M_beta, gamma):
    """构建索引"""
    # Ensure base index directory exists for ACORN to write index files
    os.makedirs(os.path.join(INDEX_DIR, DATASET), exist_ok=True)

    dir_name = f"M={M}_Mb={M_beta}_gamma={gamma}"
    index_dir = os.path.join(INDEX_DIR, DATASET, dir_name)
    os.makedirs(index_dir, exist_ok=True)

    log_file = os.path.join(index_dir, 'build.log')

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma), BASE_FILE,
        str(M), str(M_beta), INDEX_DIR, DATASET
    ]

    env = os.environ.copy()
    env['debugSearchFlag'] = '0'

    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, check=True,
                         stdout=f, stderr=subprocess.STDOUT,
                         timeout=3600)

        build_time = time.time() - start_time

        # 获取索引文件路径
        index_path = f"{INDEX_DIR}/{DATASET}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        index_size = get_index_size(index_path)

        return True, build_time, index_size

    except subprocess.TimeoutExpired:
        log(f"   ⏱️  构建超时（1小时）")
        return False, 0, 0
    except subprocess.CalledProcessError as e:
        log(f"   ❌ 构建失败: {e}")
        return False, 0, 0
    except Exception as e:
        log(f"   ❌ 异常: {e}")
        return False, 0, 0

def search_index(M, M_beta, gamma, scenario):
    """测试索引搜索性能"""
    # 输出路径应该包含dataset和scenario（与C++程序期望一致）
    output_path = os.path.join(RESULTS_DIR, DATASET, scenario)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, f'M={M}_Mb={M_beta}_gamma={gamma}_search.log')

    query_file = f"{DATA_DIR}/arxiv_query_{scenario}.fvecs"
    query_label_file = f"{DATA_DIR}/arxiv_query_{scenario}.txt"
    gt_path = f"{DATA_DIR}/arxiv_gt_{scenario}.txt"

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), DATASET,
        str(M), str(M_beta), INDEX_DIR, scenario, output_path,  # 传递完整路径
        BASE_FILE, BASE_LABEL_FILE,
        query_file, query_label_file, gt_path, str(K)
    ]

    env = os.environ.copy()
    env['debugSearchFlag'] = '0'

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, check=True,
                         stdout=f, stderr=subprocess.STDOUT,
                         timeout=1800)

        # 从CSV文件解析性能指标
        csv_file = os.path.join(output_path, f"M={M}_M_beta={M_beta}_gamma={gamma}_result.csv")
        metrics = parse_search_csv(csv_file)
        return True, metrics

    except subprocess.TimeoutExpired:
        log(f"   ⏱️  搜索超时（30分钟）")
        return False, None
    except subprocess.CalledProcessError as e:
        log(f"   ❌ 搜索失败: {e}")
        return False, None
    except Exception as e:
        log(f"   ❌ 异常: {e}")
        return False, None

def main():
    """主函数"""
    log("="*70)
    log("ACORN自动参数搜索 - Arxiv数据集")
    log("="*70)

    # 检查数据文件
    if not check_data_files():
        return

    # 初始化
    init_summary_file()
    completed_tasks = load_progress()
    build_info = load_build_info()  # 加载已有的构建信息

    # 计算总任务数
    total_tasks = 0
    valid_combinations = []

    for M in Ms:
        for M_beta in M_betas:
            if M_beta < M:
                continue
            for gamma in gammas:
                if M_beta > 2 * M * gamma:
                    continue
                valid_combinations.append((M, M_beta, gamma))
                total_tasks += len(SCENARIOS)

    log(f"\n参数配置：")
    log(f"  M范围: {Ms}")
    log(f"  M_beta范围: {M_betas}")
    log(f"  gamma范围: {gammas}")
    log(f"  查询场景: {SCENARIOS}")
    log(f"\n有效参数组合: {len(valid_combinations)}")
    log(f"总任务数: {total_tasks} (包含{len(SCENARIOS)}个场景)")
    log(f"已完成: {len(completed_tasks)}")
    log(f"剩余: {total_tasks - len(completed_tasks)}\n")

    start_time = time.time()
    current_task = 0

    # 遍历所有参数组合
    for M, M_beta, gamma in valid_combinations:

        log(f"\n{'='*70}")
        log(f"参数组合: M={M}, M_beta={M_beta}, gamma={gamma}")
        log(f"{'='*70}")

        # 步骤1: 构建索引（如果还没构建）
        build_key = (M, M_beta, gamma, 'build')

        # 检查索引文件是否存在且完整
        index_path = f"{INDEX_DIR}/{DATASET}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        index_exists = os.path.exists(index_path) and os.path.getsize(index_path) > 1000  # 至少1KB

        if build_key not in completed_tasks or not index_exists:
            if not index_exists and build_key in completed_tasks:
                log(f"⚠️  索引文件损坏或不完整，重新构建...")
                completed_tasks.discard(build_key)  # 从已完成列表移除
            else:
                log(f"🔨 构建索引...")

            success, build_time, index_size = build_index(M, M_beta, gamma)

            if success:
                log(f"   ✅ 构建成功: {build_time:.1f}秒, 大小: {index_size:.1f}MB")
                completed_tasks.add(build_key)
                save_progress(completed_tasks)
            else:
                log(f"   ❌ 构建失败，跳过此参数组合")
                continue
        else:
            log(f"⏭️  索引已存在，跳过构建")
            # 尝试从之前的summary恢复构建信息
            param_key = (M, M_beta, gamma)
            if param_key in build_info:
                build_time = build_info[param_key]['build_time']
                index_size = build_info[param_key]['index_size']
                log(f"   📋 恢复构建信息: {build_time:.1f}秒, {index_size:.1f}MB")
            else:
                # 如果没有历史数据，只能读取文件大小
                build_time = 0  # 无法恢复
                index_size = get_index_size(index_path)
                log(f"   📏 索引大小: {index_size:.1f}MB (构建时间未记录)")

        # 步骤2: 测试所有场景
        for scenario in SCENARIOS:
            current_task += 1
            task_key = (M, M_beta, gamma, scenario)

            if task_key in completed_tasks:
                log(f"⏭️  [{current_task}/{total_tasks}] 场景 {scenario.upper()} 已完成")
                continue

            progress_pct = (current_task / total_tasks) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / current_task) * (total_tasks - current_task) if current_task > 0 else 0

            log(f"\n🔍 [{current_task}/{total_tasks}] ({progress_pct:.1f}%) 测试场景: {scenario.upper()}")
            log(f"⏱️  已用时: {elapsed/60:.1f}分钟 | 预计剩余: {eta/60:.1f}分钟")

            success, metrics = search_index(M, M_beta, gamma, scenario)

            if success and metrics:
                log(f"   ✅ 测试成功:")
                log(f"      QPS (with filter): {metrics['qps']:.2f}")
                log(f"      Recall@10: {metrics['recall@10']:.4f}")

                # 记录结果
                with open(SUMMARY_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        M, M_beta, gamma, scenario,
                        f"{build_time:.2f}",
                        f"{metrics['qps']:.2f}",
                        f"{metrics['qps_no_filter']:.2f}",
                        f"{metrics['recall@10']:.4f}",
                        f"{index_size:.1f}",
                        'success',
                        datetime.now().isoformat()
                    ])

                completed_tasks.add(task_key)
                save_progress(completed_tasks)
            else:
                log(f"   ❌ 测试失败")
                # 也记录失败的结果
                with open(SUMMARY_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        M, M_beta, gamma, scenario,
                        0, 0, 0, 0, 0,
                        'failed',
                        datetime.now().isoformat()
                    ])

    # 完成总结
    total_time = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"🎉 所有任务完成！")
    log(f"⏱️  总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    log(f"📊 结果保存在: {SUMMARY_FILE}")
    log(f"{'='*70}")

    # 分析最优参数
    analyze_results()

def analyze_results():
    """分析结果，找出最优参数"""
    log(f"\n{'='*70}")
    log("📈 最优参数分析")
    log(f"{'='*70}\n")

    try:
        import pandas as pd
        df = pd.read_csv(SUMMARY_FILE)
        df = df[df['status'] == 'success']

        for scenario in SCENARIOS:
            log(f"\n场景: {scenario.upper()}")
            log("-" * 50)

            scenario_df = df[df['scenario'] == scenario]

            if len(scenario_df) == 0:
                log("  没有成功的结果")
                continue

            # 按recall@10排序，取前5
            top5 = scenario_df.nlargest(5, 'recall@10')

            log(f"  {'M':>3} {'Mb':>3} {'γ':>2}  {'R@10':>7}  {'QPS':>9}  {'内存(MB)':>9}")
            log("  " + "-" * 50)

            for _, row in top5.iterrows():
                log(f"  {int(row['M']):>3} {int(row['M_beta']):>3} {int(row['gamma']):>2}  "
                    f"{row['recall@10']:>7.4f}  {row['qps']:>9.2f}  "
                    f"{row['index_size_mb']:>9.1f}")

            # 推荐最优（recall最高且QPS合理）
            best = top5.iloc[0]
            log(f"\n  🏆 推荐: M={int(best['M'])}, M_beta={int(best['M_beta'])}, gamma={int(best['gamma'])}")
            log(f"     Recall@10={best['recall@10']:.4f}, QPS={best['qps']:.2f}")

    except ImportError:
        log("提示：安装pandas可以自动分析最优参数")
        log("  pip install pandas")
    except Exception as e:
        log(f"分析失败: {e}")

if __name__ == '__main__':
    main()
