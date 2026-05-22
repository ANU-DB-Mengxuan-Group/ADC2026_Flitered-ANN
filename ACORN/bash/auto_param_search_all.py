#!/usr/bin/env python3
"""
ACORN参数自动搜索 - 支持所有数据集
用法: python auto_param_search_all.py [dataset_name]
不指定dataset则搜索所有数据集
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

DATASETS_CONFIG = {
    "arxiv": {
        "N": 132687,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/arxiv",
        "scenarios": ["equal", "or", "and"],
        # 参数范围：基于arxiv测试结果
        "Ms": [32, 48, 64],
        "M_betas": [48, 64, 96, 128],
        "gammas": [4, 8, 12, 24],
    },
    "LAION1M": {
        "N": 1000448,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M",
        "scenarios": ["and"],  # LAION1M只支持and场景
        # 参数范围：可能需要调整
        "Ms": [32, 48, 64],
        "M_betas": [48, 64, 96, 128],
        "gammas": [4, 8, 12, 24],
    },
    "ytb_video": {
        "N": 5000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video",
        "scenarios": ["equal", "or", "and"],
        # 参数范围：大数据集可能需要调整
        "Ms": [32, 48, 64],
        "M_betas": [48, 64, 96, 128],
        "gammas": [4, 8, 12, 24],
    },
    "tripclick": {
        "N": 1055976,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick",
        "scenarios": ["equal", "and"],  # tripclick不支持or场景
        # 参数范围：可能需要调整
        "Ms": [32, 48, 64],
        "M_betas": [48, 64, 96, 128],
        "gammas": [4, 8, 12, 24],
    },
    "yfcc": {
        "N": 1000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc",
        "scenarios": ["equal", "or", "and"],
        # 参数范围：可能需要调整
        "Ms": [32, 48, 64],
        "M_betas": [48, 64, 96, 128],
        "gammas": [4, 8, 12, 24],
    },
}

# 全局常量
K = 10  # Recall@10

# ==================== 辅助函数 ====================

def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def check_dataset_files(dataset, config):
    """检查数据集文件是否存在"""
    data_dir = config['data_dir']
    required_files = [
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt",
    ]

    for scenario in config['scenarios']:
        required_files.extend([
            f"{data_dir}/{dataset}_query_{scenario}.fvecs",
            f"{data_dir}/{dataset}_query_{scenario}.txt",
            f"{data_dir}/{dataset}_gt_{scenario}.txt",
        ])

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        log(f"❌ 数据集 {dataset} 缺少文件:")
        for f in missing:
            log(f"   - {f}")
        return False

    log(f"✅ 数据集 {dataset} 文件检查通过")
    return True

def load_progress(progress_file):
    """加载进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(tuple(x) for x in data.get('completed', []))
    return set()

def save_progress(progress_file, completed_tasks):
    """保存进度"""
    output_dir = os.path.dirname(progress_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump({
            'completed': list(completed_tasks),
            'last_update': datetime.now().isoformat(),
            'total': len(completed_tasks)
        }, f, indent=2)

def load_build_info(summary_file):
    """从已有的summary.csv加载构建信息"""
    build_info = {}

    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
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

def init_summary_file(summary_file):
    """初始化汇总文件"""
    if not os.path.exists(summary_file):
        output_dir = os.path.dirname(summary_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'M', 'M_beta', 'gamma', 'scenario',
                'build_time_s', 'qps', 'qps_no_filter',
                'recall@10', 'index_size_mb', 'status', 'timestamp'
            ])

def get_index_size(index_path):
    """获取索引文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(index_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def parse_search_csv(csv_file):
    """从CSV文件提取性能指标（最优Recall值）"""
    try:
        import csv as csv_module
        with open(csv_file, 'r') as f:
            reader = csv_module.DictReader(f)
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

# ==================== 主流程 ====================

def build_index(dataset, M, M_beta, gamma, config, output_base):
    """构建索引"""
    index_dir = f"{output_base}/indices/{dataset}"
    os.makedirs(index_dir, exist_ok=True)

    log_file = f"{index_dir}/M={M}_Mb={M_beta}_gamma={gamma}_build.log"

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma), dataset,
        str(M), str(M_beta), index_dir,
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt"
    ]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'

    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, check=True,
                         stdout=f, stderr=subprocess.STDOUT,
                         timeout=3600)

        build_time = time.time() - start_time
        index_path = f"{index_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
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

def search_index(dataset, M, M_beta, gamma, scenario, config, output_base):
    """测试索引搜索性能"""
    index_dir = f"{output_base}/indices/{dataset}"
    results_dir = f"{output_base}/results/{dataset}/{scenario}"
    os.makedirs(results_dir, exist_ok=True)

    log_file = f"{results_dir}/M={M}_Mb={M_beta}_gamma={gamma}_search.log"

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), dataset,
        str(M), str(M_beta), index_dir, scenario, results_dir,
        f"{data_dir}/{dataset}_base.fvecs",
        f"{data_dir}/label_base.txt",
        f"{data_dir}/{dataset}_query_{scenario}.fvecs",
        f"{data_dir}/{dataset}_query_{scenario}.txt",
        f"{data_dir}/{dataset}_gt_{scenario}.txt",
        str(K)
    ]

    env = os.environ.copy()
    env['debugSearchFlag'] = '0'

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, env=env, check=True,
                         stdout=f, stderr=subprocess.STDOUT,
                         timeout=1800)

        csv_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
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

def run_param_search(dataset):
    """运行单个数据集的参数搜索"""

    if dataset not in DATASETS_CONFIG:
        log(f"❌ 未知数据集: {dataset}")
        log(f"支持的数据集: {', '.join(DATASETS_CONFIG.keys())}")
        return

    config = DATASETS_CONFIG[dataset]

    log("="*70)
    log(f"ACORN参数搜索 - {dataset.upper()} 数据集")
    log("="*70)

    # 检查数据文件
    if not check_dataset_files(dataset, config):
        return

    # 输出路径
    output_base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_{dataset}"
    summary_file = f"{output_base}/summary.csv"
    progress_file = f"{output_base}/progress.json"

    # 初始化
    init_summary_file(summary_file)
    completed_tasks = load_progress(progress_file)
    build_info = load_build_info(summary_file)

    # 计算有效参数组合
    valid_combinations = []
    for M in config['Ms']:
        for M_beta in config['M_betas']:
            if M_beta < M:
                continue
            for gamma in config['gammas']:
                if M_beta > 2 * M * gamma:
                    continue
                valid_combinations.append((M, M_beta, gamma))

    total_tasks = len(valid_combinations) * len(config['scenarios'])

    log(f"有效参数组合: {len(valid_combinations)}")
    log(f"总任务数: {total_tasks} (包含{len(config['scenarios'])}个场景)")
    log(f"已完成: {len([t for t in completed_tasks if t[-1] in config['scenarios']])}")
    log(f"剩余: {total_tasks - len([t for t in completed_tasks if t[-1] in config['scenarios']])}")

    start_time = time.time()
    current_task = 0

    # 遍历所有参数组合
    for M, M_beta, gamma in valid_combinations:

        log(f"\n{'='*70}")
        log(f"参数组合: M={M}, M_beta={M_beta}, gamma={gamma}")
        log(f"{'='*70}")

        # 步骤1: 构建索引
        build_key = (M, M_beta, gamma, 'build')
        index_path = f"{output_base}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        index_exists = os.path.exists(index_path) and os.path.getsize(index_path) > 1000

        if build_key not in completed_tasks or not index_exists:
            if not index_exists and build_key in completed_tasks:
                log(f"⚠️  索引文件损坏或不完整，重新构建...")
                completed_tasks.discard(build_key)
            else:
                log(f"🔨 构建索引...")

            success, build_time, index_size = build_index(dataset, M, M_beta, gamma, config, output_base)

            if success:
                log(f"   ✅ 构建成功: {build_time:.1f}秒, 大小: {index_size:.1f}MB")
                completed_tasks.add(build_key)
                save_progress(progress_file, completed_tasks)
            else:
                log(f"   ❌ 构建失败，跳过此参数组合")
                continue
        else:
            log(f"⏭️  索引已存在，跳过构建")
            param_key = (M, M_beta, gamma)
            if param_key in build_info:
                build_time = build_info[param_key]['build_time']
                index_size = build_info[param_key]['index_size']
                log(f"   📋 恢复构建信息: {build_time:.1f}秒, {index_size:.1f}MB")
            else:
                build_time = 0
                index_size = get_index_size(index_path)
                log(f"   📏 索引大小: {index_size:.1f}MB (构建时间未记录)")

        # 步骤2: 测试所有场景
        for scenario in config['scenarios']:
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

            success, metrics = search_index(dataset, M, M_beta, gamma, scenario, config, output_base)

            if success and metrics:
                log(f"   ✅ 测试成功:")
                log(f"      QPS (with filter): {metrics['qps']:.2f}")
                log(f"      Recall@10: {metrics['recall@10']:.4f}")

                with open(summary_file, 'a', newline='') as f:
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
                save_progress(progress_file, completed_tasks)
            else:
                log(f"   ❌ 测试失败")
                with open(summary_file, 'a', newline='') as f:
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
    log(f"🎉 {dataset.upper()} 数据集参数搜索完成！")
    log(f"⏱️  总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    log(f"📊 结果保存在: {summary_file}")
    log(f"{'='*70}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        run_param_search(dataset)
    else:
        # 运行所有数据集
        log("🚀 开始所有数据集的参数搜索")
        for dataset in DATASETS_CONFIG.keys():
            run_param_search(dataset)
            log(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
