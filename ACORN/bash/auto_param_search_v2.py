#!/usr/bin/env python3
"""
ACORN参数自动搜索 V2 - 增强版
改进:
1. 完善的断点续传机制（记录每个任务的构建/搜索状态）
2. 自动清理索引文件（搜索完成后删除，节省空间）
3. 检测损坏的构建并自动重试
4. 所有数据集支持3个场景（equal, or, and）
5. 基于测试结果优化的参数范围

用法:
  python auto_param_search_v2.py                    # 运行所有数据集
  python auto_param_search_v2.py yfcc               # 只运行指定数据集
  python auto_param_search_v2.py yfcc LAION1M       # 运行多个数据集

建议用screen或nohup运行:
  screen -S acorn_search
  python -u auto_param_search_v2.py 2>&1 | tee search.log
  # 断开: Ctrl+A, D
  # 重连: screen -r acorn_search
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

# 基于quick test结果优化的参数范围
DEFAULT_PARAMS = {
    "Ms": [32, 48, 64],
    "M_betas": [48, 64, 96],  # 移除128（M=32时会失败）
    "gammas": [4, 8, 12],  # 移除24（build超时太多，收益不大）
}

# YFCC特殊配置（Recall低，可能需要更大参数，但也可以接受低Recall）
YFCC_PARAMS = {
    "Ms": [32, 48, 64],
    "M_betas": [48, 64, 96],
    "gammas": [4, 8, 12],  # 移除24
}

DATASETS_CONFIG = {
    "yfcc": {
        "N": 1000000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/yfcc",
        "scenarios": ["equal", "or", "and"],  # 所有数据集都支持3个场景
        **YFCC_PARAMS,
    },
    "LAION1M": {
        "N": 1000448,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/LAION1M",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "tripclick": {
        "N": 1055976,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/tripclick",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "ytb_audio": {
        "N": 5000000,  # 5M向量（128维）
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_audio",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "ytb_video": {
        "N": 1000000,  # 1M向量
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/ytb_video",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    # --- V2 validation datasets ---
    "synth_192d": {
        "N": 800000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_192d",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "synth_512d": {
        "N": 800000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_512d",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "synth_768d_hc": {
        "N": 800000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/synth_768d_hc",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "yahoo800k": {
        "N": 800000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/yahoo800k",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
    "dbpedia560k": {
        "N": 560000,
        "data_dir": "/home/remote/u7905817/benchmarks/datasets/discrete/dbpedia560k",
        "scenarios": ["equal", "or", "and"],
        **DEFAULT_PARAMS,
    },
}

# 全局常量
K = 10  # Recall@10
MIN_INDEX_SIZE_MB = 50  # 最小合法索引大小（用于检测损坏）
AUTO_CLEANUP = True  # 搜索完成后自动删除索引文件

# ==================== 辅助函数 ====================

def get_label_file(path):
    """V2 数据集标签从 0 开始，部分方法用 0 做哨兵。
    如果存在 _1based 版本就自动使用。"""
    base, ext = os.path.splitext(path)
    onebased = f"{base}_1based{ext}"
    if os.path.exists(onebased):
        return onebased
    return path

def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_progress(progress_file):
    """加载进度

    返回格式: {
        (M, M_beta, gamma, scenario): {
            'build_done': True/False,
            'search_done': True/False,
            'build_time': 1234.5,
            'index_size': 1500.0,
            'recall': 0.92,
            'qps': 1200.0
        }
    }
    """
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                # 兼容两种格式：旧格式（带progress包装）和新格式（直接数据）
                if 'progress' in data:
                    progress_data = data['progress']
                else:
                    progress_data = data
                # 转换key从JSON字符串回tuple
                return {tuple(json.loads(k)): v for k, v in progress_data.items()}
        except Exception as e:
            log(f"⚠️  加载进度文件失败: {e}")
    return {}

def save_progress(progress_file, progress):
    """保存进度（带错误处理和重试）"""
    output_dir = os.path.dirname(progress_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 转换key从tuple到JSON字符串
    serializable = {json.dumps(k): v for k, v in progress.items()}

    data = {
        'progress': serializable,
        'last_update': datetime.now().isoformat(),
        'total_tasks': len(progress),
        'completed_builds': sum(1 for v in progress.values() if v.get('build_done')),
        'completed_searches': sum(1 for v in progress.values() if v.get('search_done'))
    }

    # 尝试保存，失败时警告但不中断
    for attempt in range(3):
        try:
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
            return  # 成功保存
        except Exception as e:
            if attempt < 2:
                log(f"⚠️  保存进度失败（尝试 {attempt+1}/3）: {e}")
                time.sleep(1)  # 等待1秒后重试
            else:
                log(f"❌ 保存进度失败（已尝试3次）: {e}")
                log(f"⚠️  警告：进度可能未保存，重启后可能重复执行部分任务")

def init_summary_file(summary_file):
    """初始化汇总CSV文件"""
    if not os.path.exists(summary_file):
        output_dir = os.path.dirname(summary_file)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'M', 'M_beta', 'gamma', 'scenario',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'qps_no_filter',
                'status', 'timestamp'
            ])

def append_to_summary(summary_file, M, M_beta, gamma, scenario, metrics):
    """追加结果到CSV"""
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            M, M_beta, gamma, scenario,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('qps_no_filter', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_path):
    """获取索引文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(index_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def parse_search_csv(csv_file):
    """从CSV文件提取最佳性能指标"""
    try:
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
                    'recall': best_recall,
                    'qps': best_qps,
                    'qps_no_filter': best_qps_no_filter
                }
    except Exception as e:
        log(f"⚠️  解析CSV失败 {csv_file}: {e}")
    return None

def cleanup_index(index_path):
    """删除索引文件（节省空间）

    注意：ACORN build程序会同时生成gamma=1的副产品文件，也需要清理
    """
    if not AUTO_CLEANUP:
        return

    total_freed = 0

    # 删除主索引文件
    if os.path.exists(index_path):
        try:
            size_mb = get_index_size(index_path)
            os.remove(index_path)
            total_freed += size_mb
            log(f"🗑️  已删除索引文件（释放 {size_mb:.1f}MB）")
        except Exception as e:
            log(f"⚠️  删除索引失败: {e}")

    # 删除gamma=1副产品（ACORN build会自动生成）
    # 从 hybrid_M=32_Mb=48_gamma=4.json 提取出 hybrid_M=32_Mb=48_gamma=1.json
    if '_gamma=' in index_path and not index_path.endswith('_gamma=1.json'):
        gamma1_path = index_path.rsplit('_gamma=', 1)[0] + '_gamma=1.json'
        if os.path.exists(gamma1_path):
            try:
                size_mb = get_index_size(gamma1_path)
                os.remove(gamma1_path)
                total_freed += size_mb
                log(f"🗑️  已删除gamma=1副产品（释放 {size_mb:.1f}MB）")
            except Exception as e:
                log(f"⚠️  删除gamma=1副产品失败: {e}")

# ==================== 主流程 ====================

def build_index(dataset, M, M_beta, gamma, config, output_base, progress, progress_file):
    """构建索引"""
    key = (M, M_beta, gamma, None)  # scenario=None表示构建阶段

    indices_base = f"{output_base}/indices"
    dataset_dir = f"{indices_base}/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)

    index_path = f"{dataset_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

    # 检查是否已完成（只看progress记录，不检查文件是否存在）
    if key in progress:
        if progress[key].get('build_done'):
            index_size = progress[key].get('index_size', 0)
            build_time = progress[key].get('build_time', 0)
            log(f"⏭️  跳过构建（已完成）: M={M}, M_beta={M_beta}, gamma={gamma}, {index_size:.1f}MB")
            return True, build_time, index_size
        elif progress[key].get('build_done') == False:
            # 已失败过，检查失败原因
            failure_reason = progress[key].get('failure_reason', '')
            if failure_reason == 'disk_quota_exceeded':
                log(f"⚠️  之前因磁盘配额失败，重试构建: M={M}, M_beta={M_beta}, gamma={gamma}")
            elif failure_reason in ['timeout', 'error']:
                log(f"⏭️  跳过构建（之前失败: {failure_reason}）: M={M}, M_beta={M_beta}, gamma={gamma}")
                return False, 0, 0

    log(f"🔨 构建索引: M={M}, M_beta={M_beta}, gamma={gamma}")

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/build_acorn_index',
        str(N), str(gamma),
        f"{data_dir}/{dataset}_base.fvecs",
        str(M), str(M_beta),
        indices_base, dataset
    ]

    log_file = f"{dataset_dir}/M={M}_Mb={M_beta}_gamma={gamma}_build.log"

    start = time.time()
    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=18000)  # 5小时

        build_time = time.time() - start
        index_size = get_index_size(index_path)

        if index_size < MIN_INDEX_SIZE_MB:
            log(f"❌ 构建失败: 索引文件太小 ({index_size:.1f}MB < {MIN_INDEX_SIZE_MB}MB)")
            return False, 0, 0

        log(f"✅ 构建成功: {build_time:.1f}秒, {index_size:.1f}MB")

        # 更新进度
        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = True
        progress[key]['build_time'] = build_time
        progress[key]['index_size'] = index_size
        save_progress(progress_file, progress)

        return True, build_time, index_size

    except subprocess.TimeoutExpired:
        log(f"❌ 构建超时（3小时）")
        # 删除可能生成的不完整索引文件
        if os.path.exists(index_path):
            try:
                size = get_index_size(index_path)
                os.remove(index_path)
                log(f"🗑️  已删除不完整索引 ({size:.1f}MB)")
            except:
                pass
        # 记录失败原因
        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = False
        progress[key]['failure_reason'] = 'timeout'
        progress[key]['failure_time'] = datetime.now().isoformat()
        save_progress(progress_file, progress)
        return False, 0, 0
    except Exception as e:
        error_msg = str(e)
        log(f"❌ 构建失败: {error_msg}")
        # 删除可能生成的不完整索引文件
        if os.path.exists(index_path):
            try:
                size = get_index_size(index_path)
                os.remove(index_path)
                log(f"🗑️  已删除不完整索引 ({size:.1f}MB)")
            except:
                pass
        # 记录失败原因
        if key not in progress:
            progress[key] = {}
        progress[key]['build_done'] = False
        # 判断失败类型
        if 'quota' in error_msg.lower() or 'space' in error_msg.lower():
            progress[key]['failure_reason'] = 'disk_quota_exceeded'
        else:
            progress[key]['failure_reason'] = 'error'
        progress[key]['failure_time'] = datetime.now().isoformat()
        progress[key]['error_message'] = error_msg[:200]  # 只保存前200字符
        save_progress(progress_file, progress)
        return False, 0, 0

def search_index(dataset, M, M_beta, gamma, scenario, config, output_base, progress, progress_file, summary_file, indices_base_override=None):
    """搜索测试

    Args:
        indices_base_override: 可选，指定索引文件所在目录（用于gamma=1副产品搜索，索引在主目录而非gamma1目录）
    """
    key = (M, M_beta, gamma, scenario)

    # 检查是否已完成或失败过
    if key in progress:
        if progress[key].get('search_done'):
            recall = progress[key].get('recall', 0)
            log(f"⏭️  跳过搜索（已完成）: scenario={scenario}, Recall={recall:.4f}")
            return True
        elif progress[key].get('search_done') == False:
            # 已失败过，检查失败原因
            failure_reason = progress[key].get('failure_reason', '')
            if failure_reason == 'disk_quota_exceeded':
                log(f"⚠️  之前因磁盘配额失败，重试搜索: scenario={scenario}")
            elif failure_reason in ['timeout', 'error', 'parse_error']:
                log(f"⏭️  跳过搜索（之前失败: {failure_reason}）: scenario={scenario}")
                return False

    # 检查索引文件是否存在（可能被自动清理了）
    indices_base = indices_base_override if indices_base_override else f"{output_base}/indices"
    dataset_dir = f"{indices_base}/{dataset}"
    index_path = f"{dataset_dir}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

    if not os.path.exists(index_path):
        log(f"⚠️  索引文件不存在（可能已被清理），需要重新构建")
        return False  # 返回False表示需要重新构建

    log(f"🔍 搜索测试: scenario={scenario}")

    results_dir = f"{output_base}/results/{dataset}/{scenario}"
    os.makedirs(results_dir, exist_ok=True)

    data_dir = config['data_dir']
    N = config['N']

    cmd = [
        '../build/demos/search_acorn_index',
        str(N), str(gamma), dataset,
        str(M), str(M_beta), indices_base, scenario, results_dir,
        f"{data_dir}/{dataset}_base.fvecs",
        get_label_file(f"{data_dir}/label_base.txt"),
        f"{data_dir}/{dataset}_query_{scenario}.fvecs",
        get_label_file(f"{data_dir}/{dataset}_query_{scenario}.txt"),
        f"{data_dir}/{dataset}_gt_{scenario}.txt",
        str(K)
    ]

    log_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_search.log"

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=1800)

        # 解析结果
        csv_file = f"{results_dir}/M={M}_M_beta={M_beta}_gamma={gamma}_result.csv"
        metrics = parse_search_csv(csv_file)

        if metrics:
            log(f"✅ 搜索成功: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")

            # 合并构建信息
            build_key = (M, M_beta, gamma, None)
            if build_key in progress:
                metrics['build_time'] = progress[build_key].get('build_time', 0)
                metrics['index_size'] = progress[build_key].get('index_size', 0)

            metrics['status'] = 'success'

            # 保存到CSV
            append_to_summary(summary_file, M, M_beta, gamma, scenario, metrics)

            # 更新进度
            if key not in progress:
                progress[key] = {}
            progress[key]['search_done'] = True
            progress[key]['recall'] = metrics['recall']
            progress[key]['qps'] = metrics['qps']
            save_progress(progress_file, progress)

            return True
        else:
            log(f"❌ 搜索失败: 无法解析结果")
            # 记录失败
            if key not in progress:
                progress[key] = {}
            progress[key]['search_done'] = False
            progress[key]['failure_reason'] = 'parse_error'
            progress[key]['failure_time'] = datetime.now().isoformat()
            save_progress(progress_file, progress)
            return False

    except Exception as e:
        error_msg = str(e)
        log(f"❌ 搜索失败: {error_msg}")
        # 记录失败原因
        if key not in progress:
            progress[key] = {}
        progress[key]['search_done'] = False
        # 判断失败类型
        if 'quota' in error_msg.lower() or 'space' in error_msg.lower():
            progress[key]['failure_reason'] = 'disk_quota_exceeded'
        elif 'timeout' in error_msg.lower():
            progress[key]['failure_reason'] = 'timeout'
        else:
            progress[key]['failure_reason'] = 'error'
        progress[key]['failure_time'] = datetime.now().isoformat()
        progress[key]['error_message'] = error_msg[:200]
        save_progress(progress_file, progress)
        return False

def cleanup_orphaned_indices(indices_dir):
    """启动时清理残留的索引文件（防止之前崩溃留下的文件占用空间）"""
    if not os.path.exists(indices_dir):
        return

    cleaned = 0
    cleaned_size = 0
    for filename in os.listdir(indices_dir):
        if filename.endswith('.json') and filename.startswith('hybrid_'):
            filepath = os.path.join(indices_dir, filename)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                os.remove(filepath)
                cleaned += 1
                cleaned_size += size_mb
                log(f"🗑️  清理残留索引: {filename} ({size_mb:.1f}MB)")
            except Exception as e:
                log(f"⚠️  清理失败: {filename}: {e}")

    if cleaned > 0:
        log(f"✅ 共清理 {cleaned} 个残留索引，释放 {cleaned_size:.1f}MB")

def run_dataset(dataset, config, output_base):
    """运行单个数据集的完整参数搜索"""

    # 启动时清理残留索引文件
    indices_dir = f"{output_base}/indices/{dataset}"
    cleanup_orphaned_indices(indices_dir)

    log("=" * 70)
    log(f"数据集: {dataset.upper()}")
    log(f"数据规模: N={config['N']:,}")
    log(f"测试场景: {', '.join(config['scenarios'])}")
    log(f"参数范围: M={config['Ms']}, M_beta={config['M_betas']}, gamma={config['gammas']}")
    log("=" * 70)

    progress_file = f"{output_base}/progress.json"
    summary_file = f"{output_base}/results/{dataset}/summary.csv"

    # 加载进度
    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    # 生成所有任务
    Ms = config['Ms']
    M_betas = config['M_betas']
    gammas = config['gammas']
    scenarios = config['scenarios']

    total_tasks = len(Ms) * len(M_betas) * len(gammas) * len(scenarios)
    log(f"总任务数: {total_tasks} (构建: {len(Ms) * len(M_betas) * len(gammas)}, 搜索: {total_tasks})")

    # 统计已完成任务
    done_builds = sum(1 for k, v in progress.items() if k[3] is None and v.get('build_done'))
    done_searches = sum(1 for k, v in progress.items() if k[3] is not None and v.get('search_done'))
    if done_builds > 0 or done_searches > 0:
        log(f"📌 已完成: {done_builds} 构建, {done_searches} 搜索（会自动跳过）")
    else:
        log(f"🆕 从头开始")

    completed_builds = 0
    completed_searches = 0
    failed_tasks = []

    # 遍历所有参数组合
    for M_idx, M in enumerate(Ms):
        for M_beta_idx, M_beta in enumerate(M_betas):
            # 检查约束
            if M_beta < M:
                continue

            for gamma_idx, gamma in enumerate(gammas):
                # 检查约束: M_beta <= 2*M*gamma
                if M_beta > 2 * M * gamma:
                    continue

                log("")
                log("=" * 70)
                log(f"参数组合: M={M}, M_beta={M_beta}, gamma={gamma}")
                log("=" * 70)

                # 构建索引
                build_success, build_time, index_size = build_index(
                    dataset, M, M_beta, gamma, config, output_base, progress, progress_file
                )

                if build_success:
                    completed_builds += 1

                    # 获取索引路径（用于后续清理）
                    index_path = f"{output_base}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"

                    # 对每个场景进行搜索
                    all_scenarios_success = True
                    for scenario_idx, scenario in enumerate(scenarios):
                        # search_index 内部会检查是否已完成并跳过
                        search_success = search_index(
                            dataset, M, M_beta, gamma, scenario, config,
                            output_base, progress, progress_file, summary_file
                        )

                        if search_success:
                            completed_searches += 1
                        elif not os.path.exists(index_path):
                            # 索引文件不存在，需要重新构建
                            log(f"⚠️  索引已被清理，重新构建后继续搜索")
                            # 标记build_done=False以触发重新构建
                            build_key = (M, M_beta, gamma, None)
                            if build_key in progress:
                                progress[build_key]['build_done'] = False
                                save_progress(progress_file, progress)

                            # 重新构建
                            rebuild_success, _, _ = build_index(
                                dataset, M, M_beta, gamma, config, output_base, progress, progress_file
                            )

                            if rebuild_success:
                                # 重新搜索
                                search_success = search_index(
                                    dataset, M, M_beta, gamma, scenario, config,
                                    output_base, progress, progress_file, summary_file
                                )
                                if search_success:
                                    completed_searches += 1
                                else:
                                    all_scenarios_success = False
                                    failed_tasks.append((M, M_beta, gamma, scenario))
                            else:
                                all_scenarios_success = False
                                failed_tasks.append((M, M_beta, gamma, scenario))
                        else:
                            all_scenarios_success = False
                            failed_tasks.append((M, M_beta, gamma, scenario))

                    # ========== 顺便用 gamma=1 副产品索引也跑搜索 ==========
                    # ACORN build 会自动生成 gamma=1 索引，我们可以"免费"获取这些数据
                    if gamma != 1:
                        gamma1_index_path = f"{output_base}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma=1.json"
                        if os.path.exists(gamma1_index_path):
                            log("")
                            log(f"🎁 发现 gamma=1 副产品索引，顺便搜索...")

                            # gamma=1 的结果保存到单独的文件夹
                            gamma1_output_base = f"{output_base}_gamma1"
                            gamma1_progress_file = f"{gamma1_output_base}/progress.json"
                            gamma1_summary_file = f"{gamma1_output_base}/results/{dataset}/summary.csv"

                            # 加载 gamma=1 的进度
                            gamma1_progress = load_progress(gamma1_progress_file)
                            init_summary_file(gamma1_summary_file)

                            # 记录 gamma=1 的构建信息（复用主索引的构建时间）
                            gamma1_build_key = (M, M_beta, 1, None)
                            if gamma1_build_key not in gamma1_progress:
                                gamma1_progress[gamma1_build_key] = {}
                            gamma1_progress[gamma1_build_key]['build_done'] = True
                            gamma1_progress[gamma1_build_key]['build_time'] = build_time
                            gamma1_progress[gamma1_build_key]['index_size'] = get_index_size(gamma1_index_path)
                            save_progress(gamma1_progress_file, gamma1_progress)

                            # 对 gamma=1 索引运行搜索
                            # 注意：gamma=1 索引在主目录 output_base/indices，不在 gamma1_output_base/indices
                            main_indices_base = f"{output_base}/indices"
                            for scenario in scenarios:
                                search_key = (M, M_beta, 1, scenario)
                                if search_key in gamma1_progress and gamma1_progress[search_key].get('search_done'):
                                    log(f"⏭️  跳过 gamma=1 搜索（已完成）: scenario={scenario}")
                                    continue

                                search_success = search_index(
                                    dataset, M, M_beta, 1, scenario, config,
                                    gamma1_output_base, gamma1_progress, gamma1_progress_file, gamma1_summary_file,
                                    indices_base_override=main_indices_base  # 索引在主目录
                                )
                                if search_success:
                                    log(f"✅ gamma=1 搜索成功: scenario={scenario}")

                            log(f"🎁 gamma=1 副产品搜索完成")

                    # 所有场景尝试完毕后，无论成功失败都删除索引文件（节省空间）
                    cleanup_index(index_path)
                else:
                    log(f"⚠️  构建失败，跳过所有搜索")
                    # 构建失败也要清理可能存在的不完整索引
                    index_path = f"{output_base}/indices/{dataset}/hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
                    cleanup_index(index_path)
                    for scenario in scenarios:
                        failed_tasks.append((M, M_beta, gamma, scenario))

    # 打印摘要
    log("")
    log("=" * 70)
    log(f"数据集 {dataset.upper()} 完成!")
    log("=" * 70)
    log(f"完成构建: {completed_builds}/{len(Ms) * len(M_betas) * len(gammas)}")
    log(f"完成搜索: {completed_searches}/{total_tasks}")

    if failed_tasks:
        log(f"失败任务: {len(failed_tasks)}")
        for task in failed_tasks[:10]:  # 只显示前10个
            log(f"  - M={task[0]}, M_beta={task[1]}, gamma={task[2]}, scenario={task[3]}")

    log(f"结果文件: {summary_file}")
    log("")

# ==================== 主入口 ====================

def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        datasets_to_run = [ds for ds in sys.argv[1:] if ds in DATASETS_CONFIG]
        if not datasets_to_run:
            log(f"❌ 无效的数据集: {sys.argv[1:]}")
            log(f"可用数据集: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)
    else:
        datasets_to_run = list(DATASETS_CONFIG.keys())

    log("=" * 70)
    log("ACORN参数自动搜索 V2")
    log("=" * 70)
    log(f"运行数据集: {', '.join(datasets_to_run)}")
    log(f"自动清理索引: {'是' if AUTO_CLEANUP else '否'}")
    log("")
    log("💡 建议:")
    log("  - 使用screen避免断线: screen -S acorn_search")
    log("  - 查看进度: tail -f search.log")
    log("  - 断开screen: Ctrl+A, D")
    log("  - 重连screen: screen -r acorn_search")
    log("=" * 70)
    log("")

    start_time = time.time()

    # 找到最后在处理的数据集（跳过已完成的数据集）
    start_dataset_idx = 0
    for idx, dataset in enumerate(datasets_to_run):
        output_base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_{dataset}"
        progress_file = f"{output_base}/progress.json"

        if os.path.exists(progress_file):
            progress = load_progress(progress_file)
            if progress:
                # 有进度记录，说明这个数据集已开始处理
                start_dataset_idx = idx
        else:
            # 没有进度文件，从这个数据集开始
            break

    if start_dataset_idx > 0:
        log(f"📌 检测到断点：跳过前 {start_dataset_idx} 个数据集，从 {datasets_to_run[start_dataset_idx]} 继续")
        log("")

    for idx, dataset in enumerate(datasets_to_run):
        # 跳过已完成的数据集
        if idx < start_dataset_idx:
            log(f"⏭️  跳过数据集: {dataset.upper()}（已处理）")
            continue

        config = DATASETS_CONFIG[dataset]
        output_base = f"/home/remote/u7905817/benchmarks/discrete/ACORN/data/param_search_{dataset}"

        try:
            run_dataset(dataset, config, output_base)
        except KeyboardInterrupt:
            log("⚠️  用户中断，保存进度...")
            break
        except Exception as e:
            log(f"❌ 数据集 {dataset} 失败: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    log("=" * 70)
    log(f"✅ 所有任务完成！总耗时: {total_time/3600:.2f}小时")
    log("=" * 70)

if __name__ == "__main__":
    main()
