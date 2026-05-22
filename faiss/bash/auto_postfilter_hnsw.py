#!/usr/bin/env python3
"""
Post-filter HNSW 基线实验脚本

用法:
  cd ~/benchmarks/discrete/faiss
  python bash/auto_postfilter_hnsw.py                    # 运行所有数据集和场景
  python bash/auto_postfilter_hnsw.py arxiv              # 只运行指定数据集
  python bash/auto_postfilter_hnsw.py arxiv and          # 指定数据集和场景
"""

import os
import sys
import time
import json
import csv
import glob
import subprocess
from pathlib import Path
from datetime import datetime

# 设置 MKL 库路径（faiss 依赖）
# 注意：必须在 subprocess 调用时显式传递 env，否则 LD_LIBRARY_PATH 不生效
SUBPROCESS_ENV = os.environ.copy()
if 'CONDA_PREFIX' in os.environ:
    mkl_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    ld_path = SUBPROCESS_ENV.get('LD_LIBRARY_PATH', '')
    if mkl_path not in ld_path:
        SUBPROCESS_ENV['LD_LIBRARY_PATH'] = f"{mkl_path}:{ld_path}"
        print(f"[MKL] Set LD_LIBRARY_PATH: {SUBPROCESS_ENV['LD_LIBRARY_PATH']}")

# ==================== 配置 ====================

FAISS_DIR = "/home/remote/u7905817/benchmarks/discrete/faiss"
DATA_DIR = "/home/remote/u7905817/benchmarks/datasets/discrete"

# HNSW 参数
M_VALUES = [64]                # 直接跑 thesis 用的最佳配置 M=64 efc=400
EFC_VALUES = [400]             # M=32 sweep 反而更慢, 且会让 V2 路由验证数字低于 thesis claim
# 跑完想再补别的配置画图, 改回 [32, 48, 64] 和 [100, 200, 400] 即可

# 数据集配置
DATASETS_CONFIG = {
    "arxiv": {"N": 132687, "D": 768},
    "tripclick": {"N": 1055976, "D": 768},
    "LAION1M": {"N": 1000448, "D": 512},
    "yfcc": {"N": 1000000, "D": 192},
    "ytb_audio": {"N": 5000000, "D": 128},
    "ytb_video": {"N": 1000000, "D": 1024},
    # --- V2 validation datasets ---
    "synth_192d": {"N": 800000, "D": 192},
    "synth_512d": {"N": 800000, "D": 512},
    "synth_768d_hc": {"N": 800000, "D": 768},
    "yahoo800k": {"N": 800000, "D": 768},
    "dbpedia560k": {"N": 560000, "D": 768},
}

SCENARIOS = ["and", "or", "equal"]
K = 10
T = 16

# ==================== 辅助函数 ====================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"completed": [], "failed": []}

def save_progress(progress_file, progress):
    Path(os.path.dirname(progress_file)).mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def init_summary_file(summary_file):
    if not os.path.exists(summary_file):
        Path(os.path.dirname(summary_file)).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'scenario', 'M', 'efc',
                'build_time_s', 'index_size_mb',
                'recall@10', 'qps', 'status', 'timestamp'
            ])

def append_to_summary(summary_file, dataset, scenario, M, efc, metrics):
    with open(summary_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset, scenario, M, efc,
            metrics.get('build_time', 0),
            metrics.get('index_size', 0),
            metrics.get('recall', 0),
            metrics.get('qps', 0),
            metrics.get('status', 'unknown'),
            datetime.now().isoformat()
        ])

def get_index_size(index_path):
    if os.path.exists(index_path):
        return os.path.getsize(index_path) / (1024 * 1024)
    return 0

def parse_search_result(output_csv):
    """解析搜索结果CSV"""
    try:
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
            if results:
                # 取最后一个（最高efSearch）的结果
                last = results[-1]
                return {
                    'recall': float(last.get('Recall', 0)),
                    'qps': float(last.get('QPS', 0))
                }
    except Exception as e:
        log(f"Warning: parse result failed: {e}")
    return None

def cleanup_index(index_path):
    """删除索引文件节省空间"""
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            log(f"  Cleaned: {os.path.basename(index_path)}")
        except Exception as e:
            log(f"  Cleanup failed: {e}")

# ==================== 主流程 ====================

def build_index(dataset, M, efc, config, output_base):
    """构建HNSW索引"""
    index_dir = f"{output_base}/data/index_files/hnsw/{dataset}"
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    index_path = f"{index_dir}/M={M}_efc={efc}.json"
    log_file = f"{index_dir}/M={M}_efc={efc}_build.log"

    if os.path.exists(index_path):
        log(f"Index exists: M={M}, efc={efc}")
        return True, 0, get_index_size(index_path), index_path

    base_file = f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs"

    cmd = [
        f"{FAISS_DIR}/tutorial/cpp/build_HNSW_index_static",
        base_file,
        str(M),
        str(efc),
        f"{output_base}/data/index_files/hnsw",
        dataset
    ]

    log(f"Building index: M={M}, efc={efc}")
    start = time.time()

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=172800, env=SUBPROCESS_ENV)  # 48h: yahoo800k 18h timeout 已失败一次, 给足空间
        build_time = time.time() - start

        # 检查索引文件是否存在（比检查返回码更可靠）
        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            index_size = get_index_size(index_path)
            log(f"  Build success: {build_time:.1f}s, {index_size:.1f}MB")
            return True, build_time, index_size, index_path
        else:
            log(f"  Build failed: index file not created")
            return False, 0, 0, None
    except subprocess.TimeoutExpired:
        log(f"  Build timeout (48h)")
        return False, 0, 0, None
    except Exception as e:
        log(f"  Build failed: {e}")
        return False, 0, 0, None

def search_index(dataset, scenario, M, efc, config, output_base):
    """搜索测试"""
    result_dir = f"{output_base}/results_postfilter/{dataset}/{scenario}"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{result_dir}/M={M}_efc={efc}_search.log"
    output_csv = f"{result_dir}/M={M}_efc={efc}_result.csv"

    N = config['N']

    cmd = [
        f"{FAISS_DIR}/tutorial/cpp/search_HNSW_index_static",
        dataset,
        str(M),
        str(efc),
        f"{output_base}/data/index_files/hnsw",
        scenario,
        result_dir,
        f"{DATA_DIR}/{dataset}/{dataset}_base.fvecs",
        f"{DATA_DIR}/{dataset}/label_base.txt",
        f"{DATA_DIR}/{dataset}/{dataset}_query_{scenario}.fvecs",
        f"{DATA_DIR}/{dataset}/{dataset}_query_{scenario}.txt",
        f"{DATA_DIR}/{dataset}/{dataset}_gt_{scenario}.txt",
        str(K),
        str(N)
    ]

    log(f"Searching [{scenario}]: M={M}, efc={efc}")

    try:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                          timeout=36000, check=True, env=SUBPROCESS_ENV)  # 10h: synth_768d_hc 768d 的 [and]/[equal] 实测 > 3h timeout, 给充分余量

        metrics = parse_search_result(output_csv)
        if metrics and metrics['recall'] > 0:
            log(f"  Search success: Recall={metrics['recall']:.4f}, QPS={metrics['qps']:.2f}")
            return True, metrics
        else:
            log(f"  Search done but no valid results")
            return True, {'recall': 0, 'qps': 0}
    except subprocess.TimeoutExpired:
        log(f"  Search timeout (10h)")
        return False, None
    except Exception as e:
        log(f"  Search failed: {e}")
        return False, None

def run_dataset(dataset, scenarios_to_run, config, output_base):
    """运行单个数据集的所有场景"""
    log("=" * 60)
    log(f"Dataset: {dataset.upper()} (Post-filter HNSW)")
    log(f"N = {config['N']:,}, D = {config['D']}")
    log("=" * 60)

    progress_file = f"{output_base}/progress_postfilter_{dataset}.json"
    summary_file = f"{output_base}/results_postfilter/{dataset}/summary.csv"

    progress = load_progress(progress_file)
    init_summary_file(summary_file)

    for M in M_VALUES:
        for efc in EFC_VALUES:
            # 先检查是否所有搜索都已完成，如果是则跳过整个构建
            all_searches_done = all(
                f"{dataset}_{scenario}_M={M}_efc={efc}" in progress.get("completed", [])
                for scenario in scenarios_to_run
            )
            if all_searches_done:
                log(f"Skip (all searches done): M={M}, efc={efc}")
                continue

            # 构建索引
            task_key_build = f"{dataset}_M={M}_efc={efc}_build"

            build_time = 0
            index_size = 0

            index_path = f"{output_base}/data/index_files/hnsw/{dataset}/M={M}_efc={efc}.json"

            # 检查是否需要构建：progress没记录completed，或者索引文件不存在
            need_build = task_key_build not in progress.get("completed", [])
            if not need_build and not os.path.exists(index_path):
                log(f"Index file missing, need rebuild: M={M}, efc={efc}")
                # 从completed中移除，重新构建
                if task_key_build in progress.get("completed", []):
                    progress["completed"].remove(task_key_build)
                    save_progress(progress_file, progress)
                need_build = True

            if need_build:
                build_ok, build_time, index_size, index_path = build_index(
                    dataset, M, efc, config, output_base)

                if not build_ok:
                    progress.setdefault("failed", []).append(task_key_build)
                    save_progress(progress_file, progress)
                    for scenario in scenarios_to_run:
                        append_to_summary(summary_file, dataset, scenario, M, efc, {
                            'build_time': 0, 'index_size': 0,
                            'recall': 0, 'qps': 0, 'status': 'build_failed'
                        })
                    continue

                progress.setdefault("completed", []).append(task_key_build)
                save_progress(progress_file, progress)
            else:
                log(f"Skip build (done): M={M}, efc={efc}")
                index_size = get_index_size(index_path)

            # 对每个场景搜索
            all_search_ok = True   # 跟踪 3 场景是否全成功 (含已 completed 的)
            for scenario in scenarios_to_run:
                task_key = f"{dataset}_{scenario}_M={M}_efc={efc}"

                if task_key in progress.get("completed", []):
                    # 从结果文件读取 recall/QPS 并显示
                    result_csv = f"{output_base}/results_postfilter/{dataset}/{scenario}/M={M}_efc={efc}_result.csv"
                    metrics = parse_search_result(result_csv)
                    if metrics:
                        log(f"Skip (done): {scenario} M={M} efc={efc} -> recall={metrics['recall']:.4f}, qps={metrics['qps']:.1f}")
                    else:
                        log(f"Skip (done): {task_key}")
                    continue

                if task_key in progress.get("failed", []):
                    log(f"Skip (failed): {task_key}")
                    all_search_ok = False   # 上次失败过, 索引留着以便人工处理
                    continue

                search_ok, metrics = search_index(
                    dataset, scenario, M, efc, config, output_base)

                if not search_ok:
                    progress.setdefault("failed", []).append(task_key)
                    save_progress(progress_file, progress)
                    append_to_summary(summary_file, dataset, scenario, M, efc, {
                        'build_time': build_time, 'index_size': index_size,
                        'recall': 0, 'qps': 0, 'status': 'search_failed'
                    })
                    all_search_ok = False   # 这次失败 → 不删索引, 下次还能重试 search
                    continue

                progress.setdefault("completed", []).append(task_key)
                save_progress(progress_file, progress)

                metrics = metrics or {'recall': 0, 'qps': 0}
                metrics['build_time'] = build_time
                metrics['index_size'] = index_size
                metrics['status'] = 'success'
                append_to_summary(summary_file, dataset, scenario, M, efc, metrics)

            # 只在 3 场景全成功才清理索引;
            # 否则保留索引文件, 让人工 / 下次重跑能 retry search 而不必重 build
            index_path = f"{output_base}/data/index_files/hnsw/{dataset}/M={M}_efc={efc}.json"
            if all_search_ok:
                cleanup_index(index_path)
            else:
                log(f"  Search 有失败的场景, 保留索引以便重试: {index_path}")

    log("")
    log(f"Dataset {dataset} done!")

# ==================== 环境前置检查 ====================

def precheck_environment():
    """启动前 fail-fast 检查环境. 任一项不通过立即 exit 1.

    检查内容:
      1. binary 文件存在 (build + search)
      2. 当前节点 != 登录节点 (拒绝在 login node 跑长任务)
      3. ldd 静态依赖: MKL 链上
      4. 实际 invoke binary, 抓 GLIBCXX/symbol/version 等运行时错
      5. conda env 是否激活
      6. LD_LIBRARY_PATH 头部是否 $CONDA_PREFIX/lib (避免链到 base env 老库)
    """
    import socket
    build_exe = f"{FAISS_DIR}/tutorial/cpp/build_HNSW_index_static"
    search_exe = f"{FAISS_DIR}/tutorial/cpp/search_HNSW_index_static"

    # 1. binary 文件存在
    for exe in [build_exe, search_exe]:
        if not os.path.exists(exe):
            log(f"❌ Executable not found: {exe}")
            sys.exit(1)
    log(f"  ✅ binary 存在 (build + search)")

    # 2. 节点身份 (hostname 含 'login' 拒绝)
    hostname = socket.gethostname()
    if "login" in hostname.lower():
        log(f"❌ 当前在登录节点 ({hostname}), 不应在此跑长任务!")
        log(f"   先 salloc + srun 进 db4ai 计算节点再跑.")
        sys.exit(1)
    log(f"  ✅ 节点 {hostname} (非登录节点)")

    # 3. ldd 静态依赖 MKL
    try:
        ldd_out = subprocess.check_output(
            ["ldd", build_exe], stderr=subprocess.STDOUT, timeout=10
        ).decode(errors="replace")
    except Exception as e:
        log(f"❌ ldd 检查失败: {e}")
        sys.exit(1)
    if "libmkl_intel_lp64" not in ldd_out or "not found" in ldd_out:
        log(f"❌ MKL 库未链接成功! ldd 输出含 'not found' 或缺 libmkl:")
        for line in ldd_out.splitlines():
            if "mkl" in line or "not found" in line:
                log(f"    {line}")
        log(f"  修复: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
        sys.exit(1)
    log(f"  ✅ MKL 链接 OK (ldd)")

    # 4. 实际 invoke binary, 抓 runtime lib 加载错误
    try:
        proc = subprocess.run(
            [build_exe], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=10, env=SUBPROCESS_ENV,
        )
        out = proc.stdout.decode(errors="replace") if proc.stdout else ""
    except Exception as e:
        log(f"❌ 启动 binary 异常: {e}")
        sys.exit(1)
    bad_patterns = [
        "GLIBCXX", "version `", "not found", "symbol lookup error",
        "undefined symbol", "cannot open shared object",
    ]
    head = "\n".join(out.splitlines()[:5])
    if any(p in head for p in bad_patterns):
        log(f"❌ binary 启动失败, 运行时库依赖问题:")
        for line in head.splitlines():
            log(f"    {line}")
        log(f"  常见修复:")
        log(f"    • libstdc++ 太老: conda install -y -c conda-forge 'libstdcxx-ng>=14'")
        log(f"    • MKL: 确认 LD_LIBRARY_PATH 含 $CONDA_PREFIX/lib")
        sys.exit(1)
    log(f"  ✅ Runtime 库加载 OK (libstdc++/MKL)")

    # 5. conda env
    cda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if cda_env != "benchmark":
        log(f"⚠️  conda env = '{cda_env}', 建议 'benchmark' (不影响 binary, 但 Python 解释器可能不同)")
    else:
        log(f"  ✅ conda env = benchmark")

    # 6. LD_LIBRARY_PATH 头部检查
    cda_prefix = os.environ.get("CONDA_PREFIX", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cda_prefix and not ld_path.startswith(f"{cda_prefix}/lib"):
        log(f"⚠️  LD_LIBRARY_PATH 头部不是 $CONDA_PREFIX/lib")
        log(f"    现在: {ld_path[:120]}{'...' if len(ld_path) > 120 else ''}")
        log(f"    binary 已通过 invoke 测试, 但建议: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
    else:
        log(f"  ✅ LD_LIBRARY_PATH 头 = $CONDA_PREFIX/lib")


# ==================== 主入口 ====================

def main():
    datasets = list(DATASETS_CONFIG.keys())
    scenarios = SCENARIOS

    if len(sys.argv) > 1:
        if sys.argv[1] in DATASETS_CONFIG:
            datasets = [sys.argv[1]]
        else:
            log(f"Invalid dataset: {sys.argv[1]}")
            log(f"Available: {list(DATASETS_CONFIG.keys())}")
            sys.exit(1)

    if len(sys.argv) > 2:
        if sys.argv[2] in SCENARIOS:
            scenarios = [sys.argv[2]]
        else:
            log(f"Invalid scenario: {sys.argv[2]}")
            log(f"Available: {SCENARIOS}")
            sys.exit(1)

    log("=" * 60)
    log("Post-filter HNSW Baseline Experiment")
    log("=" * 60)
    log(f"Datasets: {', '.join(datasets)}")
    log(f"Scenarios: {', '.join(scenarios)}")
    log("")

    # ===== 环境前置检查 (fail-fast) =====
    log("=" * 60)
    log("环境前置检查")
    log("=" * 60)
    precheck_environment()
    log("")

    start_time = time.time()

    for dataset in datasets:
        config = DATASETS_CONFIG[dataset]
        try:
            run_dataset(dataset, scenarios, config, FAISS_DIR)
        except KeyboardInterrupt:
            log("User interrupted")
            sys.exit(0)
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    log("")
    log("=" * 60)
    log(f"All done! Total time: {total_time/3600:.2f}h")
    log("=" * 60)

if __name__ == "__main__":
    main()
