#!/usr/bin/env python3
"""
Round 4: synth_768d_hc + yahoo800k 串行跑

- synth_768d_hc: full build (~24h) + search
- yahoo800k:     full build (~25h) + search

总预估 50h 左右, 在 72h SLURM allocation 内安全.

用法 (在 weirdo, conda activate benchmark, screen 内):
    cd ~/benchmarks/discrete
    python analysis/run_postfilter_round4.py

要求:
- 已 conda activate benchmark
- 已 export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
- 跑在 db4ai 节点 (非登录节点)
- 36h build timeout 已 push (commit 57fe973+)
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DATASETS = ["synth_768d_hc", "yahoo800k"]
REPO_ROOT = Path("~/benchmarks/discrete").expanduser()
WORKER_SCRIPT = REPO_ROOT / "faiss/bash/auto_postfilter_hnsw.py"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def precheck():
    """启动前最低限度环境检查."""
    if not WORKER_SCRIPT.exists():
        log(f"❌ worker script 不存在: {WORKER_SCRIPT}")
        sys.exit(1)

    hostname = subprocess.check_output(["hostname"]).decode().strip()
    if "login" in hostname:
        log(f"❌ 登录节点 {hostname}, 不应跑长任务. 先 srun 进 db4ai 节点")
        sys.exit(1)
    log(f"  ✅ 节点 {hostname}")

    if "CONDA_DEFAULT_ENV" not in os.environ or os.environ["CONDA_DEFAULT_ENV"] != "benchmark":
        log(f"❌ conda env 不是 benchmark (现: {os.environ.get('CONDA_DEFAULT_ENV', 'none')})")
        log("    先 source ~/miniconda3/etc/profile.d/conda.sh && conda activate benchmark")
        sys.exit(1)
    log(f"  ✅ conda env = benchmark")

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    cp = os.environ.get("CONDA_PREFIX", "")
    if cp and not ld.startswith(f"{cp}/lib"):
        log(f"⚠️  LD_LIBRARY_PATH 头部不是 $CONDA_PREFIX/lib")
        log(f"    现在: {ld[:120]}")
        log(f"    建议先: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
        log("    继续可能失败 (MKL / GLIBCXX 找不到)")
    else:
        log(f"  ✅ LD_LIBRARY_PATH 头 = $CONDA_PREFIX/lib")


def run_one(dataset: str) -> int:
    log("=" * 60)
    log(f"START {dataset}")
    log("=" * 60)
    start = time.time()

    cmd = ["python", str(WORKER_SCRIPT), dataset]
    log(f"cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed_h = (time.time() - start) / 3600

    log(f"END   {dataset} (exit={result.returncode}, took {elapsed_h:.2f}h)")
    log("")
    return result.returncode


def main():
    log(f"==== Round 4 starting ====")
    log(f"Datasets: {DATASETS}")
    log(f"Worker:   {WORKER_SCRIPT}")
    log("")

    precheck()
    log("")

    failed = []
    for ds in DATASETS:
        rc = run_one(ds)
        if rc != 0:
            log(f"⚠️  {ds} exit code {rc} (build/search 中可能有失败, 但脚本继续)")
            failed.append(ds)

    log("=" * 60)
    log(f"==== Round 4 done ====")
    if failed:
        log(f"⚠️  这些 dataset 的 worker 退出非 0: {failed}")
        log(f"    去看 progress JSON 和 log 决定是否重试")
    else:
        log(f"  全部 dataset worker 正常退出")
    log("=" * 60)


if __name__ == "__main__":
    main()
