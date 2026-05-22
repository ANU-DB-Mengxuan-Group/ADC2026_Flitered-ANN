#!/usr/bin/env python3
"""查看 ACORN V2 实验进度和结果"""

import json
import sys
from pathlib import Path

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
ACORN_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def check_dataset(dataset):
    progress_file = ACORN_DATA_DIR / f"param_search_{dataset}" / "progress.json"

    if not progress_file.exists():
        print(f"\n[{dataset}] 未开始")
        return

    with open(progress_file) as f:
        data = json.load(f)

    prog = data.get('progress', {})

    # 统计 builds
    builds_done = []
    builds_timeout = []
    builds_pending = []

    for k, v in prog.items():
        if 'null' not in k or not isinstance(v, dict):
            continue
        if v.get('build_done'):
            builds_done.append((k, v))
        elif v.get('failure_reason') == 'timeout':
            builds_timeout.append((k, v))
        else:
            builds_pending.append(k)

    # 统计 searches
    searches = []
    for k, v in prog.items():
        if 'null' in k or not isinstance(v, dict):
            continue
        if v.get('search_done'):
            searches.append((k, v))

    print(f"\n[{dataset}]")
    print(f"  Builds: {len(builds_done)} done, {len(builds_timeout)} timeout")
    print(f"  Searches: {len(searches)} done")

    # Best recall per scenario
    by_scenario = {}
    for k, v in searches:
        try:
            parts = eval(k.replace('null', 'None'))
            scenario = parts[3]
            recall = v.get('recall', 0)
            qps = v.get('qps', 0)
            config = f"M={parts[0]}, M_beta={parts[1]}, gamma={parts[2]}"

            if scenario not in by_scenario or recall > by_scenario[scenario]['recall']:
                by_scenario[scenario] = {'recall': recall, 'qps': qps, 'config': config}
        except:
            continue

    if by_scenario:
        print("  Best recall per scenario:")
        for s in ['and', 'or', 'equal']:
            if s in by_scenario:
                info = by_scenario[s]
                print(f"    {s}: recall={info['recall']:.4f}, qps={info['qps']:.1f} ({info['config']})")

    # 列出 timeout 的配置
    if builds_timeout:
        print(f"  Timeout configs ({len(builds_timeout)}):")
        for k, v in builds_timeout[:5]:
            parts = eval(k.replace('null', 'None'))
            print(f"    M={parts[0]}, M_beta={parts[1]}, gamma={parts[2]}")
        if len(builds_timeout) > 5:
            print(f"    ... and {len(builds_timeout) - 5} more")

def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else V2_DATASETS

    print("=" * 50)
    print("ACORN V2 实验进度")
    print("=" * 50)

    for dataset in datasets:
        if dataset in V2_DATASETS:
            check_dataset(dataset)

    print()

if __name__ == "__main__":
    main()
