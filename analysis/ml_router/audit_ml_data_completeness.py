#!/usr/bin/env python3
"""
ML 数据完整性 audit.

跑大型实验前/中/后用来 verify 各方法 × 数据集 × 场景的产出是否齐.
输出:
  - 矩阵: methods × datasets × {aggregate, per-query, features} 是否齐
  - 缺失列表 — 让你立即知道还要补啥

用法:
  cd ~/benchmarks/discrete && python analysis/audit_ml_data_completeness.py
  cd ~/benchmarks/discrete && python analysis/audit_ml_data_completeness.py --datasets synth_192d
"""
import argparse
import os
import glob
from pathlib import Path

ROOT = Path(os.path.expanduser("~/benchmarks/discrete"))
ANALYSIS = ROOT / "analysis"

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
V1_DATASETS = ["arxiv", "yfcc", "LAION1M", "tripclick", "ytb_audio", "ytb_video"]
SCENARIOS = ["and", "or", "equal"]

# 各方法的产出 path 模式
METHOD_PATHS = {
    "UNG": {
        "aggregate": "UNG-dev/results_v2_grid/summary.csv",
        "per_query_glob": "UNG-dev/results_v2_grid/{ds}/{ung_sc}/*result_L*.csv",
        "scenarios_map": {"and": "containment", "or": "overlap", "equal": "equality"},
    },
    "Post-filter": {
        "aggregate_glob": "faiss/results_postfilter/{ds}/{sc}/M=64_efc=400_result.csv",
        "per_query_glob": "faiss/results_postfilter/{ds}/{sc}/M=64_efc=400_ef=*_idx_uint32.bin",
    },
    "SIEVE": {
        "aggregate_glob": "SIEVE/results/sieve_{ds}_*_M*.csv",
        "per_query_csv": "analysis/ml_router/perquery_recall/perquery_recall_v2_sieve.csv",  # global, not per (ds,sc)
    },
    "ACORN": {
        "aggregate_glob": "ACORN/data/param_search_{ds}*/results/{ds}/summary.csv",
        "per_query_glob": None,  # 暂未确认是否输出 per-query
    },
    "Pre-filter": {
        "aggregate_glob": "faiss/results_prefilter/{ds}/{sc}/prefilter_bruteforce_result.csv",
        "per_query_glob": None,  # 当前 brute-force 没保 per-query
    },
    "FilteredVamana": {
        "aggregate_glob": "DiskANN/data_original/results/{ds}/summary.csv",
        "per_query_glob": "DiskANN/data_original/results/{ds}/*_K.bin",
    },
}

DATASET_FEATURES = {
    "metrics_json": "analysis/{ds}_metrics.json",
}


def check_path(pattern, **kwargs):
    """Check if path or glob matches. Returns (exists, count, hint)."""
    path = pattern.format(**kwargs) if pattern else None
    if not path:
        return False, 0, "(no path defined)"
    full = ROOT / path
    if "*" in path:
        matches = glob.glob(str(full))
        return len(matches) > 0, len(matches), f"({len(matches)} match)"
    exists = full.exists() and full.stat().st_size > 0
    return exists, (1 if exists else 0), f"({full.stat().st_size if exists else 0}B)"


def audit_method(method, ds, sc, info):
    """Returns dict: {aggregate: bool, per_query: bool}."""
    out = {}

    # aggregate
    if "aggregate_glob" in info:
        exists, _, _ = check_path(info["aggregate_glob"], ds=ds, sc=sc)
        out["aggregate"] = exists
    elif "aggregate" in info:
        full = ROOT / info["aggregate"]
        out["aggregate"] = full.exists() and full.stat().st_size > 0
    else:
        out["aggregate"] = None

    # per-query
    if info.get("per_query_glob"):
        ung_sc = info.get("scenarios_map", {}).get(sc, sc)
        exists, _, _ = check_path(info["per_query_glob"], ds=ds, sc=sc, ung_sc=ung_sc)
        out["per_query"] = exists
    elif info.get("per_query_csv"):
        full = ROOT / info["per_query_csv"]
        if full.exists():
            # check if this (ds, sc) has rows
            try:
                with open(full) as f:
                    next(f)  # skip header
                    found = any(f"{ds}," in line and f",{sc}," in line for line in f)
                out["per_query"] = found
            except Exception:
                out["per_query"] = False
        else:
            out["per_query"] = False
    else:
        out["per_query"] = None  # not supported by this method

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=V2_DATASETS,
                        help="Datasets to audit (default: V2)")
    args = parser.parse_args()

    datasets = args.datasets
    methods = list(METHOD_PATHS.keys())
    print(f"\nML 数据 audit: {len(methods)} methods × {len(datasets)} datasets × 3 scenarios\n")

    # Per-method × dataset × scenario matrix
    missing = []  # collect missing items
    for method in methods:
        info = METHOD_PATHS[method]
        print(f"=== {method} ===")
        print(f"{'dataset':<15} {'sc':<6} {'agg':<5} {'pq':<5}")
        print("-" * 35)
        for ds in datasets:
            for sc in SCENARIOS:
                r = audit_method(method, ds, sc, info)
                agg = "✅" if r["aggregate"] else ("—" if r["aggregate"] is None else "❌")
                pq = "✅" if r["per_query"] else ("—" if r["per_query"] is None else "❌")
                print(f"{ds:<15} {sc:<6} {agg:<5} {pq:<5}")
                if r["aggregate"] is False:
                    missing.append(f"{method}/{ds}/{sc}: aggregate")
                if r["per_query"] is False:
                    missing.append(f"{method}/{ds}/{sc}: per-query")
        print()

    # Dataset features
    print("=== Dataset features ===")
    print(f"{'dataset':<15} {'metrics':<10}")
    print("-" * 30)
    for ds in datasets:
        ok = (ANALYSIS / f"{ds}_metrics.json").exists()
        print(f"{ds:<15} {'✅' if ok else '❌'}")
        if not ok:
            missing.append(f"dataset_metrics/{ds}")
    print()

    # Summary
    print(f"\n{'=' * 50}")
    print(f"缺失项: {len(missing)}")
    print(f"{'=' * 50}")
    if missing:
        print()
        for m in missing[:30]:
            print(f"  ❌ {m}")
        if len(missing) > 30:
            print(f"  ... 另外 {len(missing) - 30} 项")
    else:
        print("\n✅ 全部齐!")

    print()
    print("Note:")
    print("  ✅ = 文件存在且非空")
    print("  ❌ = 文件缺失")
    print("  — = 该方法不支持 (e.g. Pre-filter 没设计 per-query)")


if __name__ == "__main__":
    main()
