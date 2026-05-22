#!/usr/bin/env python3
"""Run Post-filter per-query timing on all router-used configs.

Reads `analysis/ml_router/router_used_configs.csv`, filters to
method=Post-filter, then for each unique (dataset, M, efc) builds the
HNSW index if missing, and for each (dataset, scenario, M, efc) runs
search_HNSW_index_static with PER_QUERY_CSV env var set.

The binary internally sweeps ef_search = {50,150,250,400,600,800,1200,
1500,2000} and writes one per-query CSV per ef_search value:
  {PER_QUERY_CSV}_M=X_efc=Y_ef=Z.csv

Schema: query_id,latency_us,recall_at_10
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "DATA_DIR", "/home/remote/u7905817/benchmarks/datasets/discrete"))
FAISS_DIR = REPO / "faiss"
BUILD_BIN = FAISS_DIR / "tutorial/cpp/build_HNSW_index_static"
SEARCH_BIN = FAISS_DIR / "tutorial/cpp/search_HNSW_index_static"
INDEX_ROOT = FAISS_DIR / "data/index_files/hnsw"
CONFIGS_CSV = REPO / "analysis/ml_router/router_used_configs.csv"
DEFAULT_OUT_DIR = Path.home() / "benchmarks" / "per_query_results"
K = 10

DATASET_N = {
    "synth_192d": 800000,
    "synth_512d": 800000,
    "synth_768d_hc": 800000,
    "yahoo800k": 800000,
    "dbpedia560k": 560000,
}


def make_env() -> dict:
    """SUBPROCESS_ENV w/ MKL libs path (faiss links against MKL in conda env).

    Mirrors auto_postfilter_hnsw.py:SUBPROCESS_ENV.
    """
    env = os.environ.copy()
    cp = os.environ.get("CONDA_PREFIX")
    if not cp:
        print("[MKL] WARN: CONDA_PREFIX not set — binary may fail to find libmkl. "
              "Run inside `conda activate benchmark` first.")
        return env
    mkl = os.path.join(cp, "lib")
    ld = env.get("LD_LIBRARY_PATH", "")
    if mkl not in ld:
        env["LD_LIBRARY_PATH"] = f"{mkl}:{ld}" if ld else mkl
        print(f"[MKL] Set LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    else:
        print(f"[MKL] LD_LIBRARY_PATH already contains {mkl}")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Where to write per-query CSVs.")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true",
                    help="Don't build missing indices, only search.")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep index files after successful searches "
                         "(default: cleanup to save disk).")
    ap.add_argument("--first-only", action="store_true",
                    help="Only run first (ds,sc) for quick validation.")
    args = ap.parse_args()

    if not BUILD_BIN.exists():
        print(f"ERROR: build binary missing: {BUILD_BIN}")
        sys.exit(1)
    if not SEARCH_BIN.exists():
        print(f"ERROR: search binary missing: {SEARCH_BIN}")
        sys.exit(1)
    if not CONFIGS_CSV.exists():
        print(f"ERROR: {CONFIGS_CSV} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse router_used_configs.csv for Post-filter rows
    rows = []
    with open(CONFIGS_CSV) as f:
        for r in csv.DictReader(f):
            if r["method"] != "Post-filter":
                continue
            if args.datasets and r["dataset"] not in args.datasets:
                continue
            if args.scenarios and r["scenario"] not in args.scenarios:
                continue
            m = re.match(r"M=(\d+)_efc=(\d+)", r["config"])
            if not m:
                print(f"  warn: skip bad config {r['config']}")
                continue
            rows.append({
                "dataset": r["dataset"],
                "scenario": r["scenario"],
                "M": int(m.group(1)),
                "efc": int(m.group(2)),
            })

    if args.first_only and rows:
        rows = rows[:1]

    print(f"PF: {len(rows)} (ds,sc,M,efc) rows from router_used_configs")

    # Unique builds (ds, M, efc)
    build_keys = sorted({(r["dataset"], r["M"], r["efc"]) for r in rows})
    print(f"  -> {len(build_keys)} unique (ds, M, efc) index builds")

    env = make_env()
    failures = []

    # PF binary's internal ef_search sweep — sentinel = largest ef (last written).
    # If sentinel CSV exists, we assume that (ds, sc) search already completed.
    PF_LAST_EF = 2000

    def search_done(ds, sc, M, efc):
        sentinel = out_dir / f"{ds}_{sc}_PF_M{M}_efc{efc}_M={M}_efc={efc}_ef={PF_LAST_EF}.csv"
        return sentinel.exists()

    # Interleaved: per build_key, build → run all matching searches → cleanup
    for bi, (ds, M, efc) in enumerate(build_keys, 1):
        print(f"\n>>> [{bi}/{len(build_keys)}] build_key=({ds}, M={M}, efc={efc})")
        idx_dir = INDEX_ROOT / ds
        idx_path = idx_dir / f"M={M}_efc={efc}.json"
        N = DATASET_N.get(ds)

        matching = [r for r in rows
                    if r["dataset"] == ds and r["M"] == M and r["efc"] == efc]

        # Step 0: skip whole build_key if all scenarios' CSVs already exist
        if not args.dry_run and all(search_done(ds, r["scenario"], M, efc)
                                    for r in matching):
            print(f"  [skip all] all {len(matching)} scenarios already have CSVs")
            continue

        # Step A: build if needed
        if idx_path.exists():
            print(f"  [skip build] (index exists)")
        elif args.skip_build:
            print(f"  [skip build] (--skip-build)")
        else:
            idx_dir.mkdir(parents=True, exist_ok=True)
            base_file = DATA_DIR / ds / f"{ds}_base.fvecs"
            cmd = [str(BUILD_BIN), str(base_file), str(M), str(efc),
                   str(INDEX_ROOT), ds]
            print(f"  [build]")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, env=env, capture_output=True,
                                        timeout=86400)
                if not idx_path.exists():
                    print(f"    FAILED rc={result.returncode} "
                          f"stderr={result.stderr.decode()[:300]}")
                    failures.append(("build", ds, M, efc))
                    continue  # skip searches for this build

        if not args.dry_run and not idx_path.exists():
            # build skipped + missing → skip searches
            for r in matching:
                failures.append(("search-noindex", ds, r["scenario"], M, efc))
            continue

        # Step B: all matching searches (skip ones whose CSV exists)
        all_search_ok = True
        for r in matching:
            sc = r["scenario"]
            if not args.dry_run and search_done(ds, sc, M, efc):
                print(f"  [skip search] {sc} (sentinel CSV exists)")
                continue
            result_dir = FAISS_DIR / "results_postfilter" / ds / sc
            result_dir.mkdir(parents=True, exist_ok=True)
            prefix = out_dir / f"{ds}_{sc}_PF_M{M}_efc{efc}"
            if N is None:
                print(f"  [skip search] unknown N for {ds}")
                failures.append(("search-noN", ds, sc, M, efc))
                all_search_ok = False
                continue
            cmd = [
                str(SEARCH_BIN),
                ds, str(M), str(efc),
                str(INDEX_ROOT), sc, str(result_dir),
                str(DATA_DIR / ds / f"{ds}_base.fvecs"),
                str(DATA_DIR / ds / "label_base.txt"),
                str(DATA_DIR / ds / f"{ds}_query_{sc}.fvecs"),
                str(DATA_DIR / ds / f"{ds}_query_{sc}.txt"),
                str(DATA_DIR / ds / f"{ds}_gt_{sc}.txt"),
                str(K), str(N),
            ]
            run_env = env.copy()
            run_env["PER_QUERY_CSV"] = str(prefix)
            print(f"  [search] {sc} -> {prefix}")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
                continue
            try:
                result = subprocess.run(cmd, env=run_env,
                                        capture_output=False, timeout=36000)
                if result.returncode != 0:
                    print(f"    FAILED rc={result.returncode}")
                    failures.append(("search", ds, sc, M, efc))
                    all_search_ok = False
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT")
                failures.append(("search-timeout", ds, sc, M, efc))
                all_search_ok = False

        # Step C: cleanup index iff all searches OK and not --no-cleanup
        if (not args.dry_run and all_search_ok and not args.no_cleanup
                and idx_path.exists()):
            try:
                idx_path.unlink()
                print(f"  [cleanup] removed {idx_path}")
            except OSError as e:
                print(f"  [cleanup] failed: {e}")

    print(f"\n=== Summary ===")
    print(f"  build_keys: {len(build_keys)}")
    print(f"  rows attempted: {len(rows)}")
    print(f"  failures: {len(failures)}")
    for f in failures:
        print(f"    {f}")


if __name__ == "__main__":
    main()
