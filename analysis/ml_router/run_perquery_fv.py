#!/usr/bin/env python3
"""Run FilteredVamana (DiskANN) per-query timing on all router-used configs.

Reads `analysis/ml_router/router_used_configs.csv` method=FilteredVamana.
For each unique (dataset, R) builds index if missing, then runs
search_memory_index with all router-used Lsearch values per
(ds, search_scenario, R) and PER_QUERY_CSV env var.

Scenario mapping (router row → DiskANN):
  and   → containment (file suffix: and)
  or    → overlap     (file suffix: or)
  equal → equality    (file suffix: equal)

Requires converted labels + GT bin at
  {output_base}/converted_labels/{ds}/{search_scenario}/query_labels.txt
  {output_base}/converted_labels/{ds}/{search_scenario}/gt_diskann.bin
These are produced by `auto_diskann_original.py` (label conversion via
tmp_index from `build_memory_index`). If missing for any (ds, sc),
orchestrator skips that row and reports it.

Binary writes per-query CSV per Lsearch value:
  {PER_QUERY_CSV}_L{Lsearch}.csv
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
DISKANN_DIR = REPO / "DiskANN"
BUILD_BIN = DISKANN_DIR / "build/apps/build_memory_index"
SEARCH_BIN = DISKANN_DIR / "build/apps/search_memory_index"
OUTPUT_BASE = DISKANN_DIR / "data_original"
CONFIGS_CSV = REPO / "analysis/ml_router/router_used_configs.csv"
DEFAULT_OUT_DIR = Path.home() / "benchmarks" / "per_query_results"
K = 10
L_BUILD = 100
ALPHA = 1.2
NUM_THREADS = 16

# Router scenario → (search_scenario, file_suffix)
SC_MAP = {
    "and":   ("containment", "and"),
    "or":    ("overlap",     "or"),
    "equal": ("equality",    "equal"),
}


def make_env() -> dict:
    env = os.environ.copy()
    cp = os.environ.get("CONDA_PREFIX")
    if not cp:
        print("[MKL] WARN: CONDA_PREFIX not set — run `conda activate benchmark` first.")
        return env
    mkl = os.path.join(cp, "lib")
    ld = env.get("LD_LIBRARY_PATH", "")
    if mkl not in ld:
        env["LD_LIBRARY_PATH"] = f"{mkl}:{ld}" if ld else mkl
        print(f"[MKL] Set LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    return env


def get_label_file(path: Path) -> Path:
    onebased = path.parent / f"{path.stem}_1based{path.suffix}"
    return onebased if onebased.exists() else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep index files after successful searches.")
    ap.add_argument("--first-only", action="store_true")
    args = ap.parse_args()

    for b in [BUILD_BIN, SEARCH_BIN, CONFIGS_CSV]:
        if not b.exists():
            print(f"ERROR: missing {b}")
            sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(CONFIGS_CSV) as f:
        for r in csv.DictReader(f):
            if r["method"] != "FilteredVamana":
                continue
            if args.datasets and r["dataset"] not in args.datasets:
                continue
            if args.scenarios and r["scenario"] not in args.scenarios:
                continue
            # Config format: R=128_Lsearch=100_latency_us=27923.85
            m = re.match(r"R=(\d+)_Lsearch=(\d+)", r["config"])
            if not m:
                print(f"  warn: bad config {r['config']}")
                continue
            rows.append({
                "dataset": r["dataset"],
                "scenario": r["scenario"],
                "R": int(m.group(1)),
                "Ls": int(m.group(2)),
            })

    if args.first_only and rows:
        rows = rows[:1]

    print(f"FV: {len(rows)} (ds,sc,R,Ls) rows from router_used_configs")

    # Unique builds (ds, R)
    build_keys = sorted({(r["dataset"], r["R"]) for r in rows})
    print(f"  -> {len(build_keys)} unique (ds, R) builds")

    env = make_env()
    failures = []

    # Group rows by build_key (ds, R) → list of rows
    build_to_searches = {}
    for r in rows:
        key = (r["dataset"], r["R"])
        build_to_searches.setdefault(key, []).append(r)

    def search_done(ds, sc, R, Ls_list):
        Ls_max = max(Ls_list)
        sentinel = out_dir / f"{ds}_{sc}_FV_R{R}_L{Ls_max}.csv"
        return sentinel.exists()

    # Interleaved: per build_key, build → searches → cleanup
    for bi, (ds, R) in enumerate(sorted(build_to_searches), 1):
        print(f"\n>>> [{bi}/{len(build_to_searches)}] build_key=({ds}, R={R})")

        # Step 0: precompute per-scenario Ls lists for skip check
        sc_to_Ls_preview = {}
        for r in build_to_searches[(ds, R)]:
            sc_to_Ls_preview.setdefault(r["scenario"], []).append(r["Ls"])
        if not args.dry_run and all(
                search_done(ds, sc, R, ls_list)
                for sc, ls_list in sc_to_Ls_preview.items()):
            print(f"  [skip all] all {len(sc_to_Ls_preview)} scenarios already have CSVs")
            continue

        idx_dir = OUTPUT_BASE / "indices" / ds
        idx_prefix = idx_dir / f"index_R={R}"
        idx_data = Path(f"{idx_prefix}.data")
        idx_files_glob = lambda: list(idx_dir.glob(f"index_R={R}*"))

        # Step A: build if needed
        if idx_data.exists():
            print(f"  [skip build] (index exists)")
        elif args.skip_build:
            print(f"  [skip build] (--skip-build)")
        else:
            idx_dir.mkdir(parents=True, exist_ok=True)
            base_bin = DATA_DIR / ds / f"{ds}_base.bin"
            label_base = get_label_file(DATA_DIR / ds / "label_base.txt")
            cmd = [
                str(BUILD_BIN),
                "--data_type", "float",
                "--dist_fn", "l2",
                "--data_path", str(base_bin),
                "--index_path_prefix", str(idx_prefix),
                "--label_file", str(label_base),
                "-R", str(R),
                "-L", str(L_BUILD),
                "--alpha", str(ALPHA),
                "-T", str(NUM_THREADS),
            ]
            print(f"  [build]")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
            else:
                rb = subprocess.run(cmd, env=env, capture_output=True, timeout=7200)
                if not idx_data.exists():
                    print(f"    FAILED rc={rb.returncode} "
                          f"stderr={rb.stderr.decode()[:300]}")
                    failures.append(("build", ds, R))
                    continue

        if not args.dry_run and not idx_data.exists():
            for r in build_to_searches[(ds, R)]:
                failures.append(("search-noindex", ds, r["scenario"], R))
            continue

        # Step B: group searches by scenario; one invocation per scenario
        sc_to_Ls = {}
        for r in build_to_searches[(ds, R)]:
            sc_to_Ls.setdefault(r["scenario"], []).append(r["Ls"])

        all_search_ok = True
        for sc, Ls_list in sorted(sc_to_Ls.items()):
            Ls_unique_check = sorted(set(Ls_list))
            if not args.dry_run and search_done(ds, sc, R, Ls_unique_check):
                print(f"  [skip search] {sc} (sentinel CSV exists)")
                continue
            search_sc, suffix = SC_MAP[sc]

            converted_labels = OUTPUT_BASE / "converted_labels" / ds / search_sc / "query_labels.txt"
            gt_file = OUTPUT_BASE / "converted_labels" / ds / search_sc / "gt_diskann.bin"
            if not converted_labels.exists() or not gt_file.exists():
                print(f"  [skip search] missing converted labels/gt for {ds}/{search_sc}")
                print(f"    fix: run `python DiskANN/scripts/auto_diskann_original.py {ds}` once")
                failures.append(("search-noprep", ds, sc, R))
                all_search_ok = False
                continue

            result_dir = OUTPUT_BASE / "results" / ds
            result_dir.mkdir(parents=True, exist_ok=True)
            result_prefix = result_dir / f"index_R={R}_{search_sc}_perquery"
            Ls_unique = sorted(set(Ls_list))
            prefix = out_dir / f"{ds}_{sc}_FV_R{R}"
            run_env = env.copy()
            run_env["PER_QUERY_CSV"] = str(prefix)
            query_file = DATA_DIR / ds / f"{ds}_query_{suffix}.bin"
            cmd = [
                str(SEARCH_BIN),
                "--data_type", "float",
                "--dist_fn", "l2",
                "--index_path_prefix", str(idx_prefix),
                "--query_file", str(query_file),
                "--query_filters_file", str(converted_labels),
                "--gt_file", str(gt_file),
                "--label_type", "uint",
                "--recall_at", str(K),
                "--result_path", str(result_prefix),
                "--num_threads", str(NUM_THREADS),
                "--filter_scenario", search_sc,
                "-L", *[str(L) for L in Ls_unique],
            ]
            print(f"  [search] {sc} Lsearch={Ls_unique} -> {prefix}")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
                continue
            try:
                res = subprocess.run(cmd, env=run_env,
                                     capture_output=False, timeout=7200)
                if res.returncode != 0:
                    print(f"    FAILED rc={res.returncode}")
                    failures.append(("search", ds, sc, R))
                    all_search_ok = False
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT")
                failures.append(("search-timeout", ds, sc, R))
                all_search_ok = False

        # Step C: cleanup all FV index files iff all searches OK
        if not args.dry_run and all_search_ok and not args.no_cleanup:
            removed = 0
            for f in idx_files_glob():
                try:
                    f.unlink()
                    removed += 1
                except OSError as e:
                    print(f"  [cleanup] failed {f}: {e}")
            if removed:
                print(f"  [cleanup] removed {removed} FV index files")

    print(f"\n=== Summary ===")
    print(f"  build_keys: {len(build_to_searches)}")
    print(f"  failures: {len(failures)}")
    for f in failures:
        print(f"    {f}")


if __name__ == "__main__":
    main()
