#!/usr/bin/env python3
"""Run UNG per-query timing on all router-used configs.

Reads `analysis/ml_router/router_used_configs.csv` method=UNG.
For each unique (dataset, build_scenario, max_degree, Lbuild) builds
the index if missing, then runs search_UNG_index with all router-used
Lsearch values per (ds, search_scenario, M, Lb).

Scenario mapping (router row → UNG):
  and   → build=general,   search=containment, file_suffix=and
  or    → build=general,   search=overlap,     file_suffix=or
  equal → build=equality,  search=equality,    file_suffix=equal

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
UNG_DIR = REPO / "UNG-dev"
BUILD_BIN = UNG_DIR / "build/apps/build_UNG_index"
SEARCH_BIN = UNG_DIR / "build/apps/search_UNG_index"
INDEX_ROOT = UNG_DIR / "indices_original"
CONFIGS_CSV = REPO / "analysis/ml_router/router_used_configs.csv"
DEFAULT_OUT_DIR = Path.home() / "benchmarks" / "per_query_results"
K = 10
NUM_CROSS_EDGES = 6
NUM_THREADS = 16

# Router scenario name → (build_scenario, search_scenario, file_suffix)
SC_MAP = {
    "and":   ("general",  "containment", "and"),
    "or":    ("general",  "overlap",     "or"),
    "equal": ("equality", "equality",    "equal"),
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
    """V2 datasets use *_1based.txt to avoid 0-as-sentinel segfault."""
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

    # Parse UNG rows
    rows = []
    with open(CONFIGS_CSV) as f:
        for r in csv.DictReader(f):
            if r["method"] != "UNG":
                continue
            if args.datasets and r["dataset"] not in args.datasets:
                continue
            if args.scenarios and r["scenario"] not in args.scenarios:
                continue
            m = re.match(r"max_degree=(\d+)_Lbuild=(\d+)_Lsearch=(\d+)", r["config"])
            if not m:
                print(f"  warn: bad config {r['config']}")
                continue
            rows.append({
                "dataset": r["dataset"],
                "scenario": r["scenario"],
                "M": int(m.group(1)),
                "Lb": int(m.group(2)),
                "Ls": int(m.group(3)),
            })

    if args.first_only and rows:
        rows = rows[:1]

    print(f"UNG: {len(rows)} (ds,sc,M,Lb,Ls) rows from router_used_configs")

    # Unique builds (ds, build_scenario, M, Lb)
    build_keys = sorted({
        (r["dataset"], SC_MAP[r["scenario"]][0], r["M"], r["Lb"])
        for r in rows
    })
    print(f"  -> {len(build_keys)} unique (ds, build_scenario, M, Lb) builds")

    env = make_env()
    failures = []

    # Group rows by build_key → list of (scenario, Lsearch)
    # build_key = (ds, build_sc, M, Lb) — same build_sc handles multiple scenarios
    build_to_searches = {}
    for r in rows:
        bsc = SC_MAP[r["scenario"]][0]
        key = (r["dataset"], bsc, r["M"], r["Lb"])
        build_to_searches.setdefault(key, []).append(r)

    def search_done(ds, sc, M, Lb, Ls_list):
        # Sentinel = largest Lsearch in this build_key's set
        Ls_max = max(Ls_list)
        sentinel = out_dir / f"{ds}_{sc}_UNG_M{M}_Lb{Lb}_L{Ls_max}.csv"
        return sentinel.exists()

    # Interleaved: per build_key, build → searches → cleanup
    for bi, (ds, build_sc, M, Lb) in enumerate(sorted(build_to_searches), 1):
        print(f"\n>>> [{bi}/{len(build_to_searches)}] build_key=({ds}, {build_sc}, M={M}, Lb={Lb})")

        # Step 0: precompute per-scenario Ls lists for skip check
        search_rows_preview = build_to_searches[(ds, build_sc, M, Lb)]
        sc_to_Ls_preview = {}
        for r in search_rows_preview:
            sc_to_Ls_preview.setdefault(r["scenario"], []).append(r["Ls"])
        # Skip whole build_key if all scenarios already have their sentinel CSV
        if not args.dry_run and all(
                search_done(ds, sc, M, Lb, ls_list)
                for sc, ls_list in sc_to_Ls_preview.items()):
            print(f"  [skip all] all {len(sc_to_Ls_preview)} scenarios already have CSVs")
            continue

        idx_dir = INDEX_ROOT / ds / build_sc
        idx_prefix = idx_dir / f"index_M={M}_L={Lb}"
        # UNG's save() does `fs::create_directories(index_path_prefix)` first then
        # writes sibling files (prefix+"meta", prefix+"vecs.bin", ...). A crashed
        # build leaves the empty dir but no sibling files. Count ONLY files so we
        # don't mistake the empty-dir artifact for a finished build.
        idx_files_glob = lambda: [
            f for f in idx_dir.glob(f"index_M={M}_L={Lb}*") if f.is_file()
        ]

        # Step A: build if needed
        existing = idx_files_glob()
        if existing:
            print(f"  [skip build] ({len(existing)} files exist)")
        elif args.skip_build:
            print(f"  [skip build] (--skip-build)")
        else:
            idx_dir.mkdir(parents=True, exist_ok=True)
            base_bin = DATA_DIR / ds / f"{ds}_base.bin"
            label_base = get_label_file(DATA_DIR / ds / "label_base.txt")
            cmd = [
                str(BUILD_BIN),
                "--data_type", "float",
                "--dist_fn", "L2",
                "--base_bin_file", str(base_bin),
                "--base_label_file", str(label_base),
                "--index_path_prefix", str(idx_prefix),
                "--scenario", build_sc,
                "--max_degree", str(M),
                "--Lbuild", str(Lb),
                "--num_cross_edges", str(NUM_CROSS_EDGES),
                "--num_threads", str(NUM_THREADS),
            ]
            print(f"  [build]")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
            else:
                rb = subprocess.run(cmd, env=env, capture_output=True, timeout=7200)
                if not idx_files_glob():
                    print(f"    FAILED rc={rb.returncode} "
                          f"stderr={rb.stderr.decode()[:300]}")
                    failures.append(("build", ds, build_sc, M, Lb))
                    continue

        if not args.dry_run and not idx_files_glob():
            for r in build_to_searches[(ds, build_sc, M, Lb)]:
                failures.append(("search-noindex", ds, r["scenario"], M, Lb))
            continue

        # Step B: group searches by scenario (each scenario gets all its Lsearch)
        search_rows = build_to_searches[(ds, build_sc, M, Lb)]
        sc_to_Ls = {}
        for r in search_rows:
            sc_to_Ls.setdefault(r["scenario"], []).append(r["Ls"])

        all_search_ok = True
        for sc, Ls_list in sorted(sc_to_Ls.items()):
            _, search_sc, suffix = SC_MAP[sc]
            Ls_unique = sorted(set(Ls_list))
            if not args.dry_run and search_done(ds, sc, M, Lb, Ls_unique):
                print(f"  [skip search] {sc} (sentinel CSV exists)")
                continue
            result_dir = UNG_DIR / "results_original" / ds / sc
            result_dir.mkdir(parents=True, exist_ok=True)
            result_prefix = result_dir / f"index_M={M}_L={Lb}_Ls=multi_"
            prefix = out_dir / f"{ds}_{sc}_UNG_M{M}_Lb{Lb}"
            run_env = env.copy()
            run_env["PER_QUERY_CSV"] = str(prefix)

            cmd = [
                str(SEARCH_BIN),
                "--data_type", "float",
                "--dist_fn", "L2",
                "--base_bin_file", str(DATA_DIR / ds / f"{ds}_base.bin"),
                "--query_bin_file", str(DATA_DIR / ds / f"{ds}_query_{suffix}.bin"),
                "--base_label_file", str(get_label_file(DATA_DIR / ds / "label_base.txt")),
                "--query_label_file", str(get_label_file(DATA_DIR / ds / f"{ds}_query_{suffix}.txt")),
                "--gt_file", str(DATA_DIR / ds / f"{ds}_gt_{suffix}.bin"),
                "--K", str(K),
                "--index_path_prefix", str(idx_prefix),
                "--scenario", search_sc,
                "--Lsearch", *[str(L) for L in Ls_unique],
                "--num_threads", str(NUM_THREADS),
                "--result_path_prefix", str(result_prefix),
            ]
            print(f"  [search] {sc} Lsearch={Ls_unique} -> {prefix}")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
                continue
            try:
                rs = subprocess.run(cmd, env=run_env, capture_output=False, timeout=7200)
                if rs.returncode != 0:
                    print(f"    FAILED rc={rs.returncode}")
                    failures.append(("search", ds, sc, M, Lb))
                    all_search_ok = False
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT")
                failures.append(("search-timeout", ds, sc, M, Lb))
                all_search_ok = False

        # Step C: cleanup index iff all searches OK
        # Remove both sibling files AND the empty dir artifact left by save().
        if not args.dry_run and all_search_ok and not args.no_cleanup:
            removed = 0
            import shutil
            for p in idx_dir.glob(f"index_M={M}_L={Lb}*"):
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    removed += 1
                except OSError as e:
                    print(f"  [cleanup] failed {p}: {e}")
            if removed:
                print(f"  [cleanup] removed {removed} UNG index files/dirs")

    print(f"\n=== Summary ===")
    print(f"  build_keys: {len(build_to_searches)}")
    print(f"  failures: {len(failures)}")
    for f in failures:
        print(f"    {f}")


if __name__ == "__main__":
    main()
