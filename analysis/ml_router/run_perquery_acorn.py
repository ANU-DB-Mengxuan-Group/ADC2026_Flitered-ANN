#!/usr/bin/env python3
"""Run ACORN per-query timing on all router-used configs.

Reads `analysis/ml_router/router_used_configs.csv` method=ACORN.
For each unique (dataset, M, M_beta, gamma) builds the index if missing,
then runs search_acorn_index with PER_QUERY_CSV env var for each
(dataset, scenario, M, M_beta, gamma).

Binary internally sweeps ef_search and writes per-query CSV per ef:
  {PER_QUERY_CSV}_M=X_M_beta=Y_gamma=Z_ef=E.csv

Scenarios use raw and/or/equal (no remap).
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
ACORN_DIR = REPO / "ACORN"
BUILD_BIN = ACORN_DIR / "build/demos/build_acorn_index"
SEARCH_BIN = ACORN_DIR / "build/demos/search_acorn_index"
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
    """Env for ACORN binary. Must prepend ACORN's own libfaiss.so path so the
    binary picks up the ACORN-patched faiss (has IndexACORN class), not the
    vanilla faiss we rebuilt for Post-filter.

    Symptom if wrong: `symbol lookup error: undefined symbol:
    _ZTIN5faiss10IndexACORNE` (typeinfo for faiss::IndexACORN).
    """
    env = os.environ.copy()
    acorn_faiss = str(ACORN_DIR / "build" / "faiss")
    cp = os.environ.get("CONDA_PREFIX")
    ld = env.get("LD_LIBRARY_PATH", "")
    parts = []
    parts.append(acorn_faiss)  # ACORN's faiss MUST be first
    if cp:
        parts.append(os.path.join(cp, "lib"))
    if ld:
        parts.append(ld)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    print(f"[MKL] Set LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    if not cp:
        print("[MKL] WARN: CONDA_PREFIX not set — may also be missing MKL libs.")
    return env


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
    ap.add_argument("--builds", default=None,
                    help="Comma-separated build_keys to limit to. "
                         "Format: 'ds:M:Mb:gamma,ds:M:Mb:gamma'. "
                         "Use to split across parallel jobs.")
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
            if r["method"] != "ACORN":
                continue
            if args.datasets and r["dataset"] not in args.datasets:
                continue
            if args.scenarios and r["scenario"] not in args.scenarios:
                continue
            m = re.match(r"M=(\d+)_M_beta=(\d+)_gamma=(\d+)", r["config"])
            if not m:
                print(f"  warn: bad config {r['config']}")
                continue
            rows.append({
                "dataset": r["dataset"],
                "scenario": r["scenario"],
                "M": int(m.group(1)),
                "M_beta": int(m.group(2)),
                "gamma": int(m.group(3)),
            })

    if args.builds:
        wanted = set()
        for spec in args.builds.split(","):
            parts = spec.strip().split(":")
            if len(parts) != 4:
                print(f"  warn: bad --builds spec {spec!r} (need ds:M:Mb:gamma)")
                continue
            wanted.add((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
        rows = [r for r in rows if (r["dataset"], r["M"], r["M_beta"], r["gamma"]) in wanted]
        print(f"  --builds filter: kept {len(rows)} rows matching {len(wanted)} build_keys")

    if args.first_only and rows:
        rows = rows[:1]

    print(f"ACORN: {len(rows)} (ds,sc,M,M_beta,gamma) rows")

    # Each row uses output_base per-dataset.
    # Unique builds (ds, M, M_beta, gamma)
    build_keys = sorted({
        (r["dataset"], r["M"], r["M_beta"], r["gamma"]) for r in rows
    })
    print(f"  -> {len(build_keys)} unique (ds, M, M_beta, gamma) builds")

    env = make_env()
    failures = []

    # ACORN binary's internal ef_search sweep — last value is 1200, used as sentinel
    ACORN_LAST_EF = 1200

    def search_done(ds, sc, M, M_beta, gamma):
        sentinel = out_dir / (f"{ds}_{sc}_ACORN_M{M}_Mb{M_beta}_g{gamma}_"
                              f"M={M}_M_beta={M_beta}_gamma={gamma}_ef={ACORN_LAST_EF}.csv")
        return sentinel.exists()

    # Group rows by build_key (ds, M, M_beta, gamma) → list of rows
    build_to_searches = {}
    for r in rows:
        key = (r["dataset"], r["M"], r["M_beta"], r["gamma"])
        build_to_searches.setdefault(key, []).append(r)

    # Interleaved: per build_key, build → searches → cleanup
    for bi, (ds, M, M_beta, gamma) in enumerate(sorted(build_to_searches), 1):
        print(f"\n>>> [{bi}/{len(build_to_searches)}] build_key=({ds}, M={M}, Mb={M_beta}, g={gamma})")
        matching = build_to_searches[(ds, M, M_beta, gamma)]

        # Step 0: skip whole build_key if all scenarios' CSVs already exist
        if not args.dry_run and all(search_done(ds, r["scenario"], M, M_beta, gamma)
                                    for r in matching):
            print(f"  [skip all] all {len(matching)} scenarios already have CSVs")
            continue
        output_base = ACORN_DIR / "data" / f"param_search_{ds}"
        indices_base = output_base / "indices"
        ds_idx_dir = indices_base / ds
        idx_path = ds_idx_dir / f"hybrid_M={M}_Mb={M_beta}_gamma={gamma}.json"
        # ACORN also produces gamma=1 byproduct (per auto_param_search_v2.py)
        idx_path_gamma1 = ds_idx_dir / f"hybrid_M={M}_Mb={M_beta}_gamma=1.json"
        N = DATASET_N[ds]

        # Step A: build if needed
        if idx_path.exists():
            print(f"  [skip build] (index exists)")
        elif args.skip_build:
            print(f"  [skip build] (--skip-build)")
        else:
            ds_idx_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(BUILD_BIN),
                str(N), str(gamma),
                str(DATA_DIR / ds / f"{ds}_base.fvecs"),
                str(M), str(M_beta),
                str(indices_base), ds,
            ]
            print(f"  [build]")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
            else:
                rb = subprocess.run(cmd, env=env, capture_output=True, timeout=18000)
                if not idx_path.exists():
                    print(f"    FAILED rc={rb.returncode} "
                          f"stderr={rb.stderr.decode()[:300]}")
                    failures.append(("build", ds, M, M_beta, gamma))
                    continue

        if not args.dry_run and not idx_path.exists():
            for r in build_to_searches[(ds, M, M_beta, gamma)]:
                failures.append(("search-noindex", ds, r["scenario"], M, M_beta, gamma))
            continue

        # Step B: all matching searches (skip ones whose CSV exists)
        all_search_ok = True
        for r in matching:
            sc = r["scenario"]
            if not args.dry_run and search_done(ds, sc, M, M_beta, gamma):
                print(f"  [skip search] {sc} (sentinel CSV exists)")
                continue
            result_dir = output_base / "results" / ds / sc
            result_dir.mkdir(parents=True, exist_ok=True)
            prefix = out_dir / f"{ds}_{sc}_ACORN_M{M}_Mb{M_beta}_g{gamma}"
            run_env = env.copy()
            run_env["PER_QUERY_CSV"] = str(prefix)
            run_env["debugSearchFlag"] = "0"
            cmd = [
                str(SEARCH_BIN),
                str(N), str(gamma), ds,
                str(M), str(M_beta),
                str(indices_base), sc, str(result_dir),
                str(DATA_DIR / ds / f"{ds}_base.fvecs"),
                str(DATA_DIR / ds / "label_base.txt"),
                str(DATA_DIR / ds / f"{ds}_query_{sc}.fvecs"),
                str(DATA_DIR / ds / f"{ds}_query_{sc}.txt"),
                str(DATA_DIR / ds / f"{ds}_gt_{sc}.txt"),
                str(K),
            ]
            print(f"  [search] {sc} -> {prefix}")
            if args.dry_run:
                print(f"    {' '.join(cmd)}")
                continue
            try:
                res = subprocess.run(cmd, env=run_env,
                                     capture_output=False, timeout=7200)
                if res.returncode != 0:
                    print(f"    FAILED rc={res.returncode}")
                    failures.append(("search", ds, sc, M, M_beta, gamma))
                    all_search_ok = False
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT")
                failures.append(("search-timeout", ds, sc, M, M_beta, gamma))
                all_search_ok = False

        # Step C: cleanup index + gamma=1 byproduct iff all searches OK
        if not args.dry_run and all_search_ok and not args.no_cleanup:
            removed = 0
            for p in [idx_path, idx_path_gamma1]:
                if p.exists():
                    try:
                        p.unlink()
                        removed += 1
                    except OSError as e:
                        print(f"  [cleanup] failed {p}: {e}")
            if removed:
                print(f"  [cleanup] removed {removed} ACORN index file(s)")

    print(f"\n=== Summary ===")
    print(f"  build_keys: {len(build_to_searches)}")
    print(f"  failures: {len(failures)}")
    for f in failures:
        print(f"    {f}")


if __name__ == "__main__":
    main()
