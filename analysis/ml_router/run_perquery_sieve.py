#!/usr/bin/env python3
"""Run SIEVE per-query timing on all router-used SIEVE configs.

Reads `analysis/ml_router/router_used_configs.csv`, filters to
method=SIEVE, extracts unique (dataset, scenario, M, b, h) combos, and
invokes SIEVE/run_sieve.py with PER_QUERY_CSV set for each.

Each invocation produces per-query CSVs (one per ef_search value) at
~/benchmarks/per_query_results/{tag}_M{M}_b{b}_h{h}_ef{ef}_{scenario}.csv

After this runs, the per-query data can be aggregated into one master
CSV via merge_perquery.py.
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
CONFIGS_CSV = REPO / "analysis/ml_router/router_used_configs.csv"

# SIEVE scenario remap: router_used_configs uses 'and/or/equal', SIEVE binary
# expects 'original_and / original_or / original_eq'
SC_REMAP = {"and": "original_and", "or": "original_or", "equal": "original_eq"}


def parse_sieve_config(cfg: str) -> dict:
    """Parse a SIEVE config string from router_used_configs.csv.

    Example: 'M=32_ef_construction=40_index_budget=2.0_hist_pct=0.25_ef_search=20'
    """
    parts = cfg.split("_")
    out = {}
    i = 0
    while i < len(parts):
        # handle multi-token keys like "ef_construction", "index_budget", etc.
        # any token ending in '=value' is a key=value pair
        if "=" in parts[i]:
            key, val = parts[i].split("=", 1)
            out[key] = val
            i += 1
        else:
            # token is part of a compound key, append to previous key
            if not out:
                i += 1
                continue
            last_key = list(out.keys())[-1]
            # multi-token: e.g. 'ef' + 'construction=40' -> key='ef_construction'
            if i + 1 < len(parts) and "=" in parts[i + 1]:
                new_key = parts[i] + "_" + parts[i + 1].split("=", 1)[0]
                new_val = parts[i + 1].split("=", 1)[1]
                # remove last_key entry if it was incomplete
                # actually just merge: last_key becomes last_key + "_" + parts[i]
                # this is getting complex; simpler: re-parse whole string with regex
                pass
            i += 1
    return out


def parse_sieve_config_regex(cfg: str) -> dict:
    """Parse SIEVE config string of format:
      M=32_ef_construction=40_index_budget=2.0_hist_pct=0.25_ef_search=20

    `\b` doesn't work here because `_` is a word char in Python regex.
    Use explicit `(?:^|_)` / `(?=_|$)` anchors instead.
    """
    out = {}
    keys = ["M", "ef_construction", "index_budget", "hist_pct", "ef_search"]
    for key in keys:
        pat = re.compile(rf"(?:^|_){re.escape(key)}=([\d.]+)(?=_|$)")
        m = pat.search(cfg)
        if m:
            out[key] = m.group(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=str(Path.home() / "benchmarks" / "per_query_results"),
                    help="Where to write per-query CSVs.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Limit to these V2 datasets (default: all in CSV).")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Limit to these scenarios (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print invocations without running.")
    args = ap.parse_args()

    if not CONFIGS_CSV.exists():
        print(f"ERROR: {CONFIGS_CSV} not found")
        sys.exit(1)

    # Read configs, filter to SIEVE
    sieve_configs = []
    with open(CONFIGS_CSV) as f:
        for row in csv.DictReader(f):
            if row["method"] != "SIEVE":
                continue
            if args.datasets and row["dataset"] not in args.datasets:
                continue
            if args.scenarios and row["scenario"] not in args.scenarios:
                continue
            parsed = parse_sieve_config_regex(row["config"])
            sieve_configs.append({
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                **parsed,
            })

    print(f"Loaded {len(sieve_configs)} SIEVE rows from router_used_configs.csv")

    # Group by (dataset, scenario, M, ef_construction, index_budget, hist_pct)
    # since one run_sieve.py invocation covers ALL ef_search values
    invocation_keys = set()
    for c in sieve_configs:
        key = (c["dataset"], c["scenario"], c["M"], c["ef_construction"],
               c["index_budget"], c["hist_pct"])
        invocation_keys.add(key)

    invocation_keys = sorted(invocation_keys)
    print(f"=> {len(invocation_keys)} unique (ds, sc, M, efc, b, h) invocations\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for i, (ds, sc, M, efc, b, h) in enumerate(invocation_keys, 1):
        sieve_sc = SC_REMAP.get(sc, sc)
        prefix = out_dir / f"{ds}_{sc}_SIEVE_M{M}_efc{efc}_b{b}_h{h}"
        cmd = [
            "python3", "SIEVE/run_sieve.py",
            "--dataset", ds,
            "--scenario", sieve_sc,
            "--M", str(M),
            "--ef_construction", str(efc),
            "--index_budget", str(b),
            "--hist_pct", str(h),
        ]
        print(f"[{i}/{len(invocation_keys)}] {ds}/{sc} M={M} efc={efc} b={b} h={h}")
        print(f"  PER_QUERY_CSV={prefix}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        env["PER_QUERY_CSV"] = str(prefix)
        try:
            result = subprocess.run(cmd, cwd=str(REPO), env=env,
                                    timeout=3600, capture_output=False)
            if result.returncode != 0:
                print(f"  FAILED (returncode {result.returncode})")
                failures.append((ds, sc, M, efc, b, h, "nonzero exit"))
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 1h")
            failures.append((ds, sc, M, efc, b, h, "timeout"))
        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append((ds, sc, M, efc, b, h, str(e)))

    print(f"\n=== Summary ===")
    print(f"  Total invocations: {len(invocation_keys)}")
    print(f"  Failed: {len(failures)}")
    for f in failures:
        print(f"    {f}")


if __name__ == "__main__":
    main()
