#!/usr/bin/env python3
"""
ACORN per-query recall extractor.

ACORN 跑出 `_idx_uint32.bin` × N configs (M × M_beta × gamma × ef = ~990 files / dataset),
但 .gitignore 排除 → ML routing 需要 per-query recall 但 .bin 不进 git.

解法 (同 DiskANN/SIEVE): extract 到单 CSV 入 git, .bin 留本地.

输入 (per-config):
  ACORN/data/param_search_{ds}/results/{ds}/{scenario}/M={M}_M_beta={Mb}_gamma={g}_ef={ef}_idx_uint32.bin

  二进制格式 (跟 PF 同): nq uint32 + k uint32 + nq*k uint32 IDs.

输出: analysis/ml_router/perquery_recall/perquery_recall_v2_acorn.csv
字段: query_id, dataset, scenario, method=ACORN, recall_at_10, params (M/Mb/gamma/ef)

每个 (ds, sc) 选 max-recall config 的 1000 query recall.

用法:
  cd ~/benchmarks/discrete && python analysis/extract_v2_perquery_acorn.py
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS_STD = ["and", "or", "equal"]

K = 10


def compute_perquery_recall(gt_list, res_list, k=10):
    recalls = []
    for gt_ids, res_ids in zip(gt_list, res_list):
        gt_set = set(gt_ids[:k]) - {-1, 4294967295}
        res_set = set(res_ids[:k]) - {-1, 4294967295}
        if len(gt_set) > 0:
            recalls.append(len(gt_set & res_set) / min(k, len(gt_set)))
        else:
            recalls.append(0.0)
    return recalls


def load_gt_txt(path, k=10):
    gt = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_acorn_bin(path):
    """Read ACORN result binary: nq uint32, k uint32, then nq*k uint32 (same as PF)."""
    with open(path, "rb") as f:
        nq = struct.unpack("I", f.read(4))[0]
        k = struct.unpack("I", f.read(4))[0]
        ids = struct.unpack(f"{nq * k}I", f.read(nq * k * 4))
    res = [list(ids[i * k:(i + 1) * k]) for i in range(nq)]
    return res, nq, k


def parse_config(stem: str) -> str:
    """Parse 'M=48_M_beta=96_gamma=12_ef=100' from filename stem."""
    parts = []
    for token in ("M=", "M_beta=", "gamma=", "ef="):
        if token in stem:
            v = stem.split(token, 1)[1].split("_")[0]
            parts.append(f"{token.rstrip('=')}={v}")
    return "_".join(parts)


def process_acorn(dataset, data_dir, all_rows):
    # ACORN has both `param_search_{ds}` and `param_search_{ds}_gamma1`. 主体在前者.
    candidates = [
        BASE_DIR / "ACORN" / "data" / f"param_search_{dataset}" / "results" / dataset,
        BASE_DIR / "ACORN" / "data" / f"param_search_{dataset}_gamma1" / "results" / dataset,
    ]
    base_dir = next((c for c in candidates if c.exists()), None)
    if not base_dir:
        print(f"  [ACORN] {dataset}: no results dir")
        return

    for scenario_std in SCENARIOS_STD:
        sc_dir = base_dir / scenario_std
        if not sc_dir.exists():
            print(f"  [ACORN] {dataset}/{scenario_std}: dir missing")
            continue

        bin_files = sorted(sc_dir.glob("*_idx_uint32.bin"))
        if not bin_files:
            print(f"  [ACORN] {dataset}/{scenario_std}: no bin files")
            continue

        gt_path = data_dir / dataset / f"{dataset}_gt_{scenario_std}.txt"
        if not gt_path.exists():
            print(f"  [ACORN] GT not found: {gt_path}")
            continue
        gt = load_gt_txt(gt_path, K)

        best_recalls = None
        best_params = None
        best_avg = -1.0

        for bf in bin_files:
            try:
                results, nq, k = load_acorn_bin(bf)
            except Exception as e:
                continue
            n = min(len(gt), len(results))
            if n == 0:
                continue
            recalls = compute_perquery_recall(gt[:n], results[:n], K)
            avg_r = sum(recalls) / len(recalls)
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                stem = bf.stem.replace("_idx_uint32", "")
                best_params = parse_config(stem)

        if best_recalls is None:
            print(f"  [ACORN] {dataset}/{scenario_std}: no valid per-query data")
            continue

        for qid, r in enumerate(best_recalls):
            all_rows.append({
                "query_id": qid,
                "dataset": dataset,
                "scenario": scenario_std,
                "method": "ACORN",
                "recall_at_10": r,
                "params": best_params,
            })
        print(f"  [ACORN] {dataset}/{scenario_std}: avg recall={best_avg:.4f}, n={len(best_recalls)}, params={best_params}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    datasets = args.dataset or V2_DATASETS
    data_dir = Path(args.data_dir) if args.data_dir else Path(os.path.expanduser("~/benchmarks/datasets/discrete"))

    output_dir = BASE_DIR / "analysis" / "perquery_recall"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(output_dir / "perquery_recall_v2_acorn.csv")

    all_rows = []
    for ds in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")
        process_acorn(ds, data_dir, all_rows)

    print(f"\nWriting {len(all_rows)} rows to {output_file}")
    with open(output_file, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print("Done.")


if __name__ == "__main__":
    main()
