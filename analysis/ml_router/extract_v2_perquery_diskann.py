#!/usr/bin/env python3
"""
DiskANN FilteredVamana per-query recall extractor.

DiskANN 跑完后产出:
  DiskANN/data_original/results/{ds}/index_R={R}_{scenario}_Ls={Ls}_{Ls}_idx_uint32.bin

每个 .bin 是 DiskANN 标准 CSR 格式: nq (uint32) + k (uint32) + nq*k * uint32 IDs.
跟 PF 的 idx_uint32.bin 同格式.

为啥需要这个 extractor:
  .gitignore 排除了 DiskANN/data_*/results/**/*.bin (90+ 文件 / dataset, 总 GB).
  但 ML routing 需要 per-query recall. 解法: extract 到单文件 CSV 入 git, .bin 留本地.

输出: analysis/ml_router/perquery_recall/perquery_recall_v2_diskann.csv
字段: query_id, dataset, scenario, method, recall_at_10, params

用法:
  cd ~/benchmarks/discrete && python analysis/extract_v2_perquery_diskann.py
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS_DISKANN = ["containment", "overlap", "equality"]
DISKANN_TO_STD = {"containment": "and", "overlap": "or", "equality": "equal"}

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
    """Load ground truth from text file (space-separated IDs per line)."""
    gt = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_diskann_bin(path):
    """Read DiskANN result binary: nq uint32, k uint32, then nq*k uint32."""
    with open(path, "rb") as f:
        nq = struct.unpack("I", f.read(4))[0]
        k = struct.unpack("I", f.read(4))[0]
        ids = struct.unpack(f"{nq * k}I", f.read(nq * k * 4))
    res = [list(ids[i * k:(i + 1) * k]) for i in range(nq)]
    return res, nq, k


def process_diskann(dataset, data_dir, all_rows):
    """For each scenario, find best (R, Ls) config and extract per-query recalls."""
    base = BASE_DIR / "DiskANN" / "data_original" / "results" / dataset
    if not base.exists():
        print(f"  [DiskANN] {dataset}: no results dir, skip")
        return

    for scenario_diskann in SCENARIOS_DISKANN:
        scenario_std = DISKANN_TO_STD[scenario_diskann]

        # 找所有 (R, Ls) bin (idx_uint32.bin 格式)
        bin_files = sorted(base.glob(f"index_R=*_{scenario_diskann}_Ls=*_*_idx_uint32.bin"))
        if not bin_files:
            print(f"  [DiskANN] {dataset}/{scenario_std}: no bin files")
            continue

        # 加载 GT (DiskANN 用 .txt, K=10)
        gt_path = data_dir / dataset / f"{dataset}_gt_{scenario_std}.txt"
        if not gt_path.exists():
            print(f"  [DiskANN] GT not found: {gt_path}")
            continue
        gt = load_gt_txt(gt_path, K)

        # 选 avg recall 最高的 (R, Ls) — 一般是 R=128, Ls=2000
        best_recalls = None
        best_params = None
        best_avg = -1.0

        for bf in bin_files:
            try:
                results, nq, k = load_diskann_bin(bf)
            except Exception as e:
                print(f"  [DiskANN] {bf.name}: parse fail ({e})")
                continue
            n = min(len(gt), len(results))
            if n == 0:
                continue
            recalls = compute_perquery_recall(gt[:n], results[:n], K)
            avg_r = sum(recalls) / len(recalls)
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                # parse R, Ls from filename: index_R=128_containment_Ls=2000_2000_idx_uint32.bin
                stem = bf.name
                r_part = stem.split("_R=")[1].split("_")[0]
                ls_part = stem.split("_Ls=")[1].split("_")[0]
                best_params = f"R={r_part}_Ls={ls_part}"

        if best_recalls is None:
            print(f"  [DiskANN] {dataset}/{scenario_std}: no valid per-query data")
            continue

        for qid, r in enumerate(best_recalls):
            all_rows.append({
                "query_id": qid,
                "dataset": dataset,
                "scenario": scenario_std,
                "method": "FilteredVamana",
                "recall_at_10": r,
                "params": best_params,
            })
        print(f"  [DiskANN] {dataset}/{scenario_std}: avg recall={best_avg:.4f}, n={len(best_recalls)}, params={best_params}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="V2 datasets to process (default: all 5)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to dataset dir (default: ~/benchmarks/datasets/discrete)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: analysis/ml_router/perquery_recall/perquery_recall_v2_diskann.csv)")
    args = parser.parse_args()

    datasets = args.dataset or V2_DATASETS
    data_dir = Path(args.data_dir) if args.data_dir else Path(os.path.expanduser("~/benchmarks/datasets/discrete"))

    output_dir = BASE_DIR / "analysis" / "perquery_recall"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(output_dir / "perquery_recall_v2_diskann.csv")

    all_rows = []
    for ds in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")
        process_diskann(ds, data_dir, all_rows)

    print(f"\nWriting {len(all_rows)} rows to {output_file}")
    with open(output_file, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print("Done.")


if __name__ == "__main__":
    main()
