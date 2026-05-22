#!/usr/bin/env python3
"""
V2 per-query recall 提取 (UNG + Post-filter only).

V2 数据集 (5 个): synth_192d, synth_512d, synth_768d_hc, yahoo800k, dbpedia560k

UNG V2 per-query 数据在:
  UNG-dev/results_v2_grid/{dataset}/{scenario_ung}/M=*_Lb=*_Ls=*_ce=6result_L*.csv
  scenario_ung: containment/overlap/equality (映射到 and/or/equal)
  CSV 格式: GT,Result (每行 1 个 query, 空格分隔的 ID)

Post-filter V2 per-query 数据在:
  faiss/results_postfilter/{dataset}/{scenario}/M=64_efc=400_ef=*_idx_uint32.bin
  二进制: nq uint32, k uint32, then nq*k uint32 (top-k IDs per query)

SIEVE V2: 暂时跳过 (没 per-query 数据, 要单独跑 run_sieve_perquery.py 加 V2 支持)

输出: analysis/ml_router/perquery_recall/perquery_recall_v2.csv
  字段: query_id, dataset, scenario, method, recall_at_10, params

用法:
  cd ~/benchmarks/discrete
  python analysis/extract_v2_perquery_recall.py
  python analysis/extract_v2_perquery_recall.py --dataset synth_192d
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import struct
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS_STD = ["and", "or", "equal"]
UNG_TO_STD = {"containment": "and", "overlap": "or", "equality": "equal"}
STD_TO_UNG = {v: k for k, v in UNG_TO_STD.items()}

K = 10


# ==================== Per-query recall ====================

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


# ==================== UNG ====================

def parse_ung_result_csv(path):
    """Parse UNG result CSV (GT,Result format). Returns (gt_list, res_list)."""
    gt_list, res_list = [], []
    with open(path) as f:
        header = f.readline().strip()
        if header.split(",")[:2] != ["GT", "Result"]:
            return None, None
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            gt = [int(x) for x in parts[0].strip().split() if x]
            res = [int(x) for x in parts[1].strip().split() if x]
            gt_list.append(gt)
            res_list.append(res)
    return gt_list, res_list


def process_ung_v2(dataset, all_rows):
    """For each scenario, find best UNG config (highest avg recall) per-query CSV, extract."""
    base = BASE_DIR / "UNG-dev" / "results_v2_grid" / dataset
    if not base.exists():
        print(f"  [UNG] {dataset}: no results_v2_grid dir, skip")
        return

    for scenario_std in SCENARIOS_STD:
        scenario_ung = STD_TO_UNG[scenario_std]
        sc_dir = base / scenario_ung
        if not sc_dir.exists():
            print(f"  [UNG] {dataset}/{scenario_std}: scenario dir not found, skip")
            continue

        # 找所有 _L*.csv per-query 文件 (注意 _result.csv 是 aggregate, 跳过)
        per_query_files = sorted(sc_dir.glob("*result_L*.csv"))
        if not per_query_files:
            print(f"  [UNG] {dataset}/{scenario_std}: no per-query CSVs")
            continue

        # 选 avg recall 最高的那个 config
        best_recalls = None
        best_params = None
        best_avg = -1.0

        for f in per_query_files:
            gt_list, res_list = parse_ung_result_csv(f)
            if not gt_list:
                continue
            recalls = compute_perquery_recall(gt_list, res_list, K)
            if not recalls:
                continue
            avg_r = sum(recalls) / len(recalls)
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                best_params = f.stem  # filename without ext

        if best_recalls is None:
            print(f"  [UNG] {dataset}/{scenario_std}: no valid per-query data")
            continue

        for qid, r in enumerate(best_recalls):
            all_rows.append({
                "query_id": qid,
                "dataset": dataset,
                "scenario": scenario_std,
                "method": "UNG",
                "recall_at_10": r,
                "params": best_params,
            })
        print(f"  [UNG] {dataset}/{scenario_std}: avg recall={best_avg:.4f}, n={len(best_recalls)}, params={best_params}")


# ==================== Post-filter ====================

def load_gt_txt(path, k=10):
    """Load ground truth from text file (space-separated IDs per line)."""
    gt = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt.append([int(x) for x in parts[:k]])
    return gt


def load_pf_result_bin(path):
    """Read Post-filter binary: nq uint32, k uint32, then nq*k uint32."""
    with open(path, "rb") as f:
        nq = struct.unpack("I", f.read(4))[0]
        k = struct.unpack("I", f.read(4))[0]
        ids = struct.unpack(f"{nq * k}I", f.read(nq * k * 4))
    res = [list(ids[i * k:(i + 1) * k]) for i in range(nq)]
    return res, nq, k


def process_postfilter_v2(dataset, data_dir, all_rows):
    base = BASE_DIR / "faiss" / "results_postfilter" / dataset
    if not base.exists():
        print(f"  [PF] {dataset}: no results_postfilter dir")
        return

    for scenario_std in SCENARIOS_STD:
        sc_dir = base / scenario_std
        if not sc_dir.exists():
            print(f"  [PF] {dataset}/{scenario_std}: dir not found, skip")
            continue

        # 找 M=64_efc=400 的 .bin 文件 (V2 全部跑这个 config)
        bin_files = sorted(sc_dir.glob("M=64_efc=400_ef=*_idx_uint32.bin"))
        if not bin_files:
            print(f"  [PF] {dataset}/{scenario_std}: no M=64_efc=400 bin files")
            continue

        # 加载 GT
        gt_path = data_dir / dataset / f"{dataset}_gt_{scenario_std}.txt"
        if not gt_path.exists():
            print(f"  [PF] GT not found: {gt_path}")
            continue
        gt = load_gt_txt(gt_path, K)

        # 选 avg recall 最高的 ef_search
        best_recalls = None
        best_params = None
        best_avg = -1.0

        for bf in bin_files:
            results, nq, k = load_pf_result_bin(bf)
            n = min(len(gt), len(results))
            if n == 0:
                continue
            recalls = compute_perquery_recall(gt[:n], results[:n], K)
            avg_r = sum(recalls) / len(recalls)
            if avg_r > best_avg:
                best_avg = avg_r
                best_recalls = recalls
                ef = bf.name.split("_ef=")[1].split("_")[0]
                best_params = f"M=64_efc=400_ef={ef}"

        if best_recalls is None:
            print(f"  [PF] {dataset}/{scenario_std}: no valid per-query data")
            continue

        for qid, r in enumerate(best_recalls):
            all_rows.append({
                "query_id": qid,
                "dataset": dataset,
                "scenario": scenario_std,
                "method": "Post-filter",
                "recall_at_10": r,
                "params": best_params,
            })
        print(f"  [PF] {dataset}/{scenario_std}: avg recall={best_avg:.4f}, n={len(best_recalls)}, params={best_params}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="V2 datasets to process (default: all 5)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to dataset dir (default: ~/benchmarks/datasets/discrete)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: analysis/ml_router/perquery_recall/perquery_recall_v2.csv)")
    args = parser.parse_args()

    datasets = args.dataset or V2_DATASETS
    data_dir = Path(args.data_dir) if args.data_dir else Path(os.path.expanduser("~/benchmarks/datasets/discrete"))

    output_dir = BASE_DIR / "analysis" / "perquery_recall"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(output_dir / "perquery_recall_v2.csv")

    all_rows = []
    for ds in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")
        process_ung_v2(ds, all_rows)
        process_postfilter_v2(ds, data_dir, all_rows)

    print(f"\nWriting {len(all_rows)} rows to {output_file}")
    with open(output_file, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print("Done.")


if __name__ == "__main__":
    main()
