#!/usr/bin/env python3
"""
Recompute ACORN recall for ytb_video AND scenario using corrected GT.

Background: ytb_video_query_and.bin had different vectors from .fvecs.
GT was computed from .bin, but ACORN searches with .fvecs.
Old GT was wrong; new GT has been regenerated from correct .fvecs queries.
ACORN search results (in search.log) are valid — just need recall recomputed.
"""

import os
import re
import csv
from pathlib import Path

# Paths
GT_FILE = Path(__file__).parent / "ytb_video_gt_and_new.txt"
ACORN_RESULTS_DIR = Path(__file__).parent.parent / "ACORN/data/param_search_ytb_video/results/ytb_video/and"

NQ = 200
K = 10


def load_gt(path):
    """Load GT file: one line per query, space-separated IDs."""
    gt = []
    with open(path) as f:
        for line in f:
            ids = set(int(x) for x in line.strip().split())
            gt.append(ids)
    assert len(gt) == NQ, f"Expected {NQ} queries, got {len(gt)}"
    return gt


def parse_search_log(path):
    """Parse ACORN search.log to extract per-efSearch result blocks.

    Format: after each "ACORN INDEX (efSearch=XX)" header,
    there are exactly NQ lines of space-separated result IDs.
    """
    results = {}  # efSearch -> list of NQ sets of result IDs

    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for efSearch header
        m = re.search(r'efSearch=(\d+)', line)
        if m:
            ef = int(m.group(1))
            # Skip header lines until we hit "Recall@"
            i += 1
            while i < len(lines) and 'Recall@' not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            i += 1  # skip the "Recall@: 10" line

            # Now read NQ lines of result IDs
            block = []
            for q in range(NQ):
                if i >= len(lines):
                    break
                ids_line = lines[i].strip()
                if ids_line:
                    ids = set()
                    for x in ids_line.split():
                        val = int(x)
                        if val >= 0:  # skip -1 markers
                            ids.add(val)
                    block.append(ids)
                else:
                    block.append(set())
                i += 1

            if len(block) == NQ:
                results[ef] = block
        else:
            i += 1

    return results


def compute_recall(gt, results):
    """Compute recall@K: average fraction of GT IDs found in results."""
    total = 0
    for q in range(NQ):
        if len(gt[q]) > 0:
            hits = len(results[q] & gt[q])
            total += hits / min(K, len(gt[q]))
    return total / NQ


def main():
    print(f"Loading new GT from {GT_FILE}")
    gt = load_gt(GT_FILE)
    print(f"  Loaded {len(gt)} queries, avg GT size: {sum(len(g) for g in gt)/len(gt):.1f}")

    # Find all search.log files
    log_files = sorted(ACORN_RESULTS_DIR.glob("*_search.log"))
    print(f"\nFound {len(log_files)} ACORN search.log files")

    # Also load old recall from result CSV for comparison
    all_results = []

    for log_path in log_files:
        # Extract config from filename: M=XX_M_beta=XX_gamma=XX_search.log
        name = log_path.stem.replace("_search", "")

        # Parse old recall from CSV
        csv_path = log_path.with_name(name + "_result.csv")
        old_recalls = {}
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ef = int(row['L'])
                    old_recalls[ef] = float(row['Recall'])

        # Parse search results
        ef_results = parse_search_log(log_path)

        if not ef_results:
            print(f"  WARNING: No results parsed from {log_path.name}")
            continue

        print(f"\n  Config: {name} ({len(ef_results)} efSearch values)")

        for ef in sorted(ef_results.keys()):
            new_recall = compute_recall(gt, ef_results[ef])
            old_recall = old_recalls.get(ef, None)
            old_str = f"{old_recall:.4f}" if old_recall is not None else "N/A"
            delta = f" (Δ={new_recall - old_recall:+.4f})" if old_recall is not None else ""
            all_results.append({
                'config': name,
                'efSearch': ef,
                'old_recall': old_recall,
                'new_recall': new_recall,
            })
            if ef in [10, 50, 100, 500, 1000, 3000]:  # print selected values
                print(f"    efSearch={ef:5d}: old={old_str}, new={new_recall:.4f}{delta}")

    # Summary: write corrected results to CSV
    out_path = Path(__file__).parent / "ytb_video_and_acorn_corrected_recall.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'efSearch', 'old_recall', 'new_recall'])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nSaved corrected recall to {out_path}")

    # Print best recall per config
    print("\n=== Best recall per config ===")
    configs = sorted(set(r['config'] for r in all_results))
    for cfg in configs:
        cfg_results = [r for r in all_results if r['config'] == cfg]
        best = max(cfg_results, key=lambda r: r['new_recall'])
        old_best = max(cfg_results, key=lambda r: r['old_recall'] or 0)
        print(f"  {cfg}: old_best={old_best['old_recall']:.4f} (ef={old_best['efSearch']}), "
              f"new_best={best['new_recall']:.4f} (ef={best['efSearch']})")


if __name__ == "__main__":
    main()
