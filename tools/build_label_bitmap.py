#!/usr/bin/env python3
"""Build bitmap inverted index + label-set count map for V2 datasets.

Used by online routing latency benchmark (analysis/online_routing/latency_benchmark.py).
Each query at routing time needs to compute selectivity from raw labels;
this script builds the offline data structures that make that computation O(n_base).

Output (per dataset): pickle file with
  - n_base: int (number of base vectors)
  - label_ids: sorted list of unique label ids
  - label_to_idx: dict {label_id -> row index in bitmaps}
  - bitmaps: np.ndarray shape (n_labels, n_base) dtype=bool
  - set_count: dict {frozenset(labels) -> count}   (for equality scenario)

Usage:
  python tools/build_label_bitmap.py \
      --data-root ~/benchmarks/datasets/discrete \
      --datasets synth_192d synth_512d synth_768d_hc yahoo800k dbpedia560k \
      --out-dir analysis/online_routing/bitmaps
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np


def build_for_dataset(label_base_path: Path, out_path: Path) -> dict:
    t0 = time.perf_counter()

    label_sets = []
    with open(label_base_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                label_sets.append(frozenset())
                continue
            labels = [int(x) for x in line.split(",") if x]
            label_sets.append(frozenset(labels))

    n_base = len(label_sets)
    all_labels = sorted({l for s in label_sets for l in s})
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    bitmaps = np.zeros((len(all_labels), n_base), dtype=bool)
    for vec_id, s in enumerate(label_sets):
        for l in s:
            bitmaps[label_to_idx[l], vec_id] = True

    # Pack bits: 8x smaller in memory + 8x faster bitwise ops (uint8 vs bool)
    packed_bitmaps = np.packbits(bitmaps, axis=1)  # shape (n_labels, ceil(n_base/8))

    set_count = dict(Counter(label_sets))

    data = {
        "n_base": n_base,
        "label_ids": all_labels,
        "label_to_idx": label_to_idx,
        "packed_bitmaps": packed_bitmaps,
        "set_count": set_count,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.perf_counter() - t0
    print(
        f"  n_base={n_base}, n_labels={len(all_labels)}, "
        f"unique_sets={len(set_count)}, packed_mb={packed_bitmaps.nbytes/1024/1024:.1f}, "
        f"built in {elapsed:.1f}s"
    )
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        label_path = args.data_root / ds / "label_base.txt"
        out_path = args.out_dir / f"{ds}.pkl"
        if not label_path.exists():
            print(f"[skip] {ds}: {label_path} not found")
            continue
        print(f"Building {ds}...")
        build_for_dataset(label_path, out_path)
    print(f"Done. Bitmaps written to {args.out_dir}")


if __name__ == "__main__":
    main()
