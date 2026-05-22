#!/usr/bin/env python3
"""
Compute Intra-Inter Cluster Ratio (IICR) for a discrete dataset.

- Intra-class compactness: 平均 ||x - μ_c||
- Inter-class separation: 平均 ||μ_i - μ_j||
- IICR = intra / inter  (越小越好)

用法示例：
python tools/compute_iicr.py \
    --data-root /home/remote/u7905817/benchmarks/datasets/discrete \
    --dataset arxiv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

def parse_pair_cap(value):
    if value is None:
        return None
    lower = value.lower()
    if lower == "none":
        return None
    return int(value)

def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{path} 为空")
    dim = raw.view(np.int32)[0]
    if dim <= 0:
        raise ValueError(f"{path} 维度非法: {dim}")
    try:
        vectors = raw.reshape(-1, dim + 1)[:, 1:]
    except ValueError as exc:
        raise ValueError(f"{path} reshape 失败，可能文件损坏") from exc
    return vectors


def read_flat_bin(path: Path, dim: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % dim != 0:
        raise ValueError(f"{path} 长度不能被 dim={dim} 整除")
    return raw.reshape(-1, dim)


def load_vectors(path: Path, fmt: str, dim: int | None) -> np.ndarray:
    if fmt == "fvecs":
        return read_fvecs(path)
    if fmt == "bin":
        if dim is None:
            raise ValueError("--vector-dim 必须与 bin 格式一起使用")
        return read_flat_bin(path, dim)
    raise ValueError(f"未知格式: {fmt}")


def load_labels(path: Path) -> List[List[int]]:
    labels: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                labels.append([])
                continue
            labels.append([int(tok) for tok in line.split(",") if tok])
    return labels


def build_class_membership(
    labels: Sequence[Sequence[int]],
    mode: str,
    min_size: int,
) -> Dict[Tuple[int, ...], List[int]]:
    buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx, raw in enumerate(labels):
        if not raw:
            continue
        if mode == "label":
            for token in raw:
                buckets[(token,)].append(idx)
        else:  # combo
            buckets[tuple(sorted(raw))].append(idx)
    # 剔除太小的类，避免噪声
    for key in list(buckets.keys()):
        if len(buckets[key]) < min_size:
            buckets.pop(key)
    if not buckets:
        raise ValueError("没有符合条件的类，调低 --min-class-size 试试")
    return buckets


def compute_intra(vectors: np.ndarray, membership: Dict[Tuple[int, ...], List[int]]) -> float:
    total_dist = 0.0
    total_pts = 0
    for idxs in membership.values():
        pts = vectors[idxs]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        total_dist += float(dists.sum())
        total_pts += len(idxs)
    return total_dist / max(total_pts, 1)


def average_pairwise_distance(
    centers: np.ndarray,
    max_pairs: int | None,
    seed: int,
) -> float:
    n = centers.shape[0]
    if n < 2:
        return float("nan")
    total_pairs = n * (n - 1) // 2
    if max_pairs is None or max_pairs >= total_pairs:
        total = 0.0
        count = 0
        for i in range(n - 1):
            block = centers[i + 1 :] - centers[i]
            dists = np.linalg.norm(block, axis=1)
            total += float(dists.sum())
            count += dists.shape[0]
        return total / count
    # 随机采样配对以避免 O(n^2)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n, size=max_pairs)
    idx_b = rng.integers(0, n, size=max_pairs)
    mask = idx_a != idx_b
    while mask.sum() == 0:
        idx_b = rng.integers(0, n, size=max_pairs)
        mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    dists = np.linalg.norm(centers[idx_a] - centers[idx_b], axis=1)
    return float(dists.mean())


def compute_inter(
    vectors: np.ndarray,
    membership: Dict[Tuple[int, ...], List[int]],
    pair_cap: int | None,
    seed: int,
) -> float:
    centers = np.stack([vectors[idxs].mean(axis=0) for idxs in membership.values()])
    return average_pairwise_distance(centers, pair_cap, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Intra-Inter Cluster Ratio")
    parser.add_argument("--data-root", required=True, type=Path, help="datasets/discrete 目录")
    parser.add_argument("--dataset", required=True, help="子目录名，例如 arxiv")
    parser.add_argument("--base-file", default=None, type=Path, help="可显式指定 base 向量文件")
    parser.add_argument("--vector-format", choices=("fvecs", "bin"), default="fvecs")
    parser.add_argument("--vector-dim", type=int, default=None, help="仅在 bin 格式时需要")
    parser.add_argument("--label-file", default=None, type=Path, help="默认是 label_base.txt")
    parser.add_argument("--class-mode", choices=("label", "combo"), default="label",
                        help="'label' 把每个标签值当类；'combo' 把整行组合当类")
    parser.add_argument("--min-class-size", type=int, default=2,
                        help="小于此大小的类会被忽略，避免噪声")
    parser.add_argument("--max-inter-pairs", type=parse_pair_cap, default=200_000,
                        help="限制 inter-class 采样对数；None 表示用全部配对（可能非常慢）")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子")
    args = parser.parse_args()

    dataset_dir = args.data_root / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    output_path = args.data_root / f"{args.dataset}_iicr.txt"

    base_path = args.base_file
    if base_path is None:
        suffix = ".fvecs" if args.vector_format == "fvecs" else ".bin"
        base_path = dataset_dir / f"{args.dataset}_base{suffix}"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    label_path = args.label_file or (dataset_dir / "label_base.txt")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    vectors = load_vectors(base_path, args.vector_format, args.vector_dim)
    labels = load_labels(label_path)
    if len(labels) != len(vectors):
        raise ValueError(f"向量数 {len(vectors)} 与标签行数 {len(labels)} 不一致")

    membership = build_class_membership(labels, args.class_mode, args.min_class_size)
    intra = compute_intra(vectors, membership)
    inter = compute_inter(vectors, membership, args.max_inter_pairs, args.seed)
    ratio = intra / inter

    print(f"dataset          : {args.dataset}")
    print(f"class_mode       : {args.class_mode}")
    print(f"#vectors         : {len(vectors)}")
    print(f"#classes used    : {len(membership)}")
    print(f"intra_compactness: {intra:.6f}")
    print(f"inter_separation : {inter:.6f}")
    print(f"IICR             : {ratio:.6f}")
    total_pairs = len(membership) * (len(membership) - 1) // 2
    sampled = args.max_inter_pairs is not None and args.max_inter_pairs < total_pairs
    if sampled:
        print(f"(inter-class separation 是基于随机采样 {args.max_inter_pairs} 对估计的)")

    lines = [
        f"dataset: {args.dataset}",
        f"class_mode: {args.class_mode}",
        f"#vectors: {len(vectors)}",
        f"#classes_used: {len(membership)}",
        f"intra_compactness: {intra:.6f}",
        f"inter_separation: {inter:.6f}",
        f"IICR: {ratio:.6f}",
    ]
    if sampled:
        lines.append(f"inter_separation_estimated_with: {args.max_inter_pairs} pairs")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"结果已写入 {output_path}")

if __name__ == "__main__":
    main()