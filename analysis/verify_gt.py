#!/usr/bin/env python3
"""验证 GT 是否正确生成"""
import struct
import os
import numpy as np

ds_dir = os.path.expanduser("~/benchmarks/datasets/discrete/synth_192d")

print("=" * 60)
print("验证 synth_192d GT")
print("=" * 60)

# 1. Query vectors count
qpath = f"{ds_dir}/synth_192d_query_and.fvecs"
with open(qpath, 'rb') as f:
    d = struct.unpack('i', f.read(4))[0]
fsize = os.path.getsize(qpath)
n_query_vecs = fsize // (4 + d * 4)
print(f"\n1. Query vectors: {n_query_vecs}, dim: {d}")

# 2. Query labels count
with open(f"{ds_dir}/synth_192d_query_and_1based.txt") as f:
    query_labels = [set(int(x) for x in line.strip().split(',') if x) for line in f]
print(f"2. Query labels: {len(query_labels)}")

# 3. Base labels
with open(f"{ds_dir}/label_base_1based.txt") as f:
    base_labels = [set(int(x) for x in line.strip().split(',') if x) for line in f]
print(f"3. Base labels: {len(base_labels)}")

# 4. First query's labels
print(f"\n4. First query labels (1-based): {query_labels[0]}")

# 5. New GT first query neighbors
gt_path = f"{ds_dir}/synth_192d_gt_and.bin"
with open(gt_path, 'rb') as f:
    n, k = struct.unpack('II', f.read(8))
    first_gt = struct.unpack(f'{k}i', f.read(k * 4))
print(f"5. New GT first query neighbors: {first_gt}")

# 6. Check if GT neighbors satisfy constraint
print(f"\n6. 验证前 3 个邻居是否满足 AND 约束:")
for i, nid in enumerate(first_gt[:3]):
    if nid < 0 or nid >= len(base_labels):
        print(f"   邻居 {i}: ID={nid} (无效)")
        continue
    b_labels = base_labels[nid]
    q_labels = query_labels[0]
    satisfies = q_labels.issubset(b_labels)
    print(f"   邻居 {i}: ID={nid}, labels={b_labels}")
    print(f"            query={q_labels} ⊆ base? {satisfies}")

# 7. Old GT comparison
old_gt_path = f"{ds_dir}/synth_192d_gt_and_0based.bin"
if os.path.exists(old_gt_path):
    with open(old_gt_path, 'rb') as f:
        n, k = struct.unpack('II', f.read(8))
        old_first_gt = struct.unpack(f'{k}i', f.read(k * 4))
    print(f"\n7. Old GT first query neighbors: {old_first_gt}")

    # Check overlap
    new_set = set(first_gt)
    old_set = set(old_first_gt)
    overlap = new_set & old_set
    print(f"   Overlap: {len(overlap)}/10 ({overlap})")

# 8. Load base vectors and verify distances
print(f"\n8. 验证距离排序:")
# Read base vectors
with open(f"{ds_dir}/synth_192d_base.fvecs", 'rb') as f:
    d = struct.unpack('i', f.read(4))[0]
base_data = np.fromfile(f"{ds_dir}/synth_192d_base.fvecs", dtype=np.float32).reshape(-1, d+1)[:, 1:]

# Read query vectors
with open(qpath, 'rb') as f:
    d = struct.unpack('i', f.read(4))[0]
query_data = np.fromfile(qpath, dtype=np.float32).reshape(-1, d+1)[:, 1:]

q_vec = query_data[0]
print(f"   Query 0 vector shape: {q_vec.shape}")

# Compute distances to GT neighbors
for i, nid in enumerate(first_gt[:5]):
    if nid >= 0 and nid < len(base_data):
        dist = np.sum((base_data[nid] - q_vec) ** 2)
        print(f"   GT neighbor {i}: ID={nid}, dist²={dist:.4f}")

print("\n" + "=" * 60)
