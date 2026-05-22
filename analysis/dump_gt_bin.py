#!/usr/bin/env python3
"""Dump GT bin file contents for debugging"""
import struct
import os

gt_path = os.path.expanduser("~/benchmarks/datasets/discrete/synth_192d/synth_192d_gt_and.bin")

with open(gt_path, "rb") as f:
    data = f.read(52)  # Header (8) + first 11 ints (44)

print(f"File: {gt_path}")
print(f"File size: {os.path.getsize(gt_path)} bytes")
print("\nRaw int32 values (first 13):")
for i in range(0, 52, 4):
    val = struct.unpack('i', data[i:i+4])[0]
    if i == 0:
        print(f"  offset {i:2d}: {val}  <- N (num queries)")
    elif i == 4:
        print(f"  offset {i:2d}: {val}  <- K (neighbors per query)")
    else:
        print(f"  offset {i:2d}: {val}  <- neighbor {(i-8)//4}")

print("\nExpected format: [N=1000][K=10][441736][139182][388172]...")
print("\nUNG reads:       1000 441736 388172 347405 698724 306451 231616 307566 188369 269405")
print("Our GT query 0:  441736 139182 388172 239325 347405 551441 698724 584423 306451 279172")
