#!/usr/bin/env python3
"""V2 数据集 UNG 实验 pipeline

1. 重新生成 GT（UNG 格式：无 header，(ID,dist) pairs）
2. 跑 UNG 实验
3. 输出结果用于路由验证
"""
import os
import sys
import subprocess

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
DATA_DIR = os.path.expanduser("~/benchmarks/datasets/discrete")
UNG_DIR = os.path.expanduser("~/benchmarks/discrete/UNG-dev")

def run(cmd, desc=None):
    """Run command and print output"""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def verify_gt_format(dataset):
    """验证 GT 文件大小是否符合 UNG 格式"""
    import struct

    for scenario in ["and", "or", "equal"]:
        gt_file = f"{DATA_DIR}/{dataset}/{dataset}_gt_{scenario}.bin"
        if not os.path.exists(gt_file):
            print(f"  [{scenario}] GT not found")
            continue

        size = os.path.getsize(gt_file)
        # UNG format: N * K * 8 bytes (pairs of int32+float32)
        # 1000 queries * 10 neighbors * 8 = 80000
        expected = 1000 * 10 * 8

        if size == expected:
            print(f"  [{scenario}] OK: {size} bytes (UNG format)")
        else:
            print(f"  [{scenario}] WARNING: {size} bytes, expected {expected}")
            # Check first entry
            with open(gt_file, 'rb') as f:
                data = f.read(16)
            if len(data) >= 16:
                id1, dist1, id2, dist2 = struct.unpack('ifif', data)
                print(f"    First 2 entries: ({id1}, {dist1:.4f}), ({id2}, {dist2:.4f})")

def main():
    print("="*60)
    print("V2 UNG Pipeline")
    print("="*60)

    # Step 1: Generate GT
    if "--skip-gt" not in sys.argv:
        for ds in V2_DATASETS:
            print(f"\n>>> Generating GT for {ds}...")
            if not run(f"python ~/benchmarks/discrete/analysis/regenerate_gt_1based.py {ds}"):
                print(f"ERROR: GT generation failed for {ds}")
                continue

            print(f"\n>>> Verifying GT format for {ds}...")
            verify_gt_format(ds)
    else:
        print("\n>>> Skipping GT generation (--skip-gt)")

    # Step 2: Run UNG
    if "--skip-ung" not in sys.argv:
        print("\n>>> Running UNG experiments...")
        ung_script = f"{UNG_DIR}/bash/auto_ung_original.py"

        # Modify to run only V2 datasets
        run(f"cd {UNG_DIR} && python bash/auto_ung_original.py", "Running UNG on V2 datasets")
    else:
        print("\n>>> Skipping UNG (--skip-ung)")

    # Step 3: Collect results
    print("\n>>> Results summary")
    result_dir = f"{UNG_DIR}/results_original"
    if os.path.exists(result_dir):
        for ds in V2_DATASETS:
            print(f"\n  {ds}:")
            for scenario in ["and", "or", "equal"]:
                # Find latest result file
                import glob
                pattern = f"{result_dir}/*{ds}*{scenario}*.log"
                files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                if files:
                    # Extract recall from log
                    with open(files[0]) as f:
                        content = f.read()
                    if "recall" in content.lower():
                        for line in content.split('\n'):
                            if 'recall' in line.lower() and any(c.isdigit() for c in line):
                                print(f"    [{scenario}] {line.strip()[:80]}")
                                break
                else:
                    print(f"    [{scenario}] No results yet")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()
