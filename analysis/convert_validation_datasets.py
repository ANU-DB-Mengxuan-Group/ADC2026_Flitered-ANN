"""
Convert validation datasets to formats needed by each method.

Generates:
- UNG/DiskANN/SIEVE binary vectors (.bin with N,D header)
- 1-based label files (for UNG/SIEVE)
- Binary ground truth format (pair<id,dist> no header) for UNG/DiskANN
- NHQ/CAPS label format (with N,D header)

Run on cluster:
    cd ~/benchmarks/discrete
    python analysis/convert_validation_datasets.py
"""

import struct
import numpy as np
import os

DATASETS_ROOT = os.path.expanduser("~/benchmarks/datasets/discrete")

DATASETS = {
    # --- V2 large-scale validation datasets ---
    "synth_192d": os.path.join(DATASETS_ROOT, "synth_192d"),
    "synth_512d": os.path.join(DATASETS_ROOT, "synth_512d"),
    "synth_768d_hc": os.path.join(DATASETS_ROOT, "synth_768d_hc"),
    "yahoo800k": os.path.join(DATASETS_ROOT, "yahoo800k"),
    "dbpedia560k": os.path.join(DATASETS_ROOT, "dbpedia560k"),
    # --- V1 (legacy, kept for reference) ---
    "hm21": os.path.join(DATASETS_ROOT, "hm21"),
}

SCENARIOS = ["and", "or", "equal"]


def read_fvecs(fname):
    vecs = []
    with open(fname, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            d = struct.unpack("i", buf)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32).copy()
            if len(vec) < d:
                break
            vecs.append(vec)
    return np.array(vecs)


def write_ung_bin(fname, X):
    """Write UNG binary format: uint32 N, uint32 D, then N*D float32."""
    n, d = X.shape
    with open(fname, "wb") as f:
        f.write(struct.pack("II", n, d))
        f.write(X.astype(np.float32).tobytes())


def convert_labels_to_1based(input_file, output_file):
    """Convert 0-based comma-separated labels to 1-based."""
    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            ids = [str(int(x) + 1) for x in line.strip().split(",")]
            fout.write(",".join(ids) + "\n")


def convert_gt_to_bin(gt_txt_file, gt_bin_file):
    """Convert space-separated GT to binary format (pair<id,dist> no header).
    Used by UNG, DiskANN (after further conversion), and SIEVE."""
    with open(gt_txt_file) as fin, open(gt_bin_file, "wb") as fout:
        for line in fin:
            ids = [int(x) for x in line.strip().split()]
            for id_ in ids:
                fout.write(struct.pack("if", id_, 0.0))


def convert_labels_for_nhq(input_file, output_file):
    """Convert label_base.txt to NHQ/CAPS format (adds N D header)."""
    with open(input_file) as fin:
        lines = fin.readlines()
    num_points = len(lines)
    max_attr = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for attr in line.split(","):
            try:
                val = int(attr)
                if val > max_attr:
                    max_attr = val
            except ValueError:
                pass
    num_attributes = max_attr + 1
    with open(output_file, "w") as fout:
        fout.write(f"{num_points} {num_attributes}\n")
        for line in lines:
            fout.write(line if line.endswith("\n") else line + "\n")
    print(f"  NHQ labels: {num_points} points, {num_attributes} attributes")


def main():
    for ds_name, ds_dir in DATASETS.items():
        print(f"\n=== Converting {ds_name} ===")

        if not os.path.isdir(ds_dir):
            print(f"  SKIP: directory not found: {ds_dir}")
            continue

        # 1. Convert base vectors: fvecs -> UNG bin
        fvecs_file = os.path.join(ds_dir, f"{ds_name}_base.fvecs")
        if os.path.islink(fvecs_file):
            fvecs_file = os.path.realpath(fvecs_file)

        bin_file = os.path.join(ds_dir, f"{ds_name}_base.bin")
        if os.path.exists(bin_file):
            print(f"  SKIP (exists): {bin_file}")
        else:
            print(f"  Reading {fvecs_file}...")
            X = read_fvecs(fvecs_file)
            print(f"  Vectors: {X.shape[0]} x {X.shape[1]}")
            write_ung_bin(bin_file, X)
            print(f"  Written: {bin_file}")

        # 2. Convert query vectors: fvecs -> UNG bin
        for scenario in SCENARIOS:
            qfvecs = os.path.join(ds_dir, f"{ds_name}_query_{scenario}.fvecs")
            qbin = os.path.join(ds_dir, f"{ds_name}_query_{scenario}.bin")
            if os.path.exists(qbin):
                print(f"  SKIP (exists): {qbin}")
            else:
                Q = read_fvecs(qfvecs)
                write_ung_bin(qbin, Q)
                print(f"  Written: {qbin} ({Q.shape[0]} queries)")

        # 3. Create 1-based label files
        label_file = os.path.join(ds_dir, "label_base.txt")
        label_1b = os.path.join(ds_dir, "label_base_1based.txt")
        if os.path.exists(label_1b):
            print(f"  SKIP (exists): {label_1b}")
        else:
            convert_labels_to_1based(label_file, label_1b)
            print(f"  Written: {label_1b}")

        for scenario in SCENARIOS:
            qlabel = os.path.join(ds_dir, f"{ds_name}_query_{scenario}.txt")
            qlabel_1b = os.path.join(ds_dir, f"{ds_name}_query_{scenario}_1based.txt")
            if os.path.exists(qlabel_1b):
                print(f"  SKIP (exists): {qlabel_1b}")
            else:
                convert_labels_to_1based(qlabel, qlabel_1b)
                print(f"  Written: {qlabel_1b}")

        # 4. Convert GT to binary format (used by UNG, DiskANN, SIEVE)
        for scenario in SCENARIOS:
            gt_txt = os.path.join(ds_dir, f"{ds_name}_gt_{scenario}.txt")
            gt_bin = os.path.join(ds_dir, f"{ds_name}_gt_{scenario}.bin")
            if os.path.exists(gt_bin):
                print(f"  SKIP (exists): {gt_bin}")
            else:
                convert_gt_to_bin(gt_txt, gt_bin)
                sz = os.path.getsize(gt_bin)
                print(f"  Written: {gt_bin} ({sz} bytes)")

        # 5. Create NHQ/CAPS label files (with N D header)
        nhq_base = os.path.join(ds_dir, "label_NHQ_base.txt")
        if os.path.exists(nhq_base):
            print(f"  SKIP (exists): {nhq_base}")
        else:
            convert_labels_for_nhq(label_file, nhq_base)
            print(f"  Written: {nhq_base}")

        nhq_query = os.path.join(ds_dir, "label_NHQ_query.txt")
        eq_query_label = os.path.join(ds_dir, f"{ds_name}_query_equal.txt")
        if os.path.exists(nhq_query):
            print(f"  SKIP (exists): {nhq_query}")
        elif os.path.exists(eq_query_label):
            convert_labels_for_nhq(eq_query_label, nhq_query)
            print(f"  Written: {nhq_query}")

    print("\nDone! All conversions complete.")


if __name__ == "__main__":
    main()
