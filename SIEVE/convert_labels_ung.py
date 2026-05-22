"""
Convert UNG/V2-format labels to SIEVE format.

Auto-detects 1-based (V1, e.g. arxiv/yfcc) vs 0-based (V2, e.g. yahoo800k/dbpedia560k):
  - If min label across entire file is 1 → 1-based → subtract 1 to get 0-based
  - If min label is 0 → 0-based already → no shift

Original V1-only behavior assumed 1-based and unconditionally did label-1, which
turned 0-based labels into -1 (uint32 -> 4294967295), causing segfault inside
SIEVE/hnswlib/filters.h transpose_inplace() with new_row_offsets[0xFFFFFFFF+1] OOB.
"""
import argparse
import struct
import pickle
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def read_ung_label_file(filepath, force_mode=None):
    """Read comma-separated label file. Returns 0-based lists.

    force_mode: None (auto-detect) | "0based" | "1based"
    """
    raw = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(sorted(int(x) for x in line.split(',')))
            else:
                raw.append([])

    # Detect mode by min label across all rows
    nonempty_min = None
    for row in raw:
        if row:
            m = row[0]   # already sorted
            if nonempty_min is None or m < nonempty_min:
                nonempty_min = m

    if force_mode == "0based":
        is_zero_based = True
    elif force_mode == "1based":
        is_zero_based = False
    else:
        # Auto: min == 0 → 0-based; min == 1 → 1-based; else assume 1-based
        is_zero_based = (nonempty_min == 0)

    print(f"[convert_labels_ung] {filepath}: min_label={nonempty_min}, "
          f"detected={'0-based' if is_zero_based else '1-based'}")

    if is_zero_based:
        return raw
    return [[x - 1 for x in row] for row in raw]


def write_binary_csr(labels, output_path):
    """Write labels to SIEVE's binary CSR format."""
    n_points = len(labels)
    max_label = max(max(row) for row in labels if row) + 1 if labels else 0
    n_filters = max_label
    n_nonzero = sum(len(row) for row in labels)

    row_offsets = np.zeros(n_points + 1, dtype=np.uint64)
    row_indices = np.zeros(n_nonzero, dtype=np.uint32)

    offset = 0
    for i, row in enumerate(labels):
        row_offsets[i] = offset
        for label_id in sorted(row):
            row_indices[offset] = label_id
            offset += 1
    row_offsets[n_points] = offset

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', n_points))
        f.write(struct.pack('<Q', n_filters))
        f.write(struct.pack('<Q', n_nonzero))
        f.write(row_offsets.tobytes())
        f.write(row_indices.tobytes())

    print(f"Written binary CSR: {n_points} points, {n_filters} filters, "
          f"{n_nonzero} nonzeros -> {output_path}")
    return n_filters


def write_query_pickle(labels, n_filters, output_path):
    """Write query labels as scipy sparse matrix pickle."""
    n_queries = len(labels)
    mat = lil_matrix((n_queries, n_filters), dtype=np.float32)
    for i, row in enumerate(labels):
        for label_id in row:
            if label_id < n_filters:
                mat[i, label_id] = 1.0
    sparse_mat = csr_matrix(mat)
    with open(output_path, 'wb') as f:
        pickle.dump(sparse_mat, f)
    print(f"Written query pickle: {n_queries} queries, {n_filters} filters -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert UNG labels (1-based) to SIEVE format')
    parser.add_argument('--base_labels', required=True)
    parser.add_argument('--query_labels', required=True)
    parser.add_argument('--output_base', required=True)
    parser.add_argument('--output_query', required=True)
    args = parser.parse_args()

    base_labels = read_ung_label_file(args.base_labels)
    n_filters = write_binary_csr(base_labels, args.output_base)

    query_labels = read_ung_label_file(args.query_labels)
    write_query_pickle(query_labels, n_filters, args.output_query)


if __name__ == '__main__':
    main()
