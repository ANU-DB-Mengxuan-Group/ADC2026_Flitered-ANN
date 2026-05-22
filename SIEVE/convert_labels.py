"""
Convert our benchmark label files to SIEVE's binary CSR format.

Our format: text file, one line per point, comma-separated label IDs
    e.g., "2,3,6,11" or "7,19"

SIEVE's DatasetFilters binary format:
    - uint64 n_points
    - uint64 n_filters (max label ID + 1)
    - uint64 n_nonzero (total label entries)
    - uint64[n_points + 1] row_offsets
    - uint32[n_nonzero] row_indices (sorted within each row)

Also generates query filter pickle files (scipy sparse matrix) for SIEVE's
historical workload.

Usage:
    python convert_labels.py --base_labels <path> --query_labels <path> \
        --output_base <path> --output_query <path> [--is_and]
"""
import argparse
import struct
import pickle
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def read_label_file(filepath):
    """Read comma-separated label file, return list of lists of int."""
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(sorted(int(x) for x in line.split(',')))
            else:
                labels.append([])
    return labels


def write_binary_csr(labels, output_path):
    """Write labels to SIEVE's binary CSR format."""
    n_points = len(labels)
    max_label = max(max(row) for row in labels if row) + 1 if labels else 0
    n_filters = max_label
    n_nonzero = sum(len(row) for row in labels)

    # Build CSR arrays
    row_offsets = np.zeros(n_points + 1, dtype=np.uint64)
    row_indices = np.zeros(n_nonzero, dtype=np.uint32)

    offset = 0
    for i, row in enumerate(labels):
        row_offsets[i] = offset
        for label_id in sorted(row):
            row_indices[offset] = label_id
            offset += 1
    row_offsets[n_points] = offset

    assert offset == n_nonzero

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
    """Write query labels as scipy sparse matrix pickle (SIEVE's format)."""
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
    parser = argparse.ArgumentParser(description='Convert labels to SIEVE format')
    parser.add_argument('--base_labels', required=True, help='Path to base label file (comma-separated)')
    parser.add_argument('--query_labels', required=True, help='Path to query label file (comma-separated)')
    parser.add_argument('--output_base', required=True, help='Output path for binary CSR base labels')
    parser.add_argument('--output_query', required=True, help='Output path for query pickle')
    args = parser.parse_args()

    # Convert base labels
    base_labels = read_label_file(args.base_labels)
    n_filters = write_binary_csr(base_labels, args.output_base)

    # Convert query labels
    query_labels = read_label_file(args.query_labels)
    write_query_pickle(query_labels, n_filters, args.output_query)


if __name__ == '__main__':
    main()
