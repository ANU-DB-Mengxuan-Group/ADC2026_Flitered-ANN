"""
Generate synthetic datasets with controlled LID for routing rule validation.

Method: Generate d-dimensional Gaussian vectors, project to D=768 via random
orthogonal projection. LID ≈ d.

For each target LID, generates:
- 100K base vectors (fvecs format)
- Synthetic labels (cardinality=200, Zipf distribution)
- Queries + brute-force GT for AND/OR/Equality

Usage (on cluster):
    cd ~/benchmarks/discrete
    python analysis/generate_lid_datasets.py

Output: ~/benchmarks/datasets/discrete/lid{50,80,100,120,150}/
"""

import numpy as np
import struct
import os
import sys

# Add parent directory for generate_synthetic_labels imports
sys.path.insert(0, os.path.dirname(__file__))
from generate_synthetic_labels import (
    generate_labels, generate_queries_and_gt, save_dataset, write_fvecs
)

# Configuration
TARGET_LIDS = [50, 80, 100, 120, 150]
N_VECTORS = 100_000
AMBIENT_DIM = 768
LABEL_CARDINALITY = 200
MIN_LABELS = 1
MAX_LABELS = 5
NUM_QUERIES = 1000
K = 10
OUTPUT_ROOT = os.path.expanduser("~/benchmarks/datasets/discrete")


def generate_controlled_lid_vectors(n, intrinsic_dim, ambient_dim, seed=42):
    """
    Generate vectors with controlled LID.

    1. Sample n points from N(0, 1) in intrinsic_dim dimensions
    2. Project to ambient_dim via random orthogonal projection
    3. Result: LID ≈ intrinsic_dim
    """
    rng = np.random.RandomState(seed)

    # Generate intrinsic-dimensional data
    X_low = rng.randn(n, intrinsic_dim).astype(np.float32)

    # Random projection matrix via QR decomposition
    # Generate a random matrix and orthogonalize
    random_matrix = rng.randn(intrinsic_dim, ambient_dim).astype(np.float32)
    Q, _ = np.linalg.qr(random_matrix.T)  # Q is (ambient_dim, intrinsic_dim)
    projection = Q.T  # (intrinsic_dim, ambient_dim)

    # Project: (n, intrinsic_dim) @ (intrinsic_dim, ambient_dim) = (n, ambient_dim)
    X_high = X_low @ projection

    return X_high


def main():
    for lid in TARGET_LIDS:
        name = f"lid{lid}"
        output_dir = os.path.join(OUTPUT_ROOT, name)

        if os.path.exists(os.path.join(output_dir, f"{name}_gt_and.txt")):
            print(f"\n=== SKIP {name} (already exists) ===")
            continue

        print(f"\n{'='*50}")
        print(f"Generating {name}: LID≈{lid}, N={N_VECTORS}, D={AMBIENT_DIM}")
        print(f"{'='*50}")

        # 1. Generate vectors
        print(f"Generating {N_VECTORS} vectors (intrinsic_dim={lid} -> ambient_dim={AMBIENT_DIM})...")
        vectors = generate_controlled_lid_vectors(
            N_VECTORS, lid, AMBIENT_DIM, seed=lid * 100
        )
        print(f"  Shape: {vectors.shape}, dtype: {vectors.dtype}")
        print(f"  Norm range: [{np.linalg.norm(vectors, axis=1).min():.1f}, {np.linalg.norm(vectors, axis=1).max():.1f}]")

        # 2. Generate labels
        print(f"Generating labels (cardinality={LABEL_CARDINALITY})...")
        labels = generate_labels(
            N_VECTORS, LABEL_CARDINALITY,
            labels_per_vector=(MIN_LABELS, MAX_LABELS),
            seed=lid * 100 + 1
        )

        # 3. Generate queries and GT
        print(f"Generating queries and ground truth (k={K})...")
        query_results = generate_queries_and_gt(
            vectors, labels, NUM_QUERIES, K, seed=lid * 100 + 2
        )

        # 4. Save
        save_dataset(output_dir, name, vectors, labels, query_results, K)

    print(f"\n{'='*50}")
    print(f"All LID datasets generated!")
    print(f"Datasets: {', '.join(f'lid{l}' for l in TARGET_LIDS)}")
    print(f"Location: {OUTPUT_ROOT}/lid*/")
    print(f"\nNext: run convert and benchmark scripts")


if __name__ == "__main__":
    main()
