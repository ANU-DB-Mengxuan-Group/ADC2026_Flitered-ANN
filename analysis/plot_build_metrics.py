#!/usr/bin/env python3
"""
Plot build time and index size comparison across methods.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'font.family': 'sans-serif',
})

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
DATASET_NAMES = {
    'arxiv': 'arXiv',
    'yfcc': 'YFCC',
    'LAION1M': 'LAION-1M',
    'tripclick': 'TripClick',
    'ytb_audio': 'YTB-Audio',
    'ytb_video': 'YTB-Video',
}

COLORS = {
    'ACORN-1': '#1f77b4',
    'ACORN-γ': '#ff7f0e',
    'Pre-filter': '#2ca02c',
    'Post-filter': '#d62728',
    'UNG': '#9467bd',
    'CAPS': '#8c564b',
    'NHQ': '#e377c2',
    'SIEVE': '#17becf',
}


def load_build_metrics():
    """Load build time and index size from all methods."""
    results = []

    for dataset in DATASETS:
        # ACORN-γ
        path = BASE_DIR / f"ACORN/data/param_search_{dataset}/summary.csv"
        if not path.exists():
            path = BASE_DIR / f"ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success']
            if not df.empty:
                # Get unique build configs (same M, M_beta, gamma = same index)
                # Take the first entry for each unique index config
                if 'M' in df.columns and 'gamma' in df.columns:
                    df_unique = df.drop_duplicates(subset=['M', 'gamma'] if 'M_beta' not in df.columns else ['M', 'M_beta', 'gamma'])
                else:
                    df_unique = df.head(1)
                for _, row in df_unique.iterrows():
                    results.append({
                        'dataset': dataset,
                        'method': 'ACORN-γ',
                        'build_time_s': row.get('build_time_s', 0),
                        'index_size_mb': row.get('index_size_mb', 0),
                    })

        # ACORN-1
        path = BASE_DIR / f"ACORN/data/param_search_{dataset}_gamma1/results/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success']
            if not df.empty:
                if 'M' in df.columns:
                    df_unique = df.drop_duplicates(subset=['M'])
                else:
                    df_unique = df.head(1)
                for _, row in df_unique.iterrows():
                    results.append({
                        'dataset': dataset,
                        'method': 'ACORN-1',
                        'build_time_s': row.get('build_time_s', 0),
                        'index_size_mb': row.get('index_size_mb', 0),
                    })

        # UNG
        path = BASE_DIR / f"UNG-dev/results_original/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success']
            if not df.empty:
                if 'max_degree' in df.columns:
                    df_unique = df.drop_duplicates(subset=['max_degree'])
                else:
                    df_unique = df.head(1)
                for _, row in df_unique.iterrows():
                    results.append({
                        'dataset': dataset,
                        'method': 'UNG',
                        'build_time_s': row.get('build_time_s', 0),
                        'index_size_mb': row.get('index_size_mb', 0),
                    })

        # CAPS
        path = BASE_DIR / f"CAPS/data/results/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success']
            if not df.empty and 'build_time_s' in df.columns:
                df_unique = df.drop_duplicates(subset=['nb']) if 'nb' in df.columns else df.head(1)
                for _, row in df_unique.iterrows():
                    if row.get('build_time_s', 0) > 0:
                        results.append({
                            'dataset': dataset,
                            'method': 'CAPS',
                            'build_time_s': row.get('build_time_s', 0),
                            'index_size_mb': row.get('index_size_mb', 0),
                        })

        # NHQ (synthetic data)
        path = BASE_DIR / f"NHQ/data_synthetic/results/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty and 'build_time_s' in df.columns:
                # NHQ has same build for different L values
                df_unique = df.drop_duplicates(subset=['params']) if 'params' in df.columns else df.head(1)
                for _, row in df_unique.iterrows():
                    if row.get('build_time_s', 0) > 0:
                        results.append({
                            'dataset': dataset,
                            'method': 'NHQ',
                            'build_time_s': row.get('build_time_s', 0),
                            'index_size_mb': row.get('index_size_mb', 0),
                        })

        # Post-filter HNSW
        path = BASE_DIR / f"faiss/results_postfilter/{dataset}/summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success']
            if not df.empty and 'build_time_s' in df.columns:
                if 'M' in df.columns:
                    df_unique = df.drop_duplicates(subset=['M'])
                else:
                    df_unique = df.head(1)
                for _, row in df_unique.iterrows():
                    if row.get('build_time_s', 0) > 0:
                        results.append({
                            'dataset': dataset,
                            'method': 'Post-filter',
                            'build_time_s': row.get('build_time_s', 0),
                            'index_size_mb': row.get('index_size_mb', 0),
                        })

        # SIEVE
        sieve_dir = BASE_DIR / "SIEVE/results"
        if sieve_dir.exists():
            # Find one CSV for this dataset with default config (M=32, budget=2.0, hist=0.25)
            sieve_files = list(sieve_dir.glob(f"sieve_{dataset}_*_M32_b2.0_h0.25.csv"))
            if sieve_files:
                df = pd.read_csv(sieve_files[0])
                if not df.empty:
                    # Take the first row (all rows have same build time/index size)
                    row = df.iloc[0]
                    results.append({
                        'dataset': dataset,
                        'method': 'SIEVE',
                        'build_time_s': row.get('build_time_s', 0),
                        'index_size_mb': row.get('index_size_mb', 0),
                    })

    return pd.DataFrame(results)


def plot_build_time_comparison(df):
    """Plot build time comparison across datasets and methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['Pre-filter', 'Post-filter', 'ACORN-1', 'ACORN-γ', 'UNG', 'CAPS', 'NHQ', 'SIEVE']
    methods = [m for m in methods if m in df['method'].unique()]

    x = np.arange(len(DATASETS))
    width = 0.12

    for i, method in enumerate(methods):
        method_data = []
        for dataset in DATASETS:
            subset = df[(df['dataset'] == dataset) & (df['method'] == method)]
            if not subset.empty:
                # Take mean if multiple configs
                method_data.append(subset['build_time_s'].mean())
            else:
                method_data.append(0)

        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, method_data, width, label=method, color=COLORS.get(method, 'gray'))

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Build Time (seconds)')
    ax.set_title('Index Build Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES[d] for d in DATASETS])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'build_time_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'build_time_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved build time comparison to {OUTPUT_DIR / 'build_time_comparison.png'}")
    plt.close()


def plot_index_size_comparison(df):
    """Plot index size comparison across datasets and methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['Pre-filter', 'Post-filter', 'ACORN-1', 'ACORN-γ', 'UNG', 'CAPS', 'NHQ', 'SIEVE']
    methods = [m for m in methods if m in df['method'].unique()]

    x = np.arange(len(DATASETS))
    width = 0.12

    for i, method in enumerate(methods):
        method_data = []
        for dataset in DATASETS:
            subset = df[(df['dataset'] == dataset) & (df['method'] == method)]
            if not subset.empty:
                method_data.append(subset['index_size_mb'].mean())
            else:
                method_data.append(0)

        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, method_data, width, label=method, color=COLORS.get(method, 'gray'))

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Index Size (MB)')
    ax.set_title('Index Size Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES[d] for d in DATASETS])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'index_size_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'index_size_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved index size comparison to {OUTPUT_DIR / 'index_size_comparison.png'}")
    plt.close()


def plot_build_time_vs_index_size(df):
    """Plot build time vs index size scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    methods = df['method'].unique()

    for method in methods:
        method_df = df[df['method'] == method]
        ax.scatter(
            method_df['index_size_mb'],
            method_df['build_time_s'],
            c=COLORS.get(method, 'gray'),
            label=method,
            s=80,
            alpha=0.7,
            marker='o'
        )

        # Add dataset labels
        for _, row in method_df.iterrows():
            ax.annotate(
                DATASET_NAMES[row['dataset']][:3],  # First 3 chars
                (row['index_size_mb'], row['build_time_s']),
                fontsize=6,
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points'
            )

    ax.set_xlabel('Index Size (MB)')
    ax.set_ylabel('Build Time (seconds)')
    ax.set_title('Build Time vs Index Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'build_time_vs_index_size.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'build_time_vs_index_size.png', dpi=150, bbox_inches='tight')
    print(f"Saved build time vs index size to {OUTPUT_DIR / 'build_time_vs_index_size.png'}")
    plt.close()


def print_summary_table(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Build Metrics Summary")
    print("="*80)

    # Group by method and compute stats
    summary = df.groupby('method').agg({
        'build_time_s': ['mean', 'min', 'max'],
        'index_size_mb': ['mean', 'min', 'max']
    }).round(2)

    print("\nBy Method (averaged across datasets):")
    print(summary)

    print("\n" + "-"*80)
    print("Detailed by Dataset and Method:")
    print("-"*80)

    pivot_time = df.pivot_table(
        values='build_time_s',
        index='dataset',
        columns='method',
        aggfunc='mean'
    ).round(2)
    print("\nBuild Time (seconds):")
    print(pivot_time)

    pivot_size = df.pivot_table(
        values='index_size_mb',
        index='dataset',
        columns='method',
        aggfunc='mean'
    ).round(2)
    print("\nIndex Size (MB):")
    print(pivot_size)


def main():
    print("Loading build metrics...")
    df = load_build_metrics()

    # Filter out zero values
    df = df[(df['build_time_s'] > 0) & (df['index_size_mb'] > 0)]

    print(f"Loaded {len(df)} records")
    print(f"Methods: {df['method'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")

    # Print summary
    print_summary_table(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_build_time_comparison(df)
    plot_index_size_comparison(df)
    plot_build_time_vs_index_size(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
