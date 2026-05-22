#!/usr/bin/env python3
"""
Generate Recall-QPS comparison plots for Fixed-Length Equality scenario.
Compares all methods on arxiv and yfcc datasets.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from load_results import load_fixed_eq_results, get_pareto_front

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (10, 5),
})

# All methods with distinct colors and markers
COLORS = {
    'ACORN-1': '#1f77b4',       # blue
    'ACORN-γ': '#ff7f0e',       # orange
    'CAPS': '#2ca02c',           # green
    'NHQ': '#d62728',            # red
    'UNG': '#9467bd',            # purple
    'FilteredVamana': '#8c564b', # brown
    'StitchedVamana': '#e377c2', # pink
    'Pre-filter': '#7f7f7f',     # gray
    'Post-filter': '#bcbd22',    # olive
    'SIEVE': '#17becf',          # cyan
}

MARKERS = {
    'ACORN-1': 'o',
    'ACORN-γ': 's',
    'CAPS': '^',
    'NHQ': 'D',
    'UNG': 'v',
    'FilteredVamana': 'P',
    'StitchedVamana': 'X',
    'Pre-filter': '*',
    'Post-filter': 'h',
    'SIEVE': 'p',
}

# Display order for legend
METHOD_ORDER = [
    'ACORN-γ', 'ACORN-1', 'UNG', 'FilteredVamana', 'StitchedVamana',
    'NHQ', 'CAPS', 'SIEVE', 'Pre-filter', 'Post-filter',
]

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_recall_qps(df, dataset: str, ax, show_all_points=False):
    """Plot Recall vs QPS for a dataset on given axes."""
    df_dataset = df[df['dataset'] == dataset].copy()

    # Plot methods in defined order, skip those with no data
    for method in METHOD_ORDER:
        method_df = df_dataset[df_dataset['method'] == method]
        if method_df.empty:
            continue

        # Get Pareto front for cleaner visualization
        if show_all_points:
            plot_df = method_df
            alpha = 0.3
        else:
            plot_df = get_pareto_front(method_df)
            alpha = 0.8

        # Sort by recall for line plot
        plot_df = plot_df.sort_values('recall')

        ax.scatter(
            plot_df['recall'],
            plot_df['QPS'],
            c=COLORS[method],
            marker=MARKERS[method],
            label=method,
            alpha=alpha,
            s=50,
            zorder=3,
        )

        # Connect points with line for Pareto front
        if not show_all_points and len(plot_df) > 1:
            ax.plot(
                plot_df['recall'],
                plot_df['QPS'],
                c=COLORS[method],
                alpha=0.5,
                linestyle='--',
                zorder=2,
            )

    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    ax.set_title(f'{dataset.upper()} - Fixed-Length Equality')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='best', ncol=1, fontsize=9)


def main():
    print("Loading Fixed-EQ results...")
    df = load_fixed_eq_results()
    print(f"Loaded {len(df)} rows")

    # Print summary
    print("\nData summary:")
    for dataset in ['arxiv', 'yfcc']:
        print(f"\n{dataset.upper()}:")
        df_ds = df[df['dataset'] == dataset]
        for method in METHOD_ORDER:
            method_df = df_ds[df_ds['method'] == method]
            if method_df.empty:
                continue
            recall_range = f"{method_df['recall'].min():.3f} - {method_df['recall'].max():.3f}"
            qps_range = f"{method_df['QPS'].min():.0f} - {method_df['QPS'].max():.0f}"
            print(f"  {method}: {len(method_df)} points, recall={recall_range}, QPS={qps_range}")

    # Create figure with 2 subplots (one per dataset)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_recall_qps(df, 'arxiv', axes[0])
    plot_recall_qps(df, 'yfcc', axes[1])

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fixed_eq_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")

    # Also save individual plots
    for dataset in ['arxiv', 'yfcc']:
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        plot_recall_qps(df, dataset, ax_single)
        plt.tight_layout()
        single_path = OUTPUT_DIR / f'fixed_eq_{dataset}.png'
        plt.savefig(single_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {single_path}")
        plt.close(fig_single)

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
