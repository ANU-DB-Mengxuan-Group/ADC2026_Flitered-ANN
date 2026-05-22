#!/usr/bin/env python3
"""
Generate Recall-QPS comparison plots for ALL methods:
- Pre-filter (brute-force)
- Post-filter (HNSW)
- ACORN-1
- ACORN-γ
- UNG
- CAPS (equal only)
- NHQ (equal only, arxiv/yfcc only)

Key feature: Same scale across all subplots for fair comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from load_results import (
    load_acorn_1, load_acorn_gamma, load_prefilter, load_postfilter,
    load_ung, load_caps, load_nhq,
    load_filtered_vamana_original, load_stitched_vamana_original,
    load_sieve, get_pareto_front
)
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
})

COLORS = {
    'ACORN-1': '#1f77b4',           # blue
    'ACORN-γ': '#ff7f0e',           # orange
    'Pre-filter': '#2ca02c',        # green
    'Post-filter': '#d62728',       # red
    'UNG': '#9467bd',               # purple
    'FilteredVamana': '#8c564b',    # brown
    'StitchedVamana': '#e377c2',    # pink
    'CAPS': '#bcbd22',              # olive
    'NHQ': '#17becf',               # cyan
    'SIEVE': '#7f7f7f',             # gray
}

MARKERS = {
    'ACORN-1': 'o',
    'ACORN-γ': 's',
    'Pre-filter': '^',
    'Post-filter': 'v',
    'UNG': 'D',
    'FilteredVamana': 'P',
    'StitchedVamana': 'X',
    'CAPS': 'p',
    'NHQ': 'h',
    'SIEVE': '*',
}

SCENARIOS = ['and', 'or', 'equal']
SCENARIO_NAMES = {
    'and': 'Containment (AND)',
    'or': 'Overlap (OR)',
    'equal': 'Equality',
}

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
DATASET_NAMES = {
    'arxiv': 'arXiv',
    'yfcc': 'YFCC',
    'LAION1M': 'LAION-1M',
    'tripclick': 'TripClick',
    'ytb_audio': 'YTB-Audio',
    'ytb_video': 'YTB-Video',
}

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_data() -> pd.DataFrame:
    """Load all data from all methods."""
    all_dfs = []
    for dataset in DATASETS:
        for loader, method in [
            (load_acorn_1, 'ACORN-1'),
            (load_acorn_gamma, 'ACORN-γ'),
            (load_prefilter, 'Pre-filter'),
            (load_postfilter, 'Post-filter'),
            (load_ung, 'UNG'),
            (load_filtered_vamana_original, 'FilteredVamana'),
            (load_stitched_vamana_original, 'StitchedVamana'),
            (load_caps, 'CAPS'),
            (load_nhq, 'NHQ'),
            (load_sieve, 'SIEVE'),
        ]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def plot_single(df, dataset, scenario, ax, xlim=None, ylim=None):
    """Plot a single dataset-scenario combination."""
    df_filtered = df[(df['dataset'] == dataset) & (df['scenario'] == scenario)]

    methods_order = ['Pre-filter', 'Post-filter', 'ACORN-1', 'ACORN-γ', 'UNG',
                     'FilteredVamana', 'StitchedVamana', 'CAPS', 'NHQ', 'SIEVE']

    for method in methods_order:
        method_df = df_filtered[df_filtered['method'] == method]
        if method_df.empty:
            continue

        # For Pre-filter, there's only one point per scenario
        if method == 'Pre-filter':
            ax.scatter(
                method_df['recall'],
                method_df['QPS'],
                c=COLORS[method],
                marker=MARKERS[method],
                label=method,
                s=60,
                alpha=0.9,
                zorder=10,
            )
        else:
            # Get Pareto front for other methods
            pareto = get_pareto_front(method_df)
            pareto = pareto.sort_values('recall')

            ax.scatter(
                pareto['recall'],
                pareto['QPS'],
                c=COLORS[method],
                marker=MARKERS[method],
                label=method,
                s=40,
                alpha=0.8,
            )

            if len(pareto) > 1:
                ax.plot(
                    pareto['recall'],
                    pareto['QPS'],
                    c=COLORS[method],
                    alpha=0.4,
                    linestyle='--',
                )

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Apply fixed limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def compute_global_limits(df, scenarios):
    """Compute global x and y limits across all datasets and scenarios."""
    x_min, x_max = 1.0, 0.0
    y_min, y_max = float('inf'), 0.0

    for dataset in DATASETS:
        for scenario in scenarios:
            df_filtered = df[(df['dataset'] == dataset) & (df['scenario'] == scenario)]
            if df_filtered.empty:
                continue

            recalls = df_filtered['recall'].dropna()
            qps_vals = df_filtered['QPS'].dropna()
            # Filter out zero QPS values for log scale
            qps_vals = qps_vals[qps_vals > 0]

            if len(recalls) > 0:
                x_min = min(x_min, recalls.min())
                x_max = max(x_max, recalls.max())
            if len(qps_vals) > 0:
                y_min = min(y_min, qps_vals.min())
                y_max = max(y_max, qps_vals.max())

    # Add some padding
    x_padding = (x_max - x_min) * 0.05
    x_min = max(0, x_min - x_padding)
    x_max = min(1.05, x_max + x_padding)

    # For log scale, use multiplicative padding
    # Ensure y_min is positive for log scale
    if y_min <= 0 or y_min == float('inf'):
        y_min = 1
    y_min = y_min * 0.5
    y_max = y_max * 2

    return (x_min, x_max), (y_min, y_max)


def main():
    print("Loading all data...")
    df = load_all_data()
    print(f"Loaded {len(df)} rows")

    # Summary
    print("\nData summary:")
    for dataset in DATASETS:
        df_ds = df[df['dataset'] == dataset]
        if df_ds.empty:
            print(f"  {dataset}: NO DATA")
            continue
        methods = df_ds['method'].unique()
        scenarios = df_ds['scenario'].unique()
        print(f"  {dataset}: methods={list(methods)}, scenarios={list(scenarios)}")

    # Compute global limits for uniform scale
    xlim, ylim = compute_global_limits(df, SCENARIOS)
    print(f"\nGlobal limits: x={xlim}, y={ylim}")

    # ===== Figure 1: All datasets × all scenarios (6×3 grid) =====
    print("\nGenerating main comparison figure...")
    fig, axes = plt.subplots(len(DATASETS), len(SCENARIOS), figsize=(14, 18))

    for i, dataset in enumerate(DATASETS):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]
            plot_single(df, dataset, scenario, ax, xlim=xlim, ylim=ylim)

            # Title for first row
            if i == 0:
                ax.set_title(SCENARIO_NAMES[scenario], fontweight='bold')

            # Y-label for first column
            if j == 0:
                ax.set_ylabel(f'{DATASET_NAMES[dataset]}\nQPS')
            else:
                ax.set_ylabel('')

            # X-label for last row
            if i == len(DATASETS) - 1:
                ax.set_xlabel('Recall@10')

    # Create custom legend with all methods
    from matplotlib.lines import Line2D
    all_methods = ['Pre-filter', 'Post-filter', 'ACORN-1', 'ACORN-γ', 'UNG',
                   'FilteredVamana', 'StitchedVamana', 'CAPS', 'NHQ', 'SIEVE']
    legend_elements = [Line2D([0], [0], marker=MARKERS[m], color='w', markerfacecolor=COLORS[m],
                              markersize=8, label=m) for m in all_methods]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = OUTPUT_DIR / 'all_methods_comparison.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'all_methods_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close(fig)

    # ===== Figure 2: Per-scenario figures (2×3 grid per scenario) =====
    for scenario in SCENARIOS:
        print(f"\nGenerating {scenario} figure...")

        # Compute limits for this scenario only
        xlim_sc, ylim_sc = compute_global_limits(df, [scenario])

        fig_sc, axes_sc = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat = axes_sc.flatten()

        for idx, dataset in enumerate(DATASETS):
            ax = axes_flat[idx]
            plot_single(df, dataset, scenario, ax, xlim=xlim_sc, ylim=ylim_sc)
            ax.set_title(DATASET_NAMES[dataset], fontweight='bold')
            ax.set_xlabel('Recall@10')
            if idx % 3 == 0:
                ax.set_ylabel('QPS')
            if idx == 0:
                ax.legend(loc='lower right', framealpha=0.9)

        fig_sc.suptitle(f'All Methods: {SCENARIO_NAMES[scenario]}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        sc_path = OUTPUT_DIR / f'all_methods_{scenario}.pdf'
        plt.savefig(sc_path, dpi=150, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / f'all_methods_{scenario}.png', dpi=150, bbox_inches='tight')
        print(f"Saved to {sc_path}")
        plt.close(fig_sc)

    # ===== Figure 3: Baselines only (Pre-filter vs Post-filter) =====
    # Layout: 2×3 grid (6 datasets), but legend organized by scenario rows
    print("\nGenerating baselines comparison figure...")
    df_baselines = df[df['method'].isin(['Pre-filter', 'Post-filter'])]

    xlim_bl, ylim_bl = compute_global_limits(df_baselines, SCENARIOS)

    # Define distinct colors for each scenario
    SCENARIO_COLORS = {
        'equal': '#1f77b4',    # blue
        'and': '#2ca02c',      # green
        'or': '#d62728',       # red
    }
    # Different markers for Pre-filter vs Post-filter
    METHOD_MARKERS_BL = {
        'Pre-filter': '^',     # triangle up
        'Post-filter': 'v',    # triangle down
    }

    fig_bl, axes_bl = plt.subplots(2, 3, figsize=(14, 8))

    for idx, dataset in enumerate(DATASETS):
        ax = axes_bl[idx // 3, idx % 3]

        for scenario in SCENARIOS:
            df_filtered = df_baselines[
                (df_baselines['dataset'] == dataset) &
                (df_baselines['scenario'] == scenario)
            ]

            for method in ['Pre-filter', 'Post-filter']:
                method_df = df_filtered[df_filtered['method'] == method]
                if method_df.empty:
                    continue

                if method == 'Pre-filter':
                    marker_style = {'marker': METHOD_MARKERS_BL[method], 's': 100}
                else:
                    pareto = get_pareto_front(method_df)
                    method_df = pareto.sort_values('recall')
                    marker_style = {'marker': METHOD_MARKERS_BL[method], 's': 60}

                ax.scatter(
                    method_df['recall'],
                    method_df['QPS'],
                    c=SCENARIO_COLORS[scenario],
                    alpha=0.8,
                    edgecolors='white' if method == 'Pre-filter' else 'none',
                    linewidths=1,
                    **marker_style,
                )

        ax.set_title(DATASET_NAMES[dataset], fontweight='bold')
        ax.set_xlabel('Recall@10')
        ax.set_ylabel('QPS' if idx % 3 == 0 else '')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim_bl)
        ax.set_ylim(ylim_bl)

    # Create custom legend: 3 rows × 2 cols
    # Left col = all Pre-filter, Right col = all Post-filter, each row = same color/scenario
    # matplotlib legend with ncol=2 fills column-major:
    #   col1: [0,1,2], col2: [3,4,5]
    #   displays as: row1=[0,3], row2=[1,4], row3=[2,5]
    # So we need: [0]=Pre-equal, [1]=Pre-and, [2]=Pre-or, [3]=Post-equal, [4]=Post-and, [5]=Post-or
    from matplotlib.lines import Line2D
    legend_elements = [
        # Column 1: Pre-filters
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#1f77b4', markersize=10,
               markeredgecolor='white', markeredgewidth=1, label='Pre-filter (equal)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c', markersize=10,
               markeredgecolor='white', markeredgewidth=1, label='Pre-filter (and)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#d62728', markersize=10,
               markeredgecolor='white', markeredgewidth=1, label='Pre-filter (or)'),
        # Column 2: Post-filters
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#1f77b4', markersize=8, label='Post-filter (equal)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#2ca02c', markersize=8, label='Post-filter (and)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#d62728', markersize=8, label='Post-filter (or)'),
    ]

    fig_bl.legend(handles=legend_elements, loc='upper center', ncol=2, framealpha=0.9,
                  bbox_to_anchor=(0.5, 1.02), fontsize=9)

    fig_bl.suptitle('Baseline Comparison: Pre-filter vs Post-filter', fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    bl_path = OUTPUT_DIR / 'baselines_comparison.pdf'
    plt.savefig(bl_path, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'baselines_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved to {bl_path}")
    plt.close(fig_bl)

    print("\nDone!")


if __name__ == "__main__":
    main()
