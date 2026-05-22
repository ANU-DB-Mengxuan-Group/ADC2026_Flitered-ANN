#!/usr/bin/env python3
"""
Generate Recall-QPS comparison plots for ACORN-1 vs ACORN-γ.
Compares across all 6 datasets and 3 scenarios (equal, or, and).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from load_results import load_acorn_1, load_acorn_gamma, get_pareto_front
import pandas as pd

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
})

COLORS = {
    'ACORN-1': '#1f77b4',
    'ACORN-γ': '#ff7f0e',
}

MARKERS = {
    'ACORN-1': 'o',
    'ACORN-γ': 's',
}

SCENARIOS = ['equal', 'or', 'and']
SCENARIO_NAMES = {
    'equal': 'Equality',
    'or': 'Overlap (OR)',
    'and': 'Containment (AND)',
}

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_acorn_data() -> pd.DataFrame:
    """Load all ACORN data."""
    all_dfs = []
    for dataset in DATASETS:
        for loader in [load_acorn_1, load_acorn_gamma]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def plot_single(df, dataset, scenario, ax):
    """Plot a single dataset-scenario combination."""
    df_filtered = df[(df['dataset'] == dataset) & (df['scenario'] == scenario)]

    for method in ['ACORN-1', 'ACORN-γ']:
        method_df = df_filtered[df_filtered['method'] == method]
        if method_df.empty:
            continue

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

    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def main():
    print("Loading ACORN data...")
    df = load_acorn_data()
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

    # Create big figure: 6 datasets x 3 scenarios
    fig, axes = plt.subplots(len(DATASETS), len(SCENARIOS), figsize=(15, 20))

    for i, dataset in enumerate(DATASETS):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]
            plot_single(df, dataset, scenario, ax)

            # Title for first row
            if i == 0:
                ax.set_title(SCENARIO_NAMES[scenario])

            # Y-label for first column
            if j == 0:
                ax.set_ylabel(f'{dataset}\nQPS')

            # Only show legend in first subplot
            if i == 0 and j == 0:
                ax.legend(loc='upper right')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'acorn_all_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")

    # Also create per-scenario summary (all datasets in one plot)
    for scenario in SCENARIOS:
        fig_sc, axes_sc = plt.subplots(2, 3, figsize=(15, 10))
        axes_flat = axes_sc.flatten()

        for idx, dataset in enumerate(DATASETS):
            ax = axes_flat[idx]
            plot_single(df, dataset, scenario, ax)
            ax.set_title(f'{dataset}')
            if idx == 0:
                ax.legend(loc='upper right')

        fig_sc.suptitle(f'ACORN-1 vs ACORN-γ: {SCENARIO_NAMES[scenario]}', fontsize=14)
        plt.tight_layout()
        sc_path = OUTPUT_DIR / f'acorn_{scenario}_comparison.png'
        plt.savefig(sc_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {sc_path}")
        plt.close(fig_sc)

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
