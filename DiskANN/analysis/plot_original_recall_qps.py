#!/usr/bin/env python3
"""
Plot Recall@10 vs QPS for original-label experiments (3 scenarios × 6 datasets).
Layout: 3 rows (containment/equality/overlap) × 6 columns (datasets).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
DATASET_DISPLAY = {
    'arxiv': 'Arxiv\n(133K, 768d)',
    'yfcc': 'YFCC\n(1M, 192d)',
    'LAION1M': 'LAION\n(1M, 512d)',
    'tripclick': 'TripClick\n(1M, 768d)',
    'ytb_audio': 'YTB-Audio\n(5M, 128d)',
    'ytb_video': 'YTB-Video\n(1M, 1024d)',
}

SCENARIOS = ['containment', 'overlap', 'equality']
SCENARIO_DISPLAY = {'containment': 'Containment (AND)', 'overlap': 'Overlap (OR)', 'equality': 'Equality (=)'}

ALGO_STYLE = {
    'filtered': {'color': '#2166ac', 'marker': 'o', 'label': 'FilteredVamana'},
    'stitched': {'color': '#b2182b', 'marker': 's', 'label': 'StitchedVamana'},
}

BASE_DIR = Path(__file__).parent.parent
FILTERED_DIR = BASE_DIR / 'data_original' / 'results'
STITCHED_DIR = BASE_DIR / 'data_stitched_original' / 'results'
OUTPUT_DIR = Path(__file__).parent / 'figures'


def load_results(result_dir, dataset, scenario):
    csv_path = result_dir / dataset / 'summary.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[(df['status'] == 'success') & (df['scenario'] == scenario)]
    return df if len(df) > 0 else None


def pareto_frontier(df, recall_col='recall@10', qps_col='qps'):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    best = df.groupby(recall_col)[qps_col].max().reset_index()
    best = best.sort_values(recall_col, ascending=False)
    pareto = []
    max_qps = -1
    for _, row in best.iterrows():
        if row[qps_col] >= max_qps:
            pareto.append(row)
            max_qps = row[qps_col]
    return pd.DataFrame(pareto).sort_values(recall_col)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    for row, scenario in enumerate(SCENARIOS):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]

            df_f = load_results(FILTERED_DIR, dataset, scenario)
            df_s = load_results(STITCHED_DIR, dataset, scenario)

            # Scatter all points
            for key, df in [('filtered', df_f), ('stitched', df_s)]:
                if df is not None and len(df) > 0:
                    style = ALGO_STYLE[key]
                    ax.scatter(df['recall@10'] * 100, df['qps'],
                              alpha=0.2, s=10, color=style['color'],
                              marker=style['marker'], zorder=1)

            # Pareto frontiers
            for key, df in [('filtered', df_f), ('stitched', df_s)]:
                pf = pareto_frontier(df)
                if len(pf) > 0:
                    style = ALGO_STYLE[key]
                    ax.plot(pf['recall@10'] * 100, pf['qps'],
                           marker=style['marker'], color=style['color'],
                           markersize=4, linewidth=1.3,
                           label=style['label'], zorder=3)

            # Y-axis log scale
            ax.set_yscale('log')

            # Labels
            if row == 2:
                ax.set_xlabel('Recall@10 (%)')
            if col == 0:
                ax.set_ylabel(f'{SCENARIO_DISPLAY[scenario]}\nQPS')
            if row == 0:
                ax.set_title(DATASET_DISPLAY[dataset], fontweight='bold')

            # Legend only on first subplot
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=7)

            # X range
            all_dfs = [d for d in [df_f, df_s] if d is not None]
            if all_dfs:
                all_recall = pd.concat([d['recall@10'] for d in all_dfs]) * 100
                ax.set_xlim(max(0, all_recall.min() - 5), 105)

    fig.suptitle('Recall@10 vs QPS: Original Labels (3 Scenarios × 6 Datasets)', fontsize=13, y=1.01)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'original_recall_qps.{fmt}', bbox_inches='tight',
                    dpi=300 if fmt == 'pdf' else 200)
    print(f"Saved to {OUTPUT_DIR}/original_recall_qps.{{pdf,png}}")
    plt.close()

    # Summary table
    print("\n" + "=" * 100)
    print("Best Recall@10 per dataset × scenario (original labels)")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Scenario':<14} {'FilteredV R@10':>15} {'QPS':>8} {'StitchedV R@10':>15} {'QPS':>8}")
    print("-" * 100)
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            df_f = load_results(FILTERED_DIR, dataset, scenario)
            df_s = load_results(STITCHED_DIR, dataset, scenario)
            line = f"{dataset:<12} {scenario:<14}"
            for df in [df_f, df_s]:
                if df is not None and len(df) > 0 and df['recall@10'].max() > 0:
                    best = df.loc[df['recall@10'].idxmax()]
                    line += f" {best['recall@10']:>14.4f} {best['qps']:>8.0f}"
                else:
                    line += f" {'—':>14} {'—':>8}"
            print(line)
        print()


if __name__ == '__main__':
    main()
