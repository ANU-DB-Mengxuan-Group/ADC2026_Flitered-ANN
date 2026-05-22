#!/usr/bin/env python3
"""
Plot Recall@10 vs QPS Pareto curves for Fixed EQ and Stitched DiskANN.

Generates a 2x3 grid of subplots, one per dataset.
Each subplot shows the Pareto frontier for both algorithms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

DATASETS = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']
DATASET_DISPLAY = {
    'arxiv': 'Arxiv (133K, 768d)',
    'yfcc': 'YFCC (1M, 192d)',
    'LAION1M': 'LAION (1M, 512d)',
    'tripclick': 'TripClick (1M, 768d)',
    'ytb_audio': 'YTB-Audio (5M, 128d)',
    'ytb_video': 'YTB-Video (1M, 1024d)',
}

ALGO_STYLE = {
    'fixed_eq': {
        'color': '#2166ac',
        'marker': 'o',
        'label': 'FilteredVamana',
    },
    'stitched': {
        'color': '#b2182b',
        'marker': 's',
        'label': 'StitchedVamana',
    },
}

BASE_DIR = Path(__file__).parent.parent
FIXED_EQ_DIR = BASE_DIR / 'data_fixed_eq' / 'results'
STITCHED_DIR = BASE_DIR / 'data_stitched_eq' / 'results'
OUTPUT_DIR = Path(__file__).parent / 'figures'


def load_results(result_dir, dataset):
    csv_path = result_dir / dataset / 'summary.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    return df


def pareto_frontier(df, recall_col='recall@10', qps_col='qps'):
    """Extract Pareto-optimal points (maximize both recall and QPS).

    Walk from highest recall down; keep a point if its QPS is >= all
    points with higher recall (i.e., the standard upper-right frontier).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # For each unique recall, keep max QPS
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]

        df_fixed = load_results(FIXED_EQ_DIR, dataset)
        df_stitched = load_results(STITCHED_DIR, dataset)

        # Scatter all points (faded)
        for algo_key, result_dir, df in [
            ('fixed_eq', FIXED_EQ_DIR, df_fixed),
            ('stitched', STITCHED_DIR, df_stitched),
        ]:
            if df is not None and len(df) > 0:
                style = ALGO_STYLE[algo_key]
                ax.scatter(df['recall@10'], df['qps'],
                          alpha=0.15, s=12, color=style['color'],
                          marker=style['marker'], zorder=1)

        # Pareto frontiers
        pf_fixed = pareto_frontier(df_fixed)
        pf_stitched = pareto_frontier(df_stitched)

        for algo_key, pf in [('fixed_eq', pf_fixed), ('stitched', pf_stitched)]:
            if len(pf) > 0:
                style = ALGO_STYLE[algo_key]
                ax.plot(pf['recall@10'], pf['qps'],
                       marker=style['marker'], color=style['color'],
                       markersize=4.5, linewidth=1.5,
                       label=style['label'], zorder=3)

        ax.set_title(DATASET_DISPLAY[dataset], fontweight='bold')
        ax.set_xlabel('Recall@10')
        ax.set_ylabel('QPS')
        ax.legend(loc='upper right')

        # Auto x-range: start from 5% below minimum recall
        all_dfs = [d for d in [df_fixed, df_stitched] if d is not None and len(d) > 0]
        if all_dfs:
            all_recall = pd.concat([d['recall@10'] for d in all_dfs])
            min_r = all_recall.min()
            # If min recall > 0.5, zoom in; otherwise show full range
            if min_r > 0.5:
                ax.set_xlim(max(0, min_r - 0.05), 1.02)
            else:
                ax.set_xlim(max(0, min_r - 0.05), 1.02)

    fig.suptitle('Recall@10 vs QPS: FilteredVamana vs StitchedVamana\n(Fixed-Length Equality Filter)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        outpath = OUTPUT_DIR / f'recall_qps_pareto.{fmt}'
        fig.savefig(outpath, bbox_inches='tight', dpi=300 if fmt == 'pdf' else 200)
    print(f"Saved figures to {OUTPUT_DIR}/")
    plt.close()

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("Summary: Best Recall@10 achieved per dataset")
    print("=" * 80)
    print(f"{'Dataset':<16} {'Algorithm':<18} {'Best R@10':>10} {'QPS@best':>10} {'Build(s)':>10} {'Index(MB)':>10}")
    print("-" * 80)
    for dataset in DATASETS:
        for algo, result_dir, name in [
            ('fixed_eq', FIXED_EQ_DIR, 'FilteredVamana'),
            ('stitched', STITCHED_DIR, 'StitchedVamana'),
        ]:
            df = load_results(result_dir, dataset)
            if df is not None and len(df) > 0:
                best = df.loc[df['recall@10'].idxmax()]
                print(f"{DATASET_DISPLAY[dataset].split('(')[0].strip():<16} "
                      f"{name:<18} "
                      f"{best['recall@10']:>10.4f} "
                      f"{best['qps']:>10.1f} "
                      f"{best['build_time_s']:>10.1f} "
                      f"{best['index_size_mb']:>10.1f}")
        print()

    # ---- Pareto comparison at fixed recall thresholds ----
    print("=" * 80)
    print("QPS comparison at recall thresholds")
    print("=" * 80)
    thresholds = [0.8, 0.9, 0.95, 0.99]
    print(f"{'Dataset':<16}", end="")
    for t in thresholds:
        print(f"  {'QPS@R≥' + str(t):>24}", end="")
    print()
    print(f"{'':16}", end="")
    for _ in thresholds:
        print(f"  {'Filtered':>11} {'Stitched':>12}", end="")
    print()
    print("-" * 120)

    for dataset in DATASETS:
        df_f = load_results(FIXED_EQ_DIR, dataset)
        df_s = load_results(STITCHED_DIR, dataset)
        print(f"{DATASET_DISPLAY[dataset].split('(')[0].strip():<16}", end="")
        for t in thresholds:
            for df in [df_f, df_s]:
                if df is not None:
                    above = df[df['recall@10'] >= t]
                    if len(above) > 0:
                        print(f"  {above['qps'].max():>11.0f}", end="")
                    else:
                        print(f"  {'---':>11}", end="")
                else:
                    print(f"  {'N/A':>11}", end="")
        print()


if __name__ == '__main__':
    main()
