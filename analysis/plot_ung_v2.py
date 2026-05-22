#!/usr/bin/env python3
"""Plot UNG V2 Recall-QPS results - one figure per dataset"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
})

# UNG V2 Results
DATASETS = ['synth_192d', 'synth_512d', 'synth_768d_hc', 'yahoo800k', 'dbpedia560k']
DATASET_NAMES = {
    'synth_192d': 'Synth-192d',
    'synth_512d': 'Synth-512d',
    'synth_768d_hc': 'Synth-768d-HC',
    'yahoo800k': 'Yahoo-800K',
    'dbpedia560k': 'DBpedia-560K',
}

SCENARIOS = ['and', 'or', 'equal']
SCENARIO_NAMES = {
    'and': 'Containment (AND)',
    'or': 'Overlap (OR)',
    'equal': 'Equality',
}

# Data: recall (0-1 scale), QPS
data = {
    'synth_192d': {
        'and': (0.8781, 14492.8),
        'or': (0.5558, 1562.5),
        'equal': (1.0, 33333.3),
    },
    'synth_512d': {
        'and': (0.9971, 2123.14),
        'or': (0.8798, 291.886),
        'equal': (1.0, 3267.97),
    },
    'synth_768d_hc': {
        'and': (0.9484, 16129),
        'or': (0.7462, 2123.14),
        'equal': (1.0, 10526.3),
    },
    'yahoo800k': {
        'and': (0.6207, 808.407),
        'or': (0.9812, 25.937),
        'equal': (0.9854, 1076.43),
    },
    'dbpedia560k': {
        'and': (0.7816, 697.837),
        'or': (0.9625, 55.6824),
        'equal': (0.9997, 791.766),
    },
}

COLORS = {
    'UNG': '#9467bd',      # purple (consistent with plot_all_methods.py)
}

MARKERS = {
    'UNG': 'D',            # diamond
}

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_per_dataset():
    """Plot one figure per dataset, 3 subplots (scenarios) per figure."""
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'UNG on {DATASET_NAMES[dataset]}', fontsize=14)

        for idx, scenario in enumerate(SCENARIOS):
            ax = axes[idx]
            recall, qps = data[dataset][scenario]

            # Plot single point
            ax.scatter(
                recall, qps,
                c=COLORS['UNG'],
                marker=MARKERS['UNG'],
                s=150,
                label='UNG',
                alpha=0.9,
                edgecolors='black',
                linewidths=0.5,
                zorder=10,
            )

            # Add text annotation
            ax.annotate(
                f'{recall*100:.1f}%\n{qps:.0f} QPS',
                (recall, qps),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

            ax.set_title(SCENARIO_NAMES[scenario], fontsize=11)
            ax.set_xlabel('Recall@10')
            ax.set_ylabel('QPS' if idx == 0 else '')
            ax.set_yscale('log')
            ax.set_xlim(0.3, 1.1)
            ax.set_ylim(10, 100000)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f'ung_v2_{dataset}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / f'ung_v2_{dataset}.pdf', bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


def plot_combined_grid():
    """Plot all datasets in a 5x3 grid (dataset rows x scenario cols)."""
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))

    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, scenario in enumerate(SCENARIOS):
            ax = axes[row_idx, col_idx]
            recall, qps = data[dataset][scenario]

            ax.scatter(
                recall, qps,
                c=COLORS['UNG'],
                marker=MARKERS['UNG'],
                s=120,
                alpha=0.9,
                edgecolors='black',
                linewidths=0.5,
                zorder=10,
            )

            # Annotation
            ax.annotate(
                f'{recall*100:.1f}%',
                (recall, qps),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
            )

            ax.set_yscale('log')
            ax.set_xlim(0.3, 1.1)
            ax.set_ylim(10, 100000)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.4)

            # Row labels (dataset names)
            if col_idx == 0:
                ax.set_ylabel(f'{DATASET_NAMES[dataset]}\nQPS', fontsize=10)
            else:
                ax.set_ylabel('')

            # Column labels (scenario names)
            if row_idx == 0:
                ax.set_title(SCENARIO_NAMES[scenario], fontsize=11)

            # X-axis label only on bottom row
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel('Recall@10')

    plt.suptitle('UNG Performance on V2 Validation Datasets', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ung_v2_combined_grid.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ung_v2_combined_grid.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/ung_v2_combined_grid.png/pdf")


def print_summary():
    """Print summary table."""
    print("\n=== UNG V2 Results Summary ===")
    print(f"{'Dataset':<18} {'AND':<22} {'OR':<22} {'Equal':<22}")
    print("-" * 84)
    for ds in DATASETS:
        and_r, and_q = data[ds]['and']
        or_r, or_q = data[ds]['or']
        eq_r, eq_q = data[ds]['equal']
        print(f"{DATASET_NAMES[ds]:<18} "
              f"{and_r*100:>5.1f}% @ {and_q:>8.0f} QPS  "
              f"{or_r*100:>5.1f}% @ {or_q:>8.0f} QPS  "
              f"{eq_r*100:>5.1f}% @ {eq_q:>8.0f} QPS")

    print("\n=== Key Observations ===")
    print("- Equality: UNG achieves ~100% recall on all 5 datasets ✓")
    print("- AND: High on synthetic (88-100%), lower on real (62-78%)")
    print("- OR: Low on synthetic (56-88%), high on real (96-98%)")


if __name__ == '__main__':
    plot_per_dataset()
    plot_combined_grid()
    print_summary()
