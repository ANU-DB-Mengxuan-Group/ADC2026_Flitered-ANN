"""
Intro Figure 1: YFCC OR vs YFCC Equality, showing UNG's dramatic reversal across scenarios.

1 dataset (V1 yfcc) x 2 scenarios (OR, Equality) = 1x2 grid.
Methods plotted: UNG, Post-filter, SIEVE, ACORN-gamma, FilteredVamana.
Pre-filter dropped because its perfect recall (~1.0) saturates and distracts.

Output: analysis/figures/intro_per_scenario.{pdf,png}
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_results import (
    load_prefilter, load_postfilter, load_acorn_1, load_acorn_gamma,
    load_ung, load_filtered_vamana_original, load_stitched_vamana_original,
    load_sieve, load_caps, load_nhq,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent

DATASET = "yfcc"
SCENARIOS = ["or", "equal"]
SC_LABEL = {"or": "OR (Overlap)", "equal": "Equality"}

METHODS = [
    ("Pre-filter",     load_prefilter,                "#2ca02c", "^"),
    ("Post-filter",    load_postfilter,               "#d62728", "v"),
    ("ACORN-1",        load_acorn_1,                  "#1f77b4", "o"),
    ("ACORN-gamma",    load_acorn_gamma,              "#ff7f0e", "s"),
    ("UNG",            load_ung,                      "#9467bd", "D"),
    ("FilteredVamana", load_filtered_vamana_original, "#8c564b", "+"),
    ("StitchedVamana", load_stitched_vamana_original, "#e377c2", "x"),
    ("CAPS",           load_caps,                     "#bcbd22", "p"),
    ("NHQ",            load_nhq,                      "#17becf", "h"),
    ("SIEVE",          load_sieve,                    "#7f7f7f", "*"),
]


def pareto_mask(recalls, qpses):
    """Return bool array: True for points on the upper-right Pareto frontier
    (higher recall AND higher QPS = better). A point is dominated iff some
    other point has BOTH recall>=this AND qps>=this AND is strictly better
    in at least one of them."""
    n = len(recalls)
    on_front = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (recalls[j] >= recalls[i] and qpses[j] >= qpses[i] and
                    (recalls[j] > recalls[i] or qpses[j] > qpses[i])):
                on_front[i] = False
                break
    return on_front


def plot_one(ax, sc, method_data):
    for name, color, marker, df in method_data:
        if df.empty:
            continue
        recalls = df["recall"].values
        qpses = df["QPS"].values
        on_front = pareto_mask(recalls, qpses)

        # Dominated points: faded, smaller, no line
        dom_idx = [i for i, p in enumerate(on_front) if not p]
        if dom_idx:
            ax.scatter([recalls[i] for i in dom_idx],
                       [qpses[i] for i in dom_idx],
                       marker=marker, color=color, s=18, alpha=0.30,
                       linewidths=0, zorder=1)

        # Pareto front: full color + line
        front_idx = [i for i, p in enumerate(on_front) if p]
        front_r = [recalls[i] for i in front_idx]
        front_q = [qpses[i] for i in front_idx]
        order = sorted(range(len(front_r)), key=lambda k: front_r[k])
        front_r_sorted = [front_r[k] for k in order]
        front_q_sorted = [front_q[k] for k in order]
        ax.plot(front_r_sorted, front_q_sorted,
                marker=marker, color=color, markersize=5,
                linewidth=1.4, alpha=0.95, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"YFCC / {SC_LABEL[sc]}", fontsize=11)
    ax.set_xlabel("Recall@10", fontsize=10)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)

    cached = {}
    for j, sc in enumerate(SCENARIOS):
        method_data = []
        for name, loader, color, marker in METHODS:
            if name not in cached:
                cached[name] = loader(DATASET)
            df = cached[name]
            df_sc = df[df["scenario"] == sc] if not df.empty else df
            method_data.append((name, color, marker, df_sc))
        plot_one(axes[j], sc, method_data)
        if j == 0:
            axes[j].set_ylabel("QPS (log)", fontsize=10)

    # ---- Annotations highlighting UNG vs SIEVE reversal ----
    UNG_CLR = "#9467bd"
    SIEVE_CLR = "#7f7f7f"
    # shrinkB makes arrow tip stop short of xy (visible gap, not inside data points)
    ARROW = lambda c: dict(arrowstyle="->", color=c, lw=1.2, shrinkB=14)

    # OR cell: UNG fails (bottom), SIEVE leads (top, top-right cluster)
    axes[0].annotate(
        "UNG drops\n(recall < 0.6)",
        xy=(0.52, 50), xytext=(0.05, 600),
        fontsize=10, color=UNG_CLR, weight="bold",
        arrowprops=ARROW(UNG_CLR),
    )
    axes[0].annotate(
        "SIEVE leads",
        xy=(0.85, 15000), xytext=(0.40, 2000),
        fontsize=10, color=SIEVE_CLR, weight="bold",
        arrowprops=ARROW(SIEVE_CLR),
    )

    # Equality cell: UNG dominates (top-right), SIEVE moderate (recall ~0.5)
    axes[1].annotate(
        "UNG dominates",
        xy=(0.97, 50000), xytext=(0.72, 200000),
        fontsize=10, color=UNG_CLR, weight="bold",
        arrowprops=ARROW(UNG_CLR),
    )
    axes[1].annotate(
        "SIEVE stuck\nat recall ~0.5",
        xy=(0.45, 15000), xytext=(0.65, 200),
        fontsize=10, color=SIEVE_CLR, weight="bold", verticalalignment="bottom",
        arrowprops=ARROW(SIEVE_CLR),
    )

    # Collect legend from both axes (CAPS/NHQ only show in Equality)
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="upper center",
               ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_dir = BASE_DIR / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "intro_per_scenario.pdf"
    png_path = out_dir / "intro_per_scenario.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
