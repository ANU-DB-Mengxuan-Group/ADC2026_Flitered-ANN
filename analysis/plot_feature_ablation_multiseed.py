#!/usr/bin/env python3
"""Plot multi-seed feature ablation results, single-color with annotations."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

df = pd.read_csv(REPO / "analysis/ml_router/results/v2_5methods/feature_ablation_multiseed.csv")
main = df.groupby(["feature_set", "n_features"]).agg(
    mean=("recall", "mean"),
    std=("recall", "std"),
).reset_index().sort_values("n_features")

# Plot
fig, ax = plt.subplots(figsize=(7.0, 4.3))
COLOR = "#1f77b4"

# importance-ordered nested curve: minimal -> size4 -> size5 -> size8 ->
# size12 -> size18 -> full. core (hand-picked, n=6) is omitted because it
# uses a different feature-selection strategy.
imp = main[main["feature_set"] != "core"].sort_values("n_features")
ax.errorbar(imp["n_features"], imp["mean"], yerr=imp["std"],
            fmt="-o", color=COLOR, markersize=7, linewidth=1.6, capsize=4,
            zorder=3)

# "chosen" marker on minimal (n=3); placed below the point with a gap
chosen = main[main["feature_set"] == "minimal"].iloc[0]
ax.annotate("chosen", xy=(chosen["n_features"], chosen["mean"]),
            xytext=(8, -55), textcoords="offset points",
            fontsize=10.5, fontweight="bold", color=COLOR,
            arrowprops=dict(arrowstyle="->", color=COLOR, lw=1.4,
                            shrinkA=2, shrinkB=8))

ax.set_xlabel("Number of input features (n)")
ax.set_ylabel("Validation recall@10 (mean $\\pm$ std over 5 seeds)")
ax.set_title("Feature ablation")
ax.set_xticks([3, 4, 5, 8, 12, 18, 22])
ax.set_ylim(0.80, 1.005)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = REPO / "analysis/figures/ablation_features_multiseed.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out}")
