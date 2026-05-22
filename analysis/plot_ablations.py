#!/usr/bin/env python3
"""Plot ablation figures: recall vs #features (Fig 1) and recall vs #layers (Fig 2).

Reads analysis/ml_router/results/v2_5methods/ablation_metrics.csv produced by
aggregate_ablation_recall.py. Saves two PNG/PDF files to analysis/figures/.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "analysis/figures"
FIGS.mkdir(exist_ok=True)

df = pd.read_csv(REPO / "analysis/ml_router/results/v2_5methods/ablation_metrics.csv")
oracle = df["oracle_recall"].iloc[0]

# ── Figure 1: Feature ablation ──────────────────────────────────────
feat = df[df.ablation == "features"].sort_values("n_features")
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(feat["n_features"], feat["recall"], "-o", color="#2ca02c", lw=2.2, ms=10)
for _, r in feat.iterrows():
    ax.annotate(r["variant"], (r["n_features"], r["recall"]),
                textcoords="offset points", xytext=(8, -3), fontsize=9)
    ax.annotate(f"{r['recall']:.4f}", (r["n_features"], r["recall"]),
                textcoords="offset points", xytext=(8, 8), fontsize=8, color="#555")
ax.axhline(oracle, ls="--", color="gray", lw=1.2, zorder=0)
ax.text(feat["n_features"].max(), oracle + 0.002, f"Oracle = {oracle:.4f}",
        color="gray", fontsize=9, ha="right", va="bottom")
ax.set_xlabel("Number of features")
ax.set_ylabel("Recall@10 (validation set, mean of 15,000 queries)")
ax.set_title("Feature ablation: routing recall vs feature count")
ax.set_xticks(feat["n_features"])
ax.set_xticklabels([str(x) for x in feat["n_features"]])
ax.grid(True, alpha=0.3)
ax.set_ylim(min(feat["recall"].min(), 0.85) - 0.01, oracle + 0.01)
fig.tight_layout()
out1 = FIGS / "ablation_features.png"
fig.savefig(out1, dpi=140, bbox_inches="tight")
fig.savefig(str(out1).replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out1}")

# ── Figure 2: Layers ablation ───────────────────────────────────────
lay = df[df.ablation == "layers"].sort_values("n_layers")
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(lay["n_layers"], lay["recall"], "-o", color="#1f77b4", lw=2.2, ms=10)
for _, r in lay.iterrows():
    ax.annotate(f"{r['recall']:.4f}", (r["n_layers"], r["recall"]),
                textcoords="offset points", xytext=(8, 8), fontsize=8, color="#555")
ax.axhline(oracle, ls="--", color="gray", lw=1.2, zorder=0)
ax.text(lay["n_layers"].max(), oracle + 0.002, f"Oracle = {oracle:.4f}",
        color="gray", fontsize=9, ha="right", va="bottom")
ax.set_xlabel("Number of hidden layers")
ax.set_ylabel("Recall@10 (validation set, mean of 15,000 queries)")
ax.set_title("Layers ablation: routing recall vs MLP depth")
ax.set_xticks(lay["n_layers"])
ax.grid(True, alpha=0.3)
ax.set_ylim(min(lay["recall"].min(), 0.95) - 0.005, oracle + 0.01)
fig.tight_layout()
out2 = FIGS / "ablation_layers.png"
fig.savefig(out2, dpi=140, bbox_inches="tight")
fig.savefig(str(out2).replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out2}")
