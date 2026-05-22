#!/usr/bin/env python3
"""3-cell representative main-result figure for §6.3 (top banner).

Selected cells:
  - dbpedia560k / or    : ML Router wins big (different method than rule)
  - dbpedia560k / and   : Rule fails (UNG max recall < 0.9)
  - yahoo800k / equal   : scaling-bias visible (ML pink curve appears below UNG baseline)

Reuses the same computation/plotting machinery as plot_v2_pareto_perquery.py
but restricted to the three cells, in a 1 x 3 row layout.
"""
from __future__ import annotations
import csv as _csv
from pathlib import Path
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Reuse baseline loaders from plot_v2_recall_qps
_spec = importlib.util.spec_from_file_location("plot_v2",
    str(REPO / "analysis/plot_v2_recall_qps.py"))
plot_v2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(plot_v2)

SCEN_FV = {"and": "containment", "or": "overlap", "equal": "equality"}

def load_acorn(ds, sc):
    p = REPO / f"ACORN/data/param_search_{ds}/results/{ds}/summary.csv"
    if not p.exists(): return []
    pts = []
    with open(p) as f:
        for r in _csv.DictReader(f):
            if r.get("scenario") != sc or r.get("status") != "success": continue
            try:
                rec = float(r["recall@10"]); qps = float(r["qps"])
                if rec > 0 and qps > 0: pts.append((qps, rec))
            except (ValueError, KeyError):
                continue
    return sorted(pts, key=lambda p: p[1])

def load_fv(ds, sc):
    p = REPO / f"DiskANN/data_original/results/{ds}/summary.csv"
    if not p.exists(): return []
    pts = []
    with open(p) as f:
        for r in _csv.DictReader(f):
            if (r.get("dataset") != ds or r.get("scenario") != SCEN_FV[sc]
                    or r.get("status") != "success"): continue
            try:
                rec = float(r["recall@10"]); qps = float(r["qps"])
                if rec > 0 and qps > 0: pts.append((qps, rec))
            except (ValueError, KeyError):
                continue
    return sorted(pts, key=lambda p: p[1])

# Scaling factors
SCALING = {}
with open(REPO / "analysis/ml_router/scaling_factors_audited.csv") as f:
    for r in _csv.DictReader(f):
        SCALING[r["method"]] = float(r["scaling"])

# Master per-query data
master = pd.read_pickle(REPO / "analysis/ml_router/master_per_query.pkl")
preds = pd.read_csv(REPO / "analysis/ml_router/results/v2_5methods/full_train_val_predictions_minimal_reg.csv")

PREDS_TO_MASTER = {"UNG": "UNG", "Post-filter": "PF", "SIEVE": "SIEVE",
                   "ACORN": "ACORN", "FilteredVamana": "FV"}
PRED_METHODS = list(PREDS_TO_MASTER)

# Cell-aggregate for lookup table
keys = ["method", "dataset", "scenario", "config"]
cell_agg = master.groupby(keys).agg(
    mean_recall=("recall_at_10", "mean"),
    mean_lat=("latency_us", "mean"),
).reset_index()
cell_agg["qps_16t"] = cell_agg["method"].map(SCALING) * 1e6 / cell_agg["mean_lat"]

T_GRID = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def router_strict(ds: str, sc: str, T: float):
    pred_cols = {pm: f"MLP-Reg_predrecall_{pm}" for pm in PRED_METHODS}
    pgrp = preds[(preds.dataset == ds) & (preds.scenario == sc)]
    if pgrp.empty:
        return None

    best = {}
    for pm in PRED_METHODS:
        m = PREDS_TO_MASTER[pm]
        sub = cell_agg[(cell_agg.method == m) & (cell_agg.dataset == ds)
                       & (cell_agg.scenario == sc) & (cell_agg.mean_recall >= T)]
        if not sub.empty:
            b = sub.nlargest(1, "qps_16t").iloc[0]
            best[pm] = (b.config, b.qps_16t)
    if not best:
        return None

    lookup = {}
    for pm, (cfg, _) in best.items():
        m = PREDS_TO_MASTER[pm]
        sub = master[(master.method == m) & (master.dataset == ds)
                     & (master.scenario == sc) & (master.config == cfg)]
        lookup[pm] = sub.set_index("query_id")[["latency_us", "recall_at_10"]]

    rs, eff = [], []
    for _, q in pgrp.iterrows():
        preds_q = {pm: q[pred_cols[pm]] for pm in best}
        passing = [pm for pm in best if preds_q[pm] >= T]
        pm_ch = (max(passing, key=lambda pm: best[pm][1]) if passing
                 else max(best, key=lambda pm: preds_q[pm]))
        row = lookup[pm_ch].loc[int(q.query_id)]
        rs.append(row.recall_at_10)
        eff.append(row.latency_us / SCALING[PREDS_TO_MASTER[pm_ch]])
    if not rs:
        return None
    return float(np.mean(rs)), 1e6 * len(eff) / sum(eff)


# Compute rule pick per cell
rule_pick = {
    (ds, sc): grp["RuleRouter_pred"].mode().iloc[0]
    for (ds, sc), grp in preds.groupby(["dataset", "scenario"])
}

# Cells to plot (top banner)
CELLS = [
    ("dbpedia560k", "or",    "ML Router wins by switching methods"),
    ("dbpedia560k", "and",   "Rule fails: UNG cannot reach T=0.9"),
    ("yahoo800k",   "equal", "Equal: ML Router routes to UNG (curve underestimated by scaling)"),
]

COLORS = {"UNG": "#1f77b4", "Post-filter": "#ff7f0e", "SIEVE": "#2ca02c",
          "ACORN": "#d62728", "FilteredVamana": "#9467bd"}
MARKERS = {"UNG": "o", "Post-filter": "s", "SIEVE": "^", "ACORN": "D",
           "FilteredVamana": "v"}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), squeeze=False)

for i, (ds, sc, subtitle) in enumerate(CELLS):
    ax = axes[0, i]
    baseline_pts = {}
    for name, loader in [("UNG", plot_v2.load_ung),
                         ("Post-filter", plot_v2.load_postfilter),
                         ("SIEVE", plot_v2.load_sieve),
                         ("ACORN", load_acorn),
                         ("FilteredVamana", load_fv)]:
        try:
            pts = loader(ds, sc)
        except Exception:
            pts = []
        if pts:
            baseline_pts[name] = pts
            xs = [p[1] for p in pts]
            ys = [p[0] for p in pts]
            ax.plot(xs, ys, color=COLORS[name], marker=MARKERS[name],
                    label=name, lw=1.0, ms=4, alpha=0.65)

    # ML Router strict T-sweep curve
    xs, ys = [], []
    for T in T_GRID:
        out = router_strict(ds, sc, T)
        if out is not None:
            xs.append(out[0]); ys.append(out[1])
    if xs:
        ax.plot(xs, ys, ls="-", color="magenta", marker="*", ms=9, lw=2.2,
                label="ML Router", alpha=0.95, zorder=10)

    # RuleRouter pick annotation
    pm_rule = rule_pick.get((ds, sc))
    if pm_rule:
        ax.text(0.02, 0.02, f"RuleRouter pick: {pm_rule}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.75, edgecolor="gray"))

    ax.set_yscale("log")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"{ds} / {sc}\n{subtitle}", fontsize=9)
    ax.set_xlabel("Recall@10")
    if i == 0:
        ax.set_ylabel("QPS @ 16 threads (log)")
    ax.grid(True, alpha=0.3)

# Figure-level legend on top
handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in labels:
            handles.append(hi); labels.append(li)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
           ncol=6, fontsize=9, frameon=True, columnspacing=1.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out = REPO / "analysis/figures/v2_recall_qps_perquery_3cell.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out}")
