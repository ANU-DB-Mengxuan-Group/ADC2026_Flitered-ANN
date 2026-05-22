#!/usr/bin/env python3
"""V2 recall-QPS Pareto with per-query data.

Baseline: existing 16-thread cell-avg from summary.csv (full config sweep)
Router:   per-query latency × per-method scaling factor + per-query recall

Two figures per run:
  analysis/figures/v2_recall_qps_perquery.{png,pdf}        — empirical scaling
  analysis/figures/v2_recall_qps_perquery_x16.{png,pdf}   — naive ×16 scaling

Usage:
  python3 analysis/plot_v2_pareto_perquery.py
  python3 analysis/plot_v2_pareto_perquery.py --scaling x16    # ×16 naive only
  python3 analysis/plot_v2_pareto_perquery.py --scaling empirical
"""
import argparse, csv, glob, importlib.util
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# Reuse baseline loaders from existing script
_spec = importlib.util.spec_from_file_location(
    "plot_v2", str(REPO / "analysis/plot_v2_recall_qps.py"))
plot_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_v2)


def load_scaling(mode):
    """mode = 'empirical' | 'x16'."""
    if mode == 'x16':
        return {'SIEVE': 16, 'PF': 16, 'UNG': 16, 'ACORN': 16, 'FV': 16}
    path = REPO / 'analysis/ml_router/scaling_factors_audited.csv'
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['method']] = float(r['scaling'])
    return out


_ap = argparse.ArgumentParser()
_ap.add_argument('--scaling', choices=['empirical', 'x16'], default='empirical')
_args = _ap.parse_args()
SCALING_MODE = _args.scaling
SCALING = load_scaling(SCALING_MODE)
print(f'Scaling mode: {SCALING_MODE} → {SCALING}')

SCEN_FV = {"and": "containment", "or": "overlap", "equal": "equality"}


def load_acorn(dataset, scenario):
    """ACORN baseline: scatter all configs, sorted by recall."""
    path = REPO / f"ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv"
    if not path.exists():
        return []
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("scenario") != scenario or r.get("status") != "success":
                continue
            try:
                rec = float(r["recall@10"]); qps = float(r["qps"])
                if rec > 0 and qps > 0:
                    pts.append((qps, rec))
            except (ValueError, KeyError):
                continue
    return sorted(pts, key=lambda p: p[1])


def load_fv(dataset, scenario):
    """FilteredVamana baseline."""
    path = REPO / f"DiskANN/data_original/results/{dataset}/summary.csv"
    if not path.exists():
        return []
    fv_sc = SCEN_FV[scenario]
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("dataset") != dataset or r.get("scenario") != fv_sc
                    or r.get("status") != "success"):
                continue
            try:
                rec = float(r["recall@10"]); qps = float(r["qps"])
                if rec > 0 and qps > 0:
                    pts.append((qps, rec))
            except (ValueError, KeyError):
                continue
    return sorted(pts, key=lambda p: p[1])
master = pd.read_pickle(REPO / 'analysis/ml_router/master_per_query.pkl')
preds = pd.read_csv(REPO / 'analysis/ml_router/results/v2_5methods/full_train_val_predictions_minimal_reg.csv')

THREADS = 16  # baseline QPS measured at 16 threads
PREDS_TO_MASTER = {'UNG': 'UNG', 'Post-filter': 'PF', 'SIEVE': 'SIEVE',
                   'ACORN': 'ACORN', 'FilteredVamana': 'FV'}

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS = ["and", "or", "equal"]

# ============ Per-cell aggregate from master per-query ============
keys = ['method', 'dataset', 'scenario', 'config']
cell_agg = master.groupby(keys).agg(
    mean_recall=('recall_at_10', 'mean'),
    mean_lat=('latency_us', 'mean'),
).reset_index()
# Use empirical per-method scaling factor (not naive ×16)
cell_agg['scale_factor'] = cell_agg['method'].map(SCALING)
cell_agg['qps_16t'] = cell_agg['scale_factor'] * 1e6 / cell_agg['mean_lat']

# ============ Router curve via T sweep ============
T_GRID = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

def router_per_query(T: float, mode: str):
    """Per query, MLP picks method. Returns dict[(ds,sc)] = (recall, qps)."""
    pred_cols = {pm: f'MLP-Reg_predrecall_{pm}' for pm in PREDS_TO_MASTER}
    pred_methods = list(PREDS_TO_MASTER)
    out = {}
    for (ds, sc), pgrp in preds.groupby(['dataset', 'scenario']):
        # Per-method best config at T: from cell_agg, recall >= T → max qps
        best = {}
        for pm in pred_methods:
            m = PREDS_TO_MASTER[pm]
            sub = cell_agg[(cell_agg.method == m) & (cell_agg.dataset == ds)
                           & (cell_agg.scenario == sc) & (cell_agg.mean_recall >= T)]
            if not sub.empty:
                b = sub.nlargest(1, 'qps_16t').iloc[0]
                best[pm] = (b.config, b.qps_16t)
        if not best: continue
        # per-query lookup
        lookup = {}
        for pm, (cfg, _) in best.items():
            m = PREDS_TO_MASTER[pm]
            sub = master[(master.method == m) & (master.dataset == ds)
                         & (master.scenario == sc) & (master.config == cfg)]
            lookup[pm] = sub.set_index('query_id')[['latency_us', 'recall_at_10']]
        all_r, eff_times_us = [], []
        for _, q in pgrp.iterrows():
            qid = int(q.query_id)
            preds_q = {pm: q[pred_cols[pm]] for pm in best}
            if mode == 'plain':
                pm_chosen = max(best, key=lambda pm: preds_q[pm])
            else:
                passing = [pm for pm in best if preds_q[pm] >= T]
                pm_chosen = (max(passing, key=lambda pm: best[pm][1]) if passing
                             else max(best, key=lambda pm: preds_q[pm]))
            row = lookup[pm_chosen].loc[qid]
            all_r.append(row.recall_at_10)
            # 16-thread equivalent time = single-thread latency / per-method scaling
            m_master = PREDS_TO_MASTER[pm_chosen]
            eff_times_us.append(row.latency_us / SCALING[m_master])
        if all_r:
            qps = 1e6 * len(eff_times_us) / sum(eff_times_us)
            out[(ds, sc)] = (np.mean(all_r), qps)
    return out

print('Computing router curves...')
router = {('strict', T): router_per_query(T, 'strict') for T in T_GRID}


# ============ RuleRouter pick per cell (single point on baseline curve) ============
# RuleRouter outputs one fixed method per cell. Its deploy (recall, QPS) at any
# T_target lies on that method's baseline curve (16-thread real measurement from
# summary.csv). We mark the point at T_target = T_RULE on each subplot.
rule_pick = {}
for (ds, sc), pgrp in preds.groupby(['dataset', 'scenario']):
    rule_pick[(ds, sc)] = pgrp['RuleRouter_pred'].mode().iloc[0]

T_RULE = 0.9  # deploy target for RuleRouter point


# ============ Plot per (ds, sc) panel ============
COLORS = {'UNG': '#1f77b4', 'Post-filter': '#ff7f0e', 'SIEVE': '#2ca02c',
          'ACORN': '#d62728', 'FilteredVamana': '#9467bd'}
MARKERS = {'UNG': 'o', 'Post-filter': 's', 'SIEVE': '^', 'ACORN': 'D',
           'FilteredVamana': 'v'}

fig, axes = plt.subplots(len(V2_DATASETS), len(SCENARIOS),
                         figsize=(14, 3 * len(V2_DATASETS)),
                         sharex=False, sharey=False)
for i, ds in enumerate(V2_DATASETS):
    for j, sc in enumerate(SCENARIOS):
        ax = axes[i, j]
        # Baselines from old multi-thread data
        any_baseline = False
        baseline_loaders = {'UNG': plot_v2.load_ung,
                            'Post-filter': plot_v2.load_postfilter,
                            'SIEVE': plot_v2.load_sieve,
                            'ACORN': load_acorn,
                            'FilteredVamana': load_fv}
        baseline_pts = {}
        for name, loader in baseline_loaders.items():
            try:
                pts = loader(ds, sc)
                if pts:
                    baseline_pts[name] = pts
                    xs = [p[1] for p in pts]
                    ys = [p[0] for p in pts]
                    ax.plot(xs, ys, color=COLORS[name], marker=MARKERS[name],
                            label=name, lw=1.2, ms=5, alpha=0.7)
                    any_baseline = True
            except Exception:
                pass
        # ML Router strict curve (主角)
        xs, ys = [], []
        for T in T_GRID:
            if (ds, sc) in router[('strict', T)]:
                r, q = router[('strict', T)][(ds, sc)]
                xs.append(r); ys.append(q)
        if xs:
            ax.plot(xs, ys, ls='-', color='magenta', marker='*', ms=10, lw=2.2,
                    label='ML Router (MLP-Reg)', alpha=0.95, zorder=10)
        # RuleRouter pick: text annotation only (no point on curve)
        pm_rule = rule_pick.get((ds, sc))
        if pm_rule:
            ax.text(0.02, 0.02, f"RuleRouter pick: {pm_rule}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                              alpha=0.75, edgecolor='gray'))
        ax.set_yscale('log')
        ax.set_xlim(0, 1.02)
        ax.set_title(f'{ds} / {sc}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == len(V2_DATASETS) - 1:
            ax.set_xlabel('Recall@10')
        if j == 0:
            ax.set_ylabel('QPS @ 16 threads (log)')

_scaling_label = '×16 (naive)' if SCALING_MODE == 'x16' else '× per-method scaling (audited)'
plt.suptitle(f'V2 recall-QPS Pareto: baselines (16-thread real) + ML Router ({_scaling_label})',
             y=1.005, fontsize=12)

# Figure-level legend at top (above suptitle area)
handles, labels = [], []
for row in axes:
    for ax in row:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.995), ncol=7, fontsize=9,
           frameon=True, columnspacing=1.2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
_suffix = '_x16' if SCALING_MODE == 'x16' else ''
out = REPO / f'analysis/figures/v2_recall_qps_perquery{_suffix}.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
plt.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
print(f'\nSaved {out}')

# Aggregate numbers
print('\n=== Headline numbers (mean across 15 cells) ===')
print(f'{"variant":<14} {"T":>5} {"mean recall":>12} {"mean QPS":>10} {"n_cells":>9}')
for T in T_GRID:
    d = router[('strict', T)]
    if d:
        mr = np.mean([v[0] for v in d.values()])
        mq = np.mean([v[1] for v in d.values()])
        print(f'{"ML Router":<14} {T:>5} {mr:>12.4f} {mq:>10.1f} {len(d):>9}')
print('\nRuleRouter picks per cell (overlap with baseline curve of that method):')
for (ds, sc), pm in sorted(rule_pick.items()):
    print(f'  {ds:<14s} {sc:<6s} -> {pm}')
