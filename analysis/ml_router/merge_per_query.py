#!/usr/bin/env python3
"""Merge all per-query CSVs in analysis/ml_router/per_query_results/
into one master DataFrame, saved as master_per_query.pkl.

Each row: method, dataset, scenario, config, query_id, latency_us, recall_at_10
"""
import pandas as pd, glob, os, re
from pathlib import Path

DATASETS = ['synth_192d', 'synth_512d', 'synth_768d_hc', 'yahoo800k', 'dbpedia560k']
DS_PAT = '|'.join(DATASETS)

PATTERNS = {
    'PF':    re.compile(rf'({DS_PAT})_(and|or|equal)_PF_M(\d+)_efc(\d+)_M=\d+_efc=\d+_ef=(\d+)$'),
    'UNG':   re.compile(rf'({DS_PAT})_(and|or|equal)_UNG_M(\d+)_Lb(\d+)_L(\d+)$'),
    'ACORN': re.compile(rf'({DS_PAT})_(and|or|equal)_ACORN_M(\d+)_Mb(\d+)_g(\d+)_M=\d+_M_beta=\d+_gamma=\d+_ef=(\d+)$'),
    'FV':    re.compile(rf'({DS_PAT})_(and|or|equal)_FV_R(\d+)_L(\d+)$'),
    'SIEVE': re.compile(rf'({DS_PAT})_(and|or|equal)_SIEVE_M(\d+)_efc(\d+)_b([\d.]+)_h([\d.]+)_M\d+_b[\d.]+_h[\d.]+_ef(\d+)_(and|or|eq|original_and|original_or|original_eq)$'),
}

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / 'analysis/ml_router/per_query_results'

all_rows, unmatched = [], []
for f in sorted(glob.glob(str(DIR / '*.csv'))):
    name = os.path.basename(f).replace('.csv', '')
    for method, pat in PATTERNS.items():
        m = pat.match(name)
        if not m: continue
        ds, sc = m.group(1), m.group(2)
        if method == 'PF':
            cfg = f'M={m.group(3)}_efc={m.group(4)}_ef={m.group(5)}'
        elif method == 'UNG':
            cfg = f'max_degree={m.group(3)}_Lbuild={m.group(4)}_Lsearch={m.group(5)}'
        elif method == 'ACORN':
            cfg = f'M={m.group(3)}_M_beta={m.group(4)}_gamma={m.group(5)}_ef={m.group(6)}'
        elif method == 'FV':
            cfg = f'R={m.group(3)}_Lsearch={m.group(4)}'
        elif method == 'SIEVE':
            cfg = (f'M={m.group(3)}_ef_construction={m.group(4)}'
                   f'_index_budget={m.group(5)}_hist_pct={m.group(6)}'
                   f'_ef_search={m.group(7)}')
        df = pd.read_csv(f)
        df['method'] = method
        df['dataset'] = ds
        df['scenario'] = sc
        df['config'] = cfg
        all_rows.append(df)
        break
    else:
        unmatched.append(name)

if unmatched:
    print(f'WARN: {len(unmatched)} unmatched; first 3:')
    for u in unmatched[:3]: print(f'  {u}')

master = pd.concat(all_rows, ignore_index=True)
print(f'Master: {len(master):,} rows, {master.method.value_counts().to_dict()}')

out = REPO / 'analysis/ml_router/master_per_query.pkl'
master.to_pickle(out)
print(f'Saved {out} ({os.path.getsize(out) / 1024 / 1024:.1f} MB)')
