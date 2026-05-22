#!/usr/bin/env python3
"""Merge per-method scaling_factors_audited_*.csv into one master CSV."""
import pandas as pd, glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analysis/ml_router/scaling_factors_audited.csv'

files = sorted(glob.glob(str(REPO / 'analysis/ml_router/scaling_factors_audited_*.csv')))
print(f'found {len(files)} per-method CSVs')
dfs = [pd.read_csv(f) for f in files]
m = pd.concat(dfs, ignore_index=True)
m = m.drop_duplicates(subset=['method'], keep='last')
m.to_csv(OUT, index=False)
print(m.to_string(index=False))
print(f'\nSaved {OUT}')
