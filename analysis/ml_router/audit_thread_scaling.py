#!/usr/bin/env python3
"""Audit per-method 16-thread vs 1-thread scaling on representative configs.

For each method, build index (if missing), run search in 16-thread mode
WITHOUT PER_QUERY_CSV (so binary uses its default parallel path), parse
the binary's reported QPS, then compare to single-thread per-query QPS
from master_per_query.pkl.

Empirical scaling = real_16t_QPS / single_thread_QPS
Ideal = 16 (linear). Realistic for memory-rich servers: 12-15.

Run on cluster:
  ssh weirdo
  cd ~/benchmarks/discrete && conda activate benchmark
  python3 -u analysis/online_routing/audit_thread_scaling.py
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    'DATA_DIR', '/home/remote/u7905817/benchmarks/datasets/discrete'))
MASTER = REPO / 'analysis/ml_router/master_per_query.pkl'

THREADS = 16

# One mid-range config per method (recall ~0.9 region)
AUDITS = [
    {'method': 'PF', 'ds': 'dbpedia560k', 'sc': 'and',
     'M': 64, 'efc': 400, 'ef_target': 400},
    {'method': 'UNG', 'ds': 'dbpedia560k', 'sc': 'equal',
     'max_degree': 96, 'Lbuild': 200, 'Lsearch': 500},
    {'method': 'SIEVE', 'ds': 'dbpedia560k', 'sc': 'and',
     'M': 32, 'efc': 40, 'b': 2.0, 'h': 0.25, 'ef_target': 150},
    {'method': 'ACORN', 'ds': 'dbpedia560k', 'sc': 'and',
     'M': 32, 'Mb': 64, 'g': 4, 'ef_target': 200},
    {'method': 'FV', 'ds': 'dbpedia560k', 'sc': 'and',
     'R': 64, 'Lsearch': 200},
]

SC_FV = {'and': 'containment', 'or': 'overlap', 'equal': 'equality'}
SC_UNG_SEARCH = {'and': 'containment', 'or': 'overlap', 'equal': 'equality'}
SC_UNG_BUILD  = {'and': 'general',     'or': 'general',  'equal': 'equality'}
SC_UNG_SUFFIX = {'and': 'and',         'or': 'or',       'equal': 'equal'}
SC_SIEVE = {'and': 'original_and', 'or': 'original_or', 'equal': 'original_eq'}

DATASET_N = {
    'synth_192d': 800000, 'synth_512d': 800000, 'synth_768d_hc': 800000,
    'yahoo800k': 800000, 'dbpedia560k': 560000,
}


def env_no_per_query(method='other'):
    """Env with MKL/LD set, PER_QUERY_CSV unset (16-thread mode)."""
    env = os.environ.copy()
    cp = os.environ.get('CONDA_PREFIX')
    parts = []
    if method == 'ACORN':
        parts.append(str(REPO / 'ACORN/build/faiss'))  # ACORN-patched libfaiss
    if cp:
        parts.append(os.path.join(cp, 'lib'))
    if env.get('LD_LIBRARY_PATH'):
        parts.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH'] = ':'.join(parts)
    env.pop('PER_QUERY_CSV', None)
    return env


def get_1based(path):
    p = Path(path)
    onebased = p.parent / f'{p.stem}_1based{p.suffix}'
    return onebased if onebased.exists() else p


_MASTER_CACHE = None

def single_thread_qps(method, ds, sc, config_str):
    global _MASTER_CACHE
    if _MASTER_CACHE is None:
        if not MASTER.exists():
            print(f'  WARN: master_per_query.pkl missing at {MASTER}')
            print(f'  Run merge_per_query.py first to generate it.')
            return None
        _MASTER_CACHE = pd.read_pickle(MASTER)
    df = _MASTER_CACHE
    sub = df[(df.method == method) & (df.dataset == ds) & (df.scenario == sc)
             & (df.config == config_str)]
    if sub.empty:
        print(f'  WARN: no master data for {method}/{ds}/{sc} cfg={config_str}')
        return None
    return 1e6 / sub.latency_us.mean()


# ============ PF ============
def audit_pf(a):
    ds, sc, M, efc, ef = a['ds'], a['sc'], a['M'], a['efc'], a['ef_target']
    idx_root = REPO / 'faiss/data/index_files/hnsw'
    idx_path = idx_root / ds / f'M={M}_efc={efc}.json'
    build_bin = REPO / 'faiss/tutorial/cpp/build_HNSW_index_static'
    search_bin = REPO / 'faiss/tutorial/cpp/search_HNSW_index_static'
    env = env_no_per_query()
    if not idx_path.exists():
        print(f'  [build] PF {ds} M={M} efc={efc}...')
        (idx_root / ds).mkdir(parents=True, exist_ok=True)
        cmd = [str(build_bin), str(DATA_DIR / ds / f'{ds}_base.fvecs'),
               str(M), str(efc), str(idx_root), ds]
        t0 = time.time()
        subprocess.run(cmd, env=env, capture_output=True, timeout=86400)
        print(f'  build done in {time.time()-t0:.1f}s')
    res_dir = REPO / 'audit_results/PF'
    res_dir.mkdir(parents=True, exist_ok=True)
    N = DATASET_N[ds]
    cmd = [str(search_bin), ds, str(M), str(efc), str(idx_root), sc, str(res_dir),
           str(DATA_DIR / ds / f'{ds}_base.fvecs'),
           str(DATA_DIR / ds / 'label_base.txt'),
           str(DATA_DIR / ds / f'{ds}_query_{sc}.fvecs'),
           str(DATA_DIR / ds / f'{ds}_query_{sc}.txt'),
           str(DATA_DIR / ds / f'{ds}_gt_{sc}.txt'), '10', str(N)]
    print(f'  [search 16t] PF {ds}/{sc}...')
    t0 = time.time()
    subprocess.run(cmd, env=env, capture_output=True, timeout=3600)
    wall = time.time() - t0
    print(f'  wall={wall:.1f}s')
    res_csv = res_dir / f'M={M}_efc={efc}_result.csv'
    if not res_csv.exists():
        # Try any CSV in res_dir
        cands = list(res_dir.glob('*.csv'))
        print(f'  result CSV expected at {res_csv} not found; found: {cands}')
        return None, None
    # Inspect columns
    with open(res_csv) as f:
        header = next(csv.reader(f))
    print(f'  result CSV columns: {header}')
    ef_col = next((c for c in ['L', 'efSearch', 'ef_search', 'ef'] if c in header), None)
    qps_col = next((c for c in ['QPS', 'qps', 'qps_no_filter'] if c in header), None)
    if ef_col is None or qps_col is None:
        print(f'  could not find ef/qps columns')
        return None, None
    with open(res_csv) as f:
        for r in csv.DictReader(f):
            try:
                if int(r[ef_col]) == ef:
                    return float(r[qps_col]), f'M={M}_efc={efc}_ef={ef}'
            except (KeyError, ValueError):
                continue
    return None, None


# ============ UNG ============
def audit_ung(a):
    ds, sc, M, Lb, Ls = a['ds'], a['sc'], a['max_degree'], a['Lbuild'], a['Lsearch']
    build_sc = SC_UNG_BUILD[sc]; search_sc = SC_UNG_SEARCH[sc]
    suffix = SC_UNG_SUFFIX[sc]
    idx_dir = REPO / f'UNG-dev/indices_original/{ds}/{build_sc}'
    idx_prefix = idx_dir / f'index_M={M}_L={Lb}'
    build_bin = REPO / 'UNG-dev/build/apps/build_UNG_index'
    search_bin = REPO / 'UNG-dev/build/apps/search_UNG_index'
    env = env_no_per_query()
    files = list(idx_dir.glob(f'index_M={M}_L={Lb}*'))
    has_files = any(f.is_file() for f in files)
    if not has_files:
        print(f'  [build] UNG {ds}/{build_sc} M={M} Lb={Lb}...')
        idx_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(build_bin), '--data_type', 'float', '--dist_fn', 'L2',
               '--base_bin_file', str(DATA_DIR / ds / f'{ds}_base.bin'),
               '--base_label_file', str(get_1based(DATA_DIR / ds / 'label_base.txt')),
               '--index_path_prefix', str(idx_prefix),
               '--scenario', build_sc, '--max_degree', str(M),
               '--Lbuild', str(Lb), '--num_cross_edges', '6',
               '--num_threads', str(THREADS)]
        t0 = time.time()
        subprocess.run(cmd, env=env, capture_output=True, timeout=7200)
        print(f'  build done in {time.time()-t0:.1f}s')
    res_dir = REPO / 'audit_results/UNG'
    res_dir.mkdir(parents=True, exist_ok=True)
    result_prefix = res_dir / f'M={M}_L={Lb}_Ls={Ls}_'
    cmd = [str(search_bin), '--data_type', 'float', '--dist_fn', 'L2',
           '--base_bin_file', str(DATA_DIR / ds / f'{ds}_base.bin'),
           '--query_bin_file', str(DATA_DIR / ds / f'{ds}_query_{suffix}.bin'),
           '--base_label_file', str(get_1based(DATA_DIR / ds / 'label_base.txt')),
           '--query_label_file', str(get_1based(DATA_DIR / ds / f'{ds}_query_{suffix}.txt')),
           '--gt_file', str(DATA_DIR / ds / f'{ds}_gt_{suffix}.bin'),
           '--K', '10', '--index_path_prefix', str(idx_prefix),
           '--scenario', search_sc, '--Lsearch', str(Ls),
           '--num_threads', str(THREADS),
           '--result_path_prefix', str(result_prefix)]
    print(f'  [search 16t] UNG {ds}/{sc} Ls={Ls}...')
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=3600, text=True)
    wall = time.time() - t0
    print(f'  wall={wall:.1f}s')
    # search_UNG_index prints lines like "- Lsearch=500, time=1181ms"
    # → QPS = 1000 * 1000 / time_ms = 1e6 / time_ms (in queries/sec, assuming 1000 queries)
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(rf'Lsearch={Ls},\s*time=(\d+)ms', out)
    if m:
        time_ms = int(m.group(1))
        real_qps = 1000 * 1000.0 / time_ms
        return real_qps, f'max_degree={M}_Lbuild={Lb}_Lsearch={Ls}'
    print(f'  WARN: could not parse UNG QPS from output (last 500 chars):')
    print(f'  ...{out[-500:]}')
    return None, None


# ============ SIEVE ============
def audit_sieve(a):
    ds, sc, M, efc, b, h, ef = a['ds'], a['sc'], a['M'], a['efc'], a['b'], a['h'], a['ef_target']
    sieve_sc = SC_SIEVE[sc]
    env = env_no_per_query()
    cmd = ['python3', str(REPO / 'SIEVE/run_sieve.py'),
           '--dataset', ds, '--scenario', sieve_sc,
           '--M', str(M), '--ef_construction', str(efc),
           '--index_budget', str(b), '--hist_pct', str(h),
           '--num_threads', str(THREADS)]
    print(f'  [build+search 16t] SIEVE {ds}/{sieve_sc} M={M} b={b} h={h}...')
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(REPO),
                       capture_output=True, timeout=7200, text=True)
    wall = time.time() - t0
    print(f'  wall={wall:.1f}s')
    csv_path = REPO / f'SIEVE/results/sieve_{ds}_{sieve_sc}_M{M}_b{b}_h{h}.csv'
    if not csv_path.exists():
        print(f'  WARN: SIEVE result {csv_path} not found')
        return None, None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                if int(r['ef_search']) == ef:
                    return float(r['qps']), (
                        f'M={M}_ef_construction={efc}_index_budget={b}'
                        f'_hist_pct={h}_ef_search={ef}')
            except (KeyError, ValueError):
                continue
    return None, None


# ============ ACORN ============
def audit_acorn(a):
    ds, sc, M, Mb, g, ef = a['ds'], a['sc'], a['M'], a['Mb'], a['g'], a['ef_target']
    N = DATASET_N[ds]
    output_base = REPO / f'ACORN/data/param_search_{ds}'
    indices_base = output_base / 'indices'
    idx_path = indices_base / ds / f'hybrid_M={M}_Mb={Mb}_gamma={g}.json'
    build_bin = REPO / 'ACORN/build/demos/build_acorn_index'
    search_bin = REPO / 'ACORN/build/demos/search_acorn_index'
    env = env_no_per_query(method='ACORN')
    if not idx_path.exists():
        print(f'  [build] ACORN {ds} M={M} Mb={Mb} g={g}...')
        (indices_base / ds).mkdir(parents=True, exist_ok=True)
        cmd = [str(build_bin), str(N), str(g),
               str(DATA_DIR / ds / f'{ds}_base.fvecs'),
               str(M), str(Mb), str(indices_base), ds]
        t0 = time.time()
        subprocess.run(cmd, env=env, capture_output=True, timeout=18000)
        print(f'  build done in {time.time()-t0:.1f}s')
    res_dir = output_base / 'results' / ds / sc
    res_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(search_bin), str(N), str(g), ds, str(M), str(Mb),
           str(indices_base), sc, str(res_dir),
           str(DATA_DIR / ds / f'{ds}_base.fvecs'),
           str(DATA_DIR / ds / 'label_base.txt'),
           str(DATA_DIR / ds / f'{ds}_query_{sc}.fvecs'),
           str(DATA_DIR / ds / f'{ds}_query_{sc}.txt'),
           str(DATA_DIR / ds / f'{ds}_gt_{sc}.txt'), '10']
    print(f'  [search 16t] ACORN {ds}/{sc}...')
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=3600, text=True)
    wall = time.time() - t0
    print(f'  wall={wall:.1f}s')
    out = (r.stdout or '') + (r.stderr or '')
    # ACORN prints per-ef: "QPS: <value>" lines under "efSearch=X"
    # Find ef==target then next QPS line
    blocks = re.split(r'efSearch=(\d+)', out)
    # blocks alternates [pre, ef1_val, after1, ef2_val, after2, ...]
    for i in range(1, len(blocks), 2):
        try:
            efv = int(blocks[i])
        except ValueError: continue
        if efv == ef:
            chunk = blocks[i+1]
            m = re.search(r'QPS:\s*([\d.]+)', chunk)
            if m:
                return float(m.group(1)), (
                    f'M={M}_M_beta={Mb}_gamma={g}_ef={ef}')
    print(f'  WARN: could not parse ACORN QPS for ef={ef}')
    return None, None


# ============ FV ============
def audit_fv(a):
    ds, sc, R, Ls = a['ds'], a['sc'], a['R'], a['Lsearch']
    search_sc = SC_FV[sc]
    out_base = REPO / 'DiskANN/data_original'
    idx_dir = out_base / 'indices' / ds
    idx_prefix = idx_dir / f'index_R={R}'
    build_bin = REPO / 'DiskANN/build/apps/build_memory_index'
    search_bin = REPO / 'DiskANN/build/apps/search_memory_index'
    env = env_no_per_query()
    if not Path(f'{idx_prefix}.data').exists():
        print(f'  [build] FV {ds} R={R}...')
        idx_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(build_bin), '--data_type', 'float', '--dist_fn', 'l2',
               '--data_path', str(DATA_DIR / ds / f'{ds}_base.bin'),
               '--index_path_prefix', str(idx_prefix),
               '--label_file', str(get_1based(DATA_DIR / ds / 'label_base.txt')),
               '-R', str(R), '-L', '100', '--alpha', '1.2', '-T', str(THREADS)]
        t0 = time.time()
        subprocess.run(cmd, env=env, capture_output=True, timeout=7200)
        print(f'  build done in {time.time()-t0:.1f}s')
    converted_labels = out_base / 'converted_labels' / ds / search_sc / 'query_labels.txt'
    gt_file = out_base / 'converted_labels' / ds / search_sc / 'gt_diskann.bin'
    if not converted_labels.exists() or not gt_file.exists():
        print(f'  WARN: FV converted labels missing for {ds}/{search_sc}')
        return None, None
    res_dir = out_base / 'results' / ds
    res_dir.mkdir(parents=True, exist_ok=True)
    result_prefix = res_dir / f'index_R={R}_{search_sc}_audit'
    cmd = [str(search_bin), '--data_type', 'float', '--dist_fn', 'l2',
           '--index_path_prefix', str(idx_prefix),
           '--query_file', str(DATA_DIR / ds / f'{ds}_query_{sc}.bin'),
           '--query_filters_file', str(converted_labels),
           '--gt_file', str(gt_file),
           '--label_type', 'uint', '--recall_at', '10',
           '--result_path', str(result_prefix),
           '--num_threads', str(THREADS),
           '--filter_scenario', search_sc, '-L', str(Ls)]
    print(f'  [search 16t] FV {ds}/{sc} R={R} Ls={Ls}...')
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=3600, text=True)
    wall = time.time() - t0
    print(f'  wall={wall:.1f}s')
    out = (r.stdout or '') + (r.stderr or '')
    # DiskANN search_memory_index prints: setw(4) L + setw(12) QPS + setw(20) mean_lat + ...
    # Match: "  Ls   QPS   mean_lat" → capture QPS (the FIRST number after Ls).
    m = re.search(rf'\b{Ls}\s+([\d.]+)\s+[\d.]+', out)
    if m:
        return float(m.group(1)), f'R={R}_Lsearch={Ls}'
    print(f'  WARN: could not parse FV QPS (last 500 chars):')
    print(f'  ...{out[-500:]}')
    return None, None


HANDLERS = {'PF': audit_pf, 'UNG': audit_ung, 'SIEVE': audit_sieve,
            'ACORN': audit_acorn, 'FV': audit_fv}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=None,
                    help='Only run these methods (PF/UNG/SIEVE/ACORN/FV)')
    args = ap.parse_args()
    audits = AUDITS if not args.methods else [
        a for a in AUDITS if a['method'] in args.methods]
    print(f'Running audit for {[a["method"] for a in audits]}')
    results = []
    for a in audits:
        m = a['method']
        print(f'\n========== {m} ==========')
        try:
            real_qps_16t, cfg_master = HANDLERS[m](a)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  FAILED: {e}')
            continue
        if real_qps_16t is None:
            print(f'  skip (could not parse 16-thread QPS)')
            results.append({'method': m, 'config': None,
                            'single_qps': None, 'real_qps_16t': None,
                            'scaling': None, 'note': 'parse failed'})
            continue
        single = single_thread_qps(m, a['ds'], a['sc'], cfg_master)
        if single is None:
            print(f'  WARN: no single-thread data for cfg={cfg_master}')
            results.append({'method': m, 'config': cfg_master,
                            'single_qps': None, 'real_qps_16t': real_qps_16t,
                            'scaling': None, 'note': 'no single-thread lookup'})
            continue
        scale = real_qps_16t / single
        print(f'  single-thread QPS: {single:.1f}')
        print(f'  16-thread QPS:     {real_qps_16t:.1f}')
        print(f'  scaling factor:    {scale:.2f}x  (ideal=16, expect 10-15)')
        results.append({'method': m, 'config': cfg_master,
                        'single_qps': single, 'real_qps_16t': real_qps_16t,
                        'scaling': scale, 'note': ''})

    print('\n\n=== Final scaling factors ===')
    print(f'{"method":<8} {"single":>10} {"real_16t":>10} {"scaling":>10}')
    def _fmt(v, w=10, p=1, suffix=''):
        if v is None: return ' '*(w-3) + 'N/A' + suffix
        return f'{v:>{w}.{p}f}' + suffix
    for r in results:
        print(f'{r["method"]:<8} {_fmt(r["single_qps"])} '
              f'{_fmt(r["real_qps_16t"])} {_fmt(r["scaling"], p=2, suffix="x")}')
    # Suffix output by methods so 4 parallel runs don't clobber each other
    suffix = '_' + '_'.join(sorted(set(r['method'] for r in results))) if args.methods else ''
    out_path = REPO / f'analysis/ml_router/scaling_factors_audited{suffix}.csv'
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
