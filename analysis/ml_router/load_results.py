#!/usr/bin/env python3
"""
Load experiment results from all FANNS methods.
"""
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

def load_acorn_gamma(dataset: str) -> pd.DataFrame:
    """Load ACORN-gamma results for a dataset."""
    # Try multiple possible paths
    paths = [
        BASE_DIR / f"ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv",
        BASE_DIR / f"ACORN/data/param_search_{dataset}/summary.csv",
    ]

    path = None
    for p in paths:
        if p.exists():
            path = p
            break

    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['method'] = 'ACORN-γ'
    df['dataset'] = dataset
    # Rename columns for consistency
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_acorn_1(dataset: str) -> pd.DataFrame:
    """Load ACORN-1 (gamma=1) results for a dataset."""
    path = BASE_DIR / f"ACORN/data/param_search_{dataset}_gamma1/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['method'] = 'ACORN-1'
    df['dataset'] = dataset
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_caps(dataset: str) -> pd.DataFrame:
    """Load CAPS results for a dataset from per-nb result files.

    Each nb_*_results.csv contains the full nprobe sweep (15 points).
    The summary.csv only has one point per nb and is incomplete.
    """
    result_dir = BASE_DIR / f"CAPS/data/results/{dataset}"
    if not result_dir.exists():
        return pd.DataFrame()

    all_rows = []
    for csv_file in sorted(result_dir.glob("nb_*_results.csv")):
        # Extract nb from filename like "nb_128_results.csv"
        nb = int(csv_file.stem.split('_')[1])
        df = pd.read_csv(csv_file)
        df['nb'] = nb
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'CAPS'
    df['dataset'] = dataset
    df['scenario'] = 'equal'  # CAPS only supports FIXED-EQ
    df = df.rename(columns={'Recall': 'recall'})
    return df

def load_nhq(dataset: str) -> pd.DataFrame:
    """Load NHQ results for a dataset."""
    path = BASE_DIR / f"NHQ/data_synthetic/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['method'] = 'NHQ'
    df['dataset'] = dataset
    df['scenario'] = 'equal'  # NHQ only supports FIXED-EQ
    df = df.rename(columns={'Recall@10': 'recall'})
    return df

def load_ung(dataset: str) -> pd.DataFrame:
    """Load UNG results for a dataset from BOTH per-dataset summary.csv
    (V1 / pre-grid runs, recall@10 in 0-100) AND the V2 grid file
    (results_v2_grid/summary.csv, recall in 0-1)."""
    scenario_map = {'containment': 'and', 'overlap': 'or', 'equality': 'equal'}
    frames = []

    # 1) V1 per-dataset summary.csv (recall@10 0-100)
    path1 = BASE_DIR / f"UNG-dev/results_original/{dataset}/summary.csv"
    if path1.exists():
        df1 = pd.read_csv(path1)
        df1 = df1[df1['status'] == 'success']
        if not df1.empty and 'recall@10' in df1.columns:
            df1 = df1.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
            df1['recall'] = df1['recall'] / 100.0
            frames.append(df1[['dataset', 'scenario', 'recall', 'QPS']])

    # 2) V2 grid summary.csv (recall in 0-1, single file, filter by dataset)
    path2 = BASE_DIR / "UNG-dev/results_v2_grid/summary.csv"
    if path2.exists():
        df2 = pd.read_csv(path2, on_bad_lines='skip')
        df2 = df2[(df2['status'] == 'success') & (df2['dataset'] == dataset)]
        if not df2.empty and 'recall' in df2.columns:
            df2 = df2.rename(columns={'qps': 'QPS'})
            frames.append(df2[['dataset', 'scenario', 'recall', 'QPS']])

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df['method'] = 'UNG'
    df['scenario'] = df['scenario'].map(scenario_map)
    df = df.dropna(subset=['scenario', 'recall', 'QPS'])
    return df

def load_prefilter(dataset: str) -> pd.DataFrame:
    """Load Pre-filter brute-force results for a dataset."""
    scenarios = ['and', 'or', 'equal']
    all_rows = []

    for scenario in scenarios:
        path = BASE_DIR / f"faiss/results_prefilter/{dataset}/{scenario}/prefilter_bruteforce_result.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['scenario'] = scenario
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'Pre-filter'
    df['dataset'] = dataset
    df = df.rename(columns={'Recall': 'recall'})
    return df

def load_postfilter(dataset: str) -> pd.DataFrame:
    """Load Post-filter HNSW results for a dataset."""
    scenarios = ['and', 'or', 'equal']
    all_rows = []

    for scenario in scenarios:
        scenario_dir = BASE_DIR / f"faiss/results_postfilter/{dataset}/{scenario}"
        if not scenario_dir.exists():
            continue

        for csv_file in scenario_dir.glob("*_result.csv"):
            df = pd.read_csv(csv_file)
            # Extract M and efc from filename like "M=32_efc=100_result.csv"
            fname = csv_file.stem
            parts = fname.replace('_result', '').split('_')
            params = {}
            for p in parts:
                if '=' in p:
                    k, v = p.split('=')
                    params[k] = int(v)

            df['M'] = params.get('M', 0)
            df['efc'] = params.get('efc', 0)
            df['scenario'] = scenario
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'Post-filter'
    df['dataset'] = dataset
    df = df.rename(columns={'Recall': 'recall'})
    return df

def load_filtered_vamana_original(dataset: str) -> pd.DataFrame:
    """Load FilteredVamana original-label results for a dataset."""
    path = BASE_DIR / f"DiskANN/data_original/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    df['method'] = 'FilteredVamana'
    df['dataset'] = dataset
    scenario_map = {'containment': 'and', 'overlap': 'or', 'equality': 'equal'}
    df['scenario'] = df['scenario'].map(scenario_map)
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_stitched_vamana_original(dataset: str) -> pd.DataFrame:
    """Load StitchedVamana original-label results for a dataset."""
    path = BASE_DIR / f"DiskANN/data_stitched_original/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    df['method'] = 'StitchedVamana'
    df['dataset'] = dataset
    scenario_map = {'containment': 'and', 'overlap': 'or', 'equality': 'equal'}
    df['scenario'] = df['scenario'].map(scenario_map)
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_ung_fixed_eq(dataset: str) -> pd.DataFrame:
    """Load UNG Fixed-EQ results for a dataset."""
    path = BASE_DIR / f"UNG-dev/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    df['method'] = 'UNG'
    df['dataset'] = dataset
    df['scenario'] = 'equal'
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    df['recall'] = df['recall'] / 100.0
    return df

def load_filtered_vamana(dataset: str) -> pd.DataFrame:
    """Load FilteredVamana Fixed-EQ results for a dataset."""
    path = BASE_DIR / f"DiskANN/data_fixed_eq/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    df['method'] = 'FilteredVamana'
    df['dataset'] = dataset
    df['scenario'] = 'equal'
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_stitched_vamana(dataset: str) -> pd.DataFrame:
    """Load StitchedVamana Fixed-EQ results for a dataset."""
    path = BASE_DIR / f"DiskANN/data_stitched_eq/results/{dataset}/summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'success']
    df['method'] = 'StitchedVamana'
    df['dataset'] = dataset
    df['scenario'] = 'equal'
    df = df.rename(columns={'recall@10': 'recall', 'qps': 'QPS'})
    return df

def load_sieve(dataset: str) -> pd.DataFrame:
    """Load SIEVE original-label results for a dataset."""
    result_dir = BASE_DIR / "SIEVE/results"
    if not result_dir.exists():
        return pd.DataFrame()

    all_rows = []
    # Load all CSV files for this dataset with original_* scenarios
    for scenario_prefix in ['original_and', 'original_or', 'original_eq']:
        for csv_file in result_dir.glob(f"sieve_{dataset}_{scenario_prefix}_*.csv"):
            df = pd.read_csv(csv_file)
            all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'SIEVE'
    # Map scenario names
    scenario_map = {'original_and': 'and', 'original_or': 'or', 'original_eq': 'equal'}
    df['scenario'] = df['scenario'].map(scenario_map)
    df = df.rename(columns={'recall_at_10': 'recall', 'qps': 'QPS'})
    return df

def load_sieve_fixed_eq(dataset: str) -> pd.DataFrame:
    """Load SIEVE Fixed-EQ results for a dataset."""
    result_dir = BASE_DIR / "SIEVE/results"
    if not result_dir.exists():
        return pd.DataFrame()

    all_rows = []
    for csv_file in result_dir.glob(f"sieve_{dataset}_fixed_eq_*.csv"):
        df = pd.read_csv(csv_file)
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'SIEVE'
    df['scenario'] = 'equal'
    df = df.rename(columns={'recall_at_10': 'recall', 'qps': 'QPS'})
    return df

def load_prefilter_fixed_eq(dataset: str) -> pd.DataFrame:
    """Load Pre-filter Fixed-EQ results for a dataset."""
    path = BASE_DIR / f"faiss/results_prefilter/{dataset}/fixed_eq/prefilter_bruteforce_result.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['method'] = 'Pre-filter'
    df['dataset'] = dataset
    df['scenario'] = 'equal'
    df = df.rename(columns={'Recall': 'recall'})
    return df

def load_postfilter_fixed_eq(dataset: str) -> pd.DataFrame:
    """Load Post-filter HNSW Fixed-EQ results for a dataset."""
    scenario_dir = BASE_DIR / f"faiss/results_postfilter/{dataset}/fixed_eq"
    if not scenario_dir.exists():
        return pd.DataFrame()
    all_rows = []
    for csv_file in scenario_dir.glob("*_result.csv"):
        df = pd.read_csv(csv_file)
        fname = csv_file.stem
        parts = fname.replace('_result', '').split('_')
        params = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=')
                params[k] = int(v)
        df['M'] = params.get('M', 0)
        df['efc'] = params.get('efc', 0)
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True)
    df['method'] = 'Post-filter'
    df['dataset'] = dataset
    df['scenario'] = 'equal'
    df = df.rename(columns={'Recall': 'recall'})
    return df

def load_all_results() -> pd.DataFrame:
    """Load all results from all methods and datasets."""
    datasets = ['arxiv', 'yfcc', 'LAION1M', 'tripclick', 'ytb_audio', 'ytb_video']

    all_dfs = []

    for dataset in datasets:
        # Load each method
        for loader in [load_acorn_gamma, load_acorn_1, load_caps, load_nhq, load_ung,
                       load_filtered_vamana_original, load_stitched_vamana_original,
                       load_prefilter, load_postfilter, load_sieve]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def load_fixed_eq_results() -> pd.DataFrame:
    """Load results for Fixed-Length Equality comparison (arxiv, yfcc)."""
    datasets = ['arxiv', 'yfcc']

    all_dfs = []

    for dataset in datasets:
        # ACORN methods - filter to 'equal' scenario
        for loader in [load_acorn_gamma, load_acorn_1]:
            df = loader(dataset)
            if not df.empty:
                df = df[df['scenario'] == 'equal']
                all_dfs.append(df)

        # CAPS and NHQ (only support FIXED-EQ)
        for loader in [load_caps, load_nhq]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)

        # UNG, FilteredVamana, StitchedVamana, SIEVE
        for loader in [load_ung_fixed_eq, load_filtered_vamana, load_stitched_vamana, load_sieve_fixed_eq]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)

        # Pre-filter, Post-filter (fixed-eq)
        for loader in [load_prefilter_fixed_eq, load_postfilter_fixed_eq]:
            df = loader(dataset)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def get_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Get Pareto-optimal points (maximize recall, maximize QPS)."""
    if df.empty:
        return df

    # Sort by recall descending
    df = df.sort_values('recall', ascending=False).reset_index(drop=True)

    pareto = []
    max_qps = -1

    for _, row in df.iterrows():
        if row['QPS'] > max_qps:
            pareto.append(row)
            max_qps = row['QPS']

    return pd.DataFrame(pareto)

if __name__ == "__main__":
    # Test loading
    print("Loading all results...")
    df = load_all_results()
    print(f"Total rows: {len(df)}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")

    print("\nLoading Fixed-EQ results...")
    df_eq = load_fixed_eq_results()
    print(f"Fixed-EQ rows: {len(df_eq)}")

    # Summary by method and dataset
    print("\nSummary:")
    for method in df_eq['method'].unique():
        method_df = df_eq[df_eq['method'] == method]
        datasets = method_df['dataset'].unique()
        print(f"  {method}: {len(method_df)} rows, datasets: {list(datasets)}")
