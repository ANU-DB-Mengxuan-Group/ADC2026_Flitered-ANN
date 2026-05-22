#!/usr/bin/env python3
"""
绘制ACORN的QPS-Recall曲线图

用法:
    python plot_qps_recall.py <results_dir> [--scenario equal|or|and|all] [--output output.png]

示例:
    python plot_qps_recall.py ../data/param_search_arxiv/results/arxiv --scenario equal
    python plot_qps_recall.py ../data/param_search_arxiv/results/arxiv --scenario all --output qps_recall.pdf
"""

import os
import sys
import glob
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色和标记样式
COLORS = plt.cm.tab10.colors
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


def parse_filename(filename):
    """从文件名解析参数 M, M_beta, gamma"""
    pattern = r'M=(\d+)_M_beta=(\d+)_gamma=(\d+)_result\.csv'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def load_result_csv(filepath):
    """加载单个结果CSV文件"""
    try:
        df = pd.read_csv(filepath)
        # 确保列名正确
        expected_cols = ['L', 'Cmps', 'QPS', 'Recall', 'QPS_no_filter']
        if not all(col in df.columns for col in expected_cols):
            print(f"警告: {filepath} 缺少必要的列")
            return None
        return df
    except Exception as e:
        print(f"无法读取 {filepath}: {e}")
        return None


def load_scenario_data(results_dir, scenario):
    """加载指定场景的所有结果数据"""
    scenario_dir = os.path.join(results_dir, scenario)
    if not os.path.exists(scenario_dir):
        print(f"场景目录不存在: {scenario_dir}")
        return {}

    data = {}
    csv_files = glob.glob(os.path.join(scenario_dir, '*_result.csv'))

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        M, M_beta, gamma = parse_filename(filename)
        if M is None:
            continue

        df = load_result_csv(filepath)
        if df is not None:
            key = (M, M_beta, gamma)
            data[key] = df

    return data


def plot_single_scenario(ax, data, scenario_name, show_legend=True):
    """在单个子图上绘制一个场景的QPS-Recall曲线"""
    if not data:
        ax.text(0.5, 0.5, f'No data for {scenario_name}',
                ha='center', va='center', transform=ax.transAxes)
        return

    # 按参数排序
    sorted_keys = sorted(data.keys())

    for idx, key in enumerate(sorted_keys):
        M, M_beta, gamma = key
        df = data[key]

        # 按Recall排序
        df_sorted = df.sort_values('Recall')

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = f'M={M}, Mβ={M_beta}, γ={gamma}'

        ax.plot(df_sorted['Recall'], df_sorted['QPS'],
                color=color, marker=marker, markersize=4,
                linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.set_title(f'{scenario_name.upper()} Scenario', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)

    if show_legend and len(data) <= 12:
        ax.legend(fontsize=7, loc='best', ncol=2)


def plot_qps_recall(results_dir, scenarios, output_path=None, title=None):
    """绘制QPS-Recall曲线"""

    if scenarios == ['all']:
        scenarios = ['equal', 'or', 'and']

    n_scenarios = len(scenarios)

    if n_scenarios == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 7))
        axes = [axes]
    elif n_scenarios == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, scenario in zip(axes, scenarios):
        data = load_scenario_data(results_dir, scenario)
        print(f"场景 {scenario}: 加载了 {len(data)} 个参数组合")
        plot_single_scenario(ax, data, scenario)

    # 设置标题
    dataset_name = os.path.basename(results_dir)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'ACORN QPS-Recall Curves ({dataset_name})', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()


def plot_pareto_frontier(results_dir, scenarios, output_path=None):
    """绘制Pareto前沿曲线（每个场景的最优参数组合）"""

    if scenarios == ['all']:
        scenarios = ['equal', 'or', 'and']

    fig, ax = plt.subplots(figsize=(10, 7))

    for scenario_idx, scenario in enumerate(scenarios):
        data = load_scenario_data(results_dir, scenario)
        if not data:
            continue

        # 收集所有数据点
        all_points = []
        for key, df in data.items():
            M, M_beta, gamma = key
            for _, row in df.iterrows():
                all_points.append({
                    'recall': row['Recall'],
                    'qps': row['QPS'],
                    'M': M,
                    'M_beta': M_beta,
                    'gamma': gamma,
                    'L': row['L']
                })

        if not all_points:
            continue

        # 计算Pareto前沿
        points_df = pd.DataFrame(all_points)
        points_df = points_df.sort_values('recall')

        # 找Pareto最优点（recall越高且qps越高越好）
        pareto_points = []
        max_qps = -1
        for recall in sorted(points_df['recall'].unique()):
            subset = points_df[points_df['recall'] == recall]
            best_qps = subset['qps'].max()
            if best_qps > max_qps:
                max_qps = best_qps
                best_row = subset[subset['qps'] == best_qps].iloc[0]
                pareto_points.append(best_row)

        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            color = COLORS[scenario_idx % len(COLORS)]
            ax.plot(pareto_df['recall'], pareto_df['qps'],
                    color=color, marker='o', markersize=5,
                    linewidth=2, label=f'{scenario.upper()} (Pareto)', alpha=0.9)

    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('QPS', fontsize=12)
    ax.set_title('ACORN Pareto Frontiers by Scenario', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Pareto图表已保存到: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='绘制ACORN QPS-Recall曲线')
    parser.add_argument('results_dir', help='结果目录路径 (如 ../data/param_search_arxiv/results/arxiv)')
    parser.add_argument('--scenario', '-s', default='all',
                        choices=['equal', 'or', 'and', 'all'],
                        help='要绘制的场景 (默认: all)')
    parser.add_argument('--output', '-o', help='输出文件路径 (支持 .png, .pdf, .svg)')
    parser.add_argument('--pareto', '-p', action='store_true',
                        help='只绘制Pareto前沿曲线')
    parser.add_argument('--title', '-t', help='自定义图表标题')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"错误: 目录不存在 {args.results_dir}")
        sys.exit(1)

    scenarios = [args.scenario] if args.scenario != 'all' else ['all']

    if args.pareto:
        plot_pareto_frontier(args.results_dir, scenarios, args.output)
    else:
        plot_qps_recall(args.results_dir, scenarios, args.output, args.title)


if __name__ == '__main__':
    main()
