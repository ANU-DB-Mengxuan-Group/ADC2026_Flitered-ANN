#!/usr/bin/env python3
"""
V2 路由验证评估 — 增量式

在任意方法子集完成后即可运行，逐步添加更多方法的结果。
输出：方法对比表 + 路由预测 vs Oracle 对比 + 可选图表。

用法:
  cd ~/benchmarks/discrete
  python analysis/evaluate_v2_routing.py                 # 读取所有可用结果
  python analysis/evaluate_v2_routing.py --plot           # 生成对比图
  python analysis/evaluate_v2_routing.py --plot --output analysis/v2_routing_eval.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# ── V2 数据集特征 ─────────────────────────────────────────────────────────
# label_cardinality 和 LID_mean 来自数据集生成配置
# yahoo800k/dbpedia560k 的 LID 为估计值（768d 嵌入空间典型值 20-40）
# 精确值需要跑 tools/dataset_metrics.py，但路由决策不受影响（阈值 100）

V2_DATASETS = {
    "synth_192d":    {"N": 800000, "dim": 192, "label_cardinality": 200,  "LID_mean": 25},
    "synth_512d":    {"N": 800000, "dim": 512, "label_cardinality": 30,   "LID_mean": 35},
    "synth_768d_hc": {"N": 800000, "dim": 768, "label_cardinality": 1000, "LID_mean": 25},
    "yahoo800k":     {"N": 800000, "dim": 768, "label_cardinality": 10,   "LID_mean": 30},
    "dbpedia560k":   {"N": 560000, "dim": 768, "label_cardinality": 14,   "LID_mean": 30},
}

SCENARIOS = ["and", "or", "equal"]

# 自动检测运行环境：集群 vs 本地开发
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_script_dir)  # analysis/ 的上级目录
if os.path.isdir(os.path.join(_repo_dir, "SIEVE")):
    BASE_DIR = _repo_dir  # 本地：从仓库根目录
else:
    BASE_DIR = os.path.expanduser("~/benchmarks/discrete")

# ── 规则路由器（与 train_ml_router_v2.py 保持一致）─────────────────────────

def rule_router(scenario: str, label_cardinality: int, LID_mean: float) -> str:
    if scenario == "equal":
        return "UNG"
    elif scenario == "and":
        if LID_mean > 100:
            return "UNG"
        elif label_cardinality < 100:
            return "UNG"
        else:
            return "SIEVE"
    elif scenario == "or":
        if LID_mean > 100:
            return "UNG"
        else:
            return "Post-filter"
    return "UNG"


# ── 结果提取：每个方法一个函数 ──────────────────────────────────────────────

def extract_sieve(dataset: str) -> dict[str, float]:
    """从 SIEVE 单独 CSV 提取 (scenario → best recall)"""
    results = {}
    scenario_map = {"original_and": "and", "original_or": "or", "original_eq": "equal"}
    for sieve_sc, sc in scenario_map.items():
        pattern = f"SIEVE/results/sieve_{dataset}_{sieve_sc}_*.csv"
        import glob
        files = glob.glob(os.path.join(BASE_DIR, pattern))
        if not files:
            continue
        best_recall = 0.0
        for fpath in files:
            try:
                with open(fpath) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = float(row.get("recall_at_10", 0))
                        if r > best_recall:
                            best_recall = r
            except Exception:
                pass
        if best_recall > 0:
            results[sc] = best_recall
    return results


def extract_from_summary(summary_path: str, dataset: str,
                         scenario_col: str = "scenario",
                         recall_col: str = "recall@10",
                         scenario_map: dict | None = None) -> dict[str, float]:
    """从 summary.csv 提取 (scenario → best recall)"""
    results = {}
    if not os.path.exists(summary_path):
        return results
    try:
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row.get(scenario_col, "").strip()
                if scenario_map:
                    sc = scenario_map.get(sc, sc)
                r = float(row.get(recall_col, 0))
                if sc in SCENARIOS and r > results.get(sc, 0):
                    results[sc] = r
    except Exception as e:
        print(f"  Warning: failed to read {summary_path}: {e}")
    return results


def extract_ung(dataset: str) -> dict[str, float]:
    """UNG V2 数据在 results_v2_grid/summary.csv (合并) 而不是 results_original/.
    后者是 4/2 早期实验, recall 全是 0.05-0.07 错值."""
    # 先试 v2_grid (汇总 summary, recall 字段名为 'recall', scenario 用 UNG 术语)
    grid_summary = os.path.join(BASE_DIR, "UNG-dev/results_v2_grid/summary.csv")
    if os.path.exists(grid_summary):
        results = {}
        try:
            with open(grid_summary) as f:
                reader = csv.DictReader(f)
                ung_to_thesis = {"containment": "and", "overlap": "or", "equality": "equal"}
                for row in reader:
                    if row.get("dataset", "").strip() != dataset:
                        continue
                    sc = ung_to_thesis.get(row.get("scenario", "").strip(), row.get("scenario", "").strip())
                    if sc not in SCENARIOS:
                        continue
                    raw = row.get("recall", "").strip()
                    if not raw:
                        continue
                    try:
                        r = float(raw)
                    except ValueError:
                        continue
                    if r > results.get(sc, 0):
                        results[sc] = r
        except Exception as e:
            print(f"  Warning: failed to read {grid_summary}: {e}")
        if results:
            return results
    # fallback: 老路径 (含错值, 但保留兜底)
    summary = os.path.join(BASE_DIR, f"UNG-dev/results_original/{dataset}/summary.csv")
    return extract_from_summary(summary, dataset, recall_col="recall@10",
                                scenario_map={"containment": "and", "overlap": "or", "equality": "equal"})


def extract_postfilter(dataset: str) -> dict[str, float]:
    summary = os.path.join(BASE_DIR, f"faiss/results_postfilter/{dataset}/summary.csv")
    return extract_from_summary(summary, dataset, recall_col="recall@10")


def extract_acorn(dataset: str) -> dict[str, float]:
    summary = os.path.join(BASE_DIR,
        f"ACORN/data/param_search_{dataset}/results/{dataset}/summary.csv")
    return extract_from_summary(summary, dataset, recall_col="recall@10")


def extract_fv(dataset: str) -> dict[str, float]:
    summary = os.path.join(BASE_DIR, f"DiskANN/data_original/results/{dataset}/summary.csv")
    return extract_from_summary(summary, dataset,
        scenario_map={"containment": "and", "overlap": "or", "equality": "equal"})


def extract_sv(dataset: str) -> dict[str, float]:
    """StitchedVamana: equality only"""
    summary = os.path.join(BASE_DIR, f"DiskANN/data_stitched_eq/results/{dataset}/summary.csv")
    results = {}
    if not os.path.exists(summary):
        return results
    try:
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = float(row.get("recall@10", 0))
                if r > results.get("equal", 0):
                    results["equal"] = r
    except Exception:
        pass
    return results


def extract_caps(dataset: str) -> dict[str, float]:
    """CAPS: equality only"""
    summary = os.path.join(BASE_DIR, f"CAPS/data/results/{dataset}/summary.csv")
    results = {}
    if not os.path.exists(summary):
        return results
    try:
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = float(row.get("recall@10", 0))
                if r > results.get("equal", 0):
                    results["equal"] = r
    except Exception:
        pass
    return results


def extract_nhq(dataset: str) -> dict[str, float]:
    """NHQ: equality only"""
    summary = os.path.join(BASE_DIR, f"NHQ/data/results/{dataset}/summary.csv")
    results = {}
    if not os.path.exists(summary):
        return results
    try:
        with open(summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = float(row.get("recall@10", 0))
                if r > results.get("equal", 0):
                    results["equal"] = r
    except Exception:
        pass
    return results


# ── 方法注册表 ───────────────────────────────────────────────────────────

METHOD_EXTRACTORS = {
    "UNG":         extract_ung,
    "Post-filter": extract_postfilter,
    "SIEVE":       extract_sieve,
    "ACORN":       extract_acorn,
    "FV":          extract_fv,
    "SV":          extract_sv,
    "CAPS":        extract_caps,
    "NHQ":         extract_nhq,
}

# 路由候选方法（规则路由器只从这三个里选）
ROUTING_CANDIDATES = {"UNG", "Post-filter", "SIEVE"}


# ── 主逻辑 ──────────────────────────────────────────────────────────────

def collect_all_results() -> dict:
    """收集所有可用的 V2 方法结果。
    返回: {(dataset, scenario): {method: recall}}
    """
    all_results = {}

    for ds in V2_DATASETS:
        for sc in SCENARIOS:
            all_results[(ds, sc)] = {}

    for method, extractor in METHOD_EXTRACTORS.items():
        for ds in V2_DATASETS:
            try:
                method_results = extractor(ds)
            except Exception as e:
                print(f"  Warning: {method}/{ds}: {e}")
                continue
            for sc, recall in method_results.items():
                if (ds, sc) in all_results:
                    all_results[(ds, sc)][method] = recall

    return all_results


def evaluate_routing(all_results: dict) -> list[dict]:
    """评估路由决策。"""
    rows = []

    for ds in V2_DATASETS:
        features = V2_DATASETS[ds]
        for sc in SCENARIOS:
            cell = all_results.get((ds, sc), {})
            if not cell:
                continue

            # 路由预测
            predicted = rule_router(sc, features["label_cardinality"], features["LID_mean"])

            # Oracle（所有方法中最佳）
            oracle_method = max(cell, key=cell.get)
            oracle_recall = cell[oracle_method]

            # 路由候选中的 Oracle（仅 UNG/Post-filter/SIEVE）
            candidate_cell = {m: r for m, r in cell.items() if m in ROUTING_CANDIDATES}
            if candidate_cell:
                candidate_oracle_method = max(candidate_cell, key=candidate_cell.get)
                candidate_oracle_recall = candidate_cell[candidate_oracle_method]
            else:
                candidate_oracle_method = "N/A"
                candidate_oracle_recall = 0.0

            # 路由实际达到的 recall
            routed_recall = cell.get(predicted, None)

            row = {
                "dataset": ds,
                "scenario": sc,
                "card": features["label_cardinality"],
                "LID": features["LID_mean"],
                "rule_prediction": predicted,
                "routed_recall": routed_recall,
                "oracle_method": oracle_method,
                "oracle_recall": oracle_recall,
                "candidate_oracle": candidate_oracle_method,
                "candidate_oracle_recall": candidate_oracle_recall,
                "match_oracle": predicted == oracle_method,
                "match_candidate": predicted == candidate_oracle_method,
                "methods": cell,
            }
            rows.append(row)

    return rows


def print_results(rows: list[dict]):
    """打印格式化的结果表。"""
    # 收集所有出现过的方法
    all_methods = set()
    for row in rows:
        all_methods.update(row["methods"].keys())
    all_methods = sorted(all_methods)

    available_count = len(all_methods)
    cells_with_data = len(rows)
    total_cells = len(V2_DATASETS) * len(SCENARIOS)

    print(f"\n{'='*80}")
    print(f"V2 路由验证评估")
    print(f"{'='*80}")
    print(f"可用方法: {', '.join(all_methods)} ({available_count} 个)")
    print(f"数据覆盖: {cells_with_data}/{total_cells} cells")

    # ─── 1. 方法 recall 对比表 ───
    print(f"\n{'─'*80}")
    print("方法 Recall 对比（每个 cell 的最佳 recall）")
    print(f"{'─'*80}")

    header = f"{'Dataset':<16} {'Scenario':<8}"
    for m in all_methods:
        header += f" {m:>12}"
    header += f" {'Oracle':>8}"
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['dataset']:<16} {row['scenario']:<8}"
        for m in all_methods:
            r = row["methods"].get(m)
            if r is not None:
                line += f" {r:>12.4f}"
            else:
                line += f" {'—':>12}"
        line += f" {row['oracle_recall']:>8.4f}"
        print(line)

    # ─── 2. 路由评估表 ───
    print(f"\n{'─'*80}")
    print("规则路由评估")
    print(f"{'─'*80}")
    print(f"{'Dataset':<16} {'Scenario':<8} {'Card':>5} {'LID':>4}  "
          f"{'Predicted':<13} {'Recall':>7}  {'Oracle':<13} {'Recall':>7}  {'Match':>5}")
    print("-" * 95)

    match_count = 0
    match_candidate_count = 0
    routed_recalls = []
    oracle_recalls = []
    candidate_oracle_recalls = []

    for row in rows:
        predicted = row["rule_prediction"]
        routed_r = row["routed_recall"]
        oracle_m = row["oracle_method"]
        oracle_r = row["oracle_recall"]
        match = row["match_oracle"]
        match_c = row["match_candidate"]

        routed_str = f"{routed_r:.4f}" if routed_r is not None else "N/A"
        match_str = "Y" if match else ("~" if match_c else "X")

        print(f"{row['dataset']:<16} {row['scenario']:<8} {row['card']:>5} {row['LID']:>4}  "
              f"{predicted:<13} {routed_str:>7}  {oracle_m:<13} {oracle_r:>7.4f}  {match_str:>5}")

        if match:
            match_count += 1
        if match_c:
            match_candidate_count += 1
        if routed_r is not None:
            routed_recalls.append(routed_r)
        oracle_recalls.append(oracle_r)
        candidate_oracle_recalls.append(row["candidate_oracle_recall"])

    # ─── 3. 汇总统计 ───
    print(f"\n{'─'*80}")
    print("汇总")
    print(f"{'─'*80}")

    n = len(rows)
    if n > 0:
        print(f"  Oracle match (全方法):     {match_count}/{n} ({match_count/n*100:.1f}%)")
        print(f"  Oracle match (3 候选):     {match_candidate_count}/{n} ({match_candidate_count/n*100:.1f}%)")
        print(f"  Oracle avg recall:         {np.mean(oracle_recalls):.4f}")
        if routed_recalls:
            print(f"  Route avg recall:          {np.mean(routed_recalls):.4f}")
            gap = np.mean(oracle_recalls) - np.mean(routed_recalls)
            print(f"  Gap (Oracle - Route):      {gap:+.4f}")

        # 每个方法的平均 recall
        print(f"\n  各方法平均 recall:")
        for m in all_methods:
            vals = [row["methods"].get(m) for row in rows if m in row["methods"]]
            if vals:
                print(f"    {m:<13}: {np.mean(vals):.4f}  ({len(vals)}/{n} cells)")

    # ─── 4. 缺失提示 ───
    missing_methods = set()
    for row in rows:
        predicted = row["rule_prediction"]
        if predicted not in row["methods"]:
            missing_methods.add(predicted)

    if missing_methods:
        print(f"\n  ⚠️  路由预测的方法尚无结果: {', '.join(sorted(missing_methods))}")
        print(f"     跑完这些方法后重新运行本脚本即可更新评估")


def plot_results(rows: list[dict], output: str | None = None):
    """生成对比柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return

    # 收集路由候选的数据
    datasets_scenarios = [(r["dataset"], r["scenario"]) for r in rows]
    labels = [f"{ds}\n{sc}" for ds, sc in datasets_scenarios]

    oracle_vals = [r["oracle_recall"] for r in rows]
    routed_vals = [r["routed_recall"] if r["routed_recall"] is not None else 0 for r in rows]

    # 每个方法的值
    all_methods_set = set()
    for r in rows:
        all_methods_set.update(r["methods"].keys())
    all_methods = sorted(all_methods_set)

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.2), 6))

    x = np.arange(len(rows))
    width = 0.8 / (len(all_methods) + 2)  # +2 for Oracle and Route

    # Plot each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_methods) + 2))
    for i, m in enumerate(all_methods):
        vals = [r["methods"].get(m, 0) for r in rows]
        ax.bar(x + i * width, vals, width, label=m, color=colors[i], alpha=0.8)

    # Oracle and Route
    ax.bar(x + len(all_methods) * width, oracle_vals, width,
           label="Oracle", color="gold", edgecolor="black", linewidth=1)
    ax.bar(x + (len(all_methods) + 1) * width, routed_vals, width,
           label="Rule Router", color="red", edgecolor="black", linewidth=1, alpha=0.7)

    ax.set_ylabel("Recall@10")
    ax.set_title("V2 Validation: Method Comparison & Routing Evaluation")
    ax.set_xticks(x + (len(all_methods) / 2) * width)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output or "analysis/v2_routing_eval.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


def save_csv(rows: list[dict], output: str | None = None):
    """保存结果到 CSV。"""
    out = output or os.path.join(BASE_DIR, "analysis/v2_routing_eval.csv")
    all_methods = set()
    for r in rows:
        all_methods.update(r["methods"].keys())
    all_methods = sorted(all_methods)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["dataset", "scenario", "card", "LID", "rule_prediction", "routed_recall",
                  "oracle_method", "oracle_recall", "match_oracle", "match_candidate"]
        header += [f"recall_{m}" for m in all_methods]
        writer.writerow(header)

        for row in rows:
            vals = [row["dataset"], row["scenario"], row["card"], row["LID"],
                    row["rule_prediction"], row["routed_recall"] or "",
                    row["oracle_method"], row["oracle_recall"],
                    row["match_oracle"], row["match_candidate"]]
            vals += [row["methods"].get(m, "") for m in all_methods]
            writer.writerow(vals)

    print(f"CSV 已保存: {out}")


# ── 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V2 路由验证评估")
    parser.add_argument("--plot", action="store_true", help="生成对比图")
    parser.add_argument("--output", type=str, help="图表输出路径")
    parser.add_argument("--csv", type=str, help="CSV 输出路径")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    print("收集 V2 方法结果...")
    all_results = collect_all_results()

    rows = evaluate_routing(all_results)
    if not rows:
        print("没有找到任何 V2 方法结果。请先运行实验。")
        sys.exit(1)

    print_results(rows)
    save_csv(rows, args.csv)

    if args.plot:
        plot_results(rows, args.output)


if __name__ == "__main__":
    main()
