#!/usr/bin/env python3
"""
V2 数据集 recall-QPS 曲线图 (5 数据集 × 3 场景 = 15 subplot)

每个 subplot 含 4 方法曲线/点:
  - Post-filter (M=64 efc=400, ef_search 扫描)
  - UNG        (results_v2_grid, M+Lb 取最佳, Ls 扫描)
  - SIEVE      (M+b+h 取最佳, ef_search 扫描)
  - ACORN      (各 M_beta/gamma 配置的 scatter)

输入:
  faiss/results_postfilter/{ds}/{sc}/M=64_efc=400_result.csv
  UNG-dev/results_v2_grid/summary.csv
  SIEVE/results/sieve_{ds}_original_{sc}_*.csv
  ACORN/data/param_search_{ds}/results/{ds}/summary.csv

输出:
  analysis/figures/v2_recall_qps.pdf
  analysis/figures/v2_recall_qps.png

用法:
  cd ~/benchmarks/discrete  (或本地仓库根)
  python analysis/plot_v2_recall_qps.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

V2_DATASETS = ["synth_192d", "synth_512d", "synth_768d_hc", "yahoo800k", "dbpedia560k"]
SCENARIOS = ["and", "or", "equal"]

# 方法名映射 (各方法用各自术语; thesis 统一为 and/or/equal)
SCEN_UNG = {"and": "containment", "or": "overlap", "equal": "equality"}
SCEN_SIEVE = {"and": "original_and", "or": "original_or", "equal": "original_eq"}

METHOD_COLORS = {
    "UNG": "#1f77b4",
    "Post-filter": "#ff7f0e",
    "SIEVE": "#2ca02c",
    "ACORN": "#d62728",
    "FilteredVamana": "#9467bd",
    "Pre-filter": "#8c564b",
    "ML Router": "#000000",
}
METHOD_MARKERS = {
    "UNG": "o",
    "Post-filter": "s",
    "SIEVE": "^",
    "ACORN": "D",
    "FilteredVamana": "v",
    "Pre-filter": "*",
    "ML Router": "X",
}

# 缓存 ML router 数据 (避免每个 subplot 重读)
_ML_ROUTER_CACHE = None


def load_ml_router_all():
    """读 ML router predictions + per-query recall, 算每个 (ds, sc) 的 (qps, recall).

    Algorithm:
      For each query in predictions.csv:
        chosen_method = MLP-Reg_pred (best regression model)
        Look up chosen_method's recall in perquery_recall_v2.csv
      Aggregate per (ds, sc):
        recall = mean of looked-up recalls
        qps = harmonic mean of method's "best ef QPS" weighted by chosen distribution

    Returns: dict[(ds, sc)] = (qps, recall)
    """
    global _ML_ROUTER_CACHE
    if _ML_ROUTER_CACHE is not None:
        return _ML_ROUTER_CACHE

    pred_path = BASE_DIR / "analysis" / "ml_router" / "results" / "v2_on_v2" / "full_train_val_predictions_minimal_reg.csv"
    pq_path = BASE_DIR / "analysis" / "perquery_recall" / "perquery_recall_v2.csv"
    if not pred_path.exists() or not pq_path.exists():
        _ML_ROUTER_CACHE = {}
        return _ML_ROUTER_CACHE

    # 1. 加载 per-query recall: dict[(qid, ds, sc, method)] = recall
    pq_recall = {}
    with open(pq_path) as f:
        for r in csv.DictReader(f):
            try:
                qid = int(r["query_id"])
                key = (qid, r["dataset"], r["scenario"], r["method"])
                pq_recall[key] = float(r["recall_at_10"])
            except (KeyError, ValueError):
                continue

    # 2. 每方法每 (ds, sc) 的 best (recall, QPS) — 用各方法 best ef config 的 QPS 作 chosen QPS
    method_best_qps = {}  # dict[(ds, sc, method)] = qps
    for ds in V2_DATASETS:
        for sc in SCENARIOS:
            for m, loader in [
                ("UNG", load_ung),
                ("Post-filter", load_postfilter),
                ("SIEVE", load_sieve),
            ]:
                pts = loader(ds, sc)
                if pts:
                    # 选 max recall 的 QPS
                    best = max(pts, key=lambda p: p[1])
                    method_best_qps[(ds, sc, m)] = best[0]

    # 3. 每 (ds, sc), 走 predictions, 算 mean recall + harmonic mean QPS
    cell_data = defaultdict(lambda: {"recalls": [], "qps_list": []})
    with open(pred_path) as f:
        for r in csv.DictReader(f):
            try:
                qid = int(r["query_id"])
                ds = r["dataset"]
                sc = r["scenario"]
                chosen = r["MLP-Reg_pred"]
            except KeyError:
                continue
            recall = pq_recall.get((qid, ds, sc, chosen))
            if recall is None:
                continue
            qps = method_best_qps.get((ds, sc, chosen))
            if qps is None or qps <= 0:
                continue
            cell_data[(ds, sc)]["recalls"].append(recall)
            cell_data[(ds, sc)]["qps_list"].append(qps)

    # 4. 算 aggregate
    result = {}
    for (ds, sc), d in cell_data.items():
        n = len(d["recalls"])
        if n == 0:
            continue
        mean_recall = sum(d["recalls"]) / n
        # harmonic mean of QPS (= 1 / mean(1/qps))
        inv_sum = sum(1.0 / q for q in d["qps_list"])
        harmonic_qps = n / inv_sum
        result[(ds, sc)] = (harmonic_qps, mean_recall)

    _ML_ROUTER_CACHE = result
    return result


def load_ml_router(dataset: str, scenario: str):
    """Returns [(qps, recall)] (single-point list) for ML router in this cell."""
    data = load_ml_router_all()
    pt = data.get((dataset, scenario))
    return [pt] if pt else []


# =============================================================================
# ML Router Pareto: 用 recall 阈值 — recall ≥ T, 选 QPS 最高方法
# Oracle-style 实现 (用 actual recall 而不是 predicted), 给 ML router 上界.
# =============================================================================

_PARETO_CACHE = None


def load_pareto_router_all(thresholds=(0.5, 0.7, 0.85, 0.9, 0.95, 0.99)):
    """每 (ds, sc) 算 recall threshold 列表对应的 Pareto 操作点."""
    global _PARETO_CACHE
    cache_key = tuple(thresholds)
    if _PARETO_CACHE is not None and _PARETO_CACHE.get("key") == cache_key:
        return _PARETO_CACHE["data"]

    pq_path = BASE_DIR / "analysis" / "perquery_recall" / "perquery_recall_v2.csv"
    if not pq_path.exists():
        return {}

    # per (qid, ds, sc) → {method: recall}
    per_query = defaultdict(dict)
    with open(pq_path) as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["query_id"]), r["dataset"], r["scenario"])
                per_query[key][r["method"]] = float(r["recall_at_10"])
            except (KeyError, ValueError):
                continue

    # 每方法 best-ef QPS per (ds, sc)
    method_best_qps = {}
    for ds in V2_DATASETS:
        for sc in SCENARIOS:
            for m, loader in [
                ("UNG", load_ung),
                ("Post-filter", load_postfilter),
                ("SIEVE", load_sieve),
            ]:
                pts = loader(ds, sc)
                if pts:
                    best = max(pts, key=lambda p: p[1])
                    method_best_qps[(ds, sc, m)] = best[0]

    # 每 (ds, sc) × 每 threshold T: 路由 + 聚合
    result = defaultdict(list)  # (ds, sc) → list of (qps, recall) one per T
    for (qid, ds, sc), methods in per_query.items():
        pass  # iterate below

    # 重新组织: per (ds, sc), 每 T 走全部 query
    cells = defaultdict(list)  # (ds, sc) → [(qid, methods)]
    for (qid, ds, sc), methods in per_query.items():
        cells[(ds, sc)].append((qid, methods))

    for (ds, sc), queries in cells.items():
        for T in thresholds:
            recalls = []
            qps_list = []
            for qid, m_recalls in queries:
                # 候选方法: recall ≥ T (用 actual)
                candidates = []
                for m, r in m_recalls.items():
                    if r < T:
                        continue
                    qps = method_best_qps.get((ds, sc, m))
                    if qps is None:
                        continue
                    candidates.append((m, r, qps))
                if not candidates:
                    # 阈值过严, 退而求其次 — 取 max recall
                    fallback = max(m_recalls.items(), key=lambda x: x[1])
                    fb_qps = method_best_qps.get((ds, sc, fallback[0]))
                    if fb_qps is None:
                        continue
                    recalls.append(fallback[1])
                    qps_list.append(fb_qps)
                else:
                    # Pareto: highest QPS in candidates
                    chosen = max(candidates, key=lambda x: x[2])
                    recalls.append(chosen[1])
                    qps_list.append(chosen[2])
            if not recalls:
                continue
            mean_recall = sum(recalls) / len(recalls)
            harmonic_qps = len(qps_list) / sum(1.0 / q for q in qps_list)
            result[(ds, sc)].append((harmonic_qps, mean_recall))

    _PARETO_CACHE = {"key": cache_key, "data": dict(result)}
    return _PARETO_CACHE["data"]


def load_pareto_router(dataset: str, scenario: str):
    """List of (qps, recall) — 多 threshold 的 Pareto 操作点, 已按 recall 升序."""
    data = load_pareto_router_all()
    pts = data.get((dataset, scenario), [])
    return sorted(pts, key=lambda p: p[1])


# =============================================================================
# 真实 ML Router Pareto: 用 predicted recall (不是 oracle actual recall)
# 需要 predictions CSV 含 {model}_predrecall_{method} 字段 — 重训后才有.
# =============================================================================

_ML_PARETO_CACHE = None


def load_ml_pareto_all(model="MLP-Reg",
                      thresholds=(0.5, 0.7, 0.85, 0.9, 0.95, 0.99)):
    """每 (ds, sc) × T 用 predicted recall 阈值 + max QPS 路由.

    评价用 actual recall (paper 报告真实表现) + actual QPS.
    需要 predictions CSV 含 {model}_predrecall_{method} 字段.
    """
    global _ML_PARETO_CACHE
    cache_key = (model, tuple(thresholds))
    if _ML_PARETO_CACHE is not None and _ML_PARETO_CACHE.get("key") == cache_key:
        return _ML_PARETO_CACHE["data"]

    pred_path = BASE_DIR / "analysis" / "ml_router" / "results" / "v2_on_v2" / "full_train_val_predictions_minimal_reg.csv"
    pq_path = BASE_DIR / "analysis" / "perquery_recall" / "perquery_recall_v2.csv"
    if not pred_path.exists() or not pq_path.exists():
        return {}

    # actual per-query recall: dict[(qid, ds, sc, method)] = recall
    actual_recall = {}
    with open(pq_path) as f:
        for r in csv.DictReader(f):
            try:
                key = (int(r["query_id"]), r["dataset"], r["scenario"], r["method"])
                actual_recall[key] = float(r["recall_at_10"])
            except (KeyError, ValueError):
                continue

    # 每方法 best-ef QPS per (ds, sc)
    method_best_qps = {}
    for ds in V2_DATASETS:
        for sc in SCENARIOS:
            for m, loader in [("UNG", load_ung), ("Post-filter", load_postfilter), ("SIEVE", load_sieve)]:
                pts = loader(ds, sc)
                if pts:
                    best = max(pts, key=lambda p: p[1])
                    method_best_qps[(ds, sc, m)] = best[0]

    methods = ["UNG", "Post-filter", "SIEVE"]
    pred_field = lambda m: f"{model}_predrecall_{m}"

    # 检查 predictions CSV 有没有 predrecall 字段 (旧文件没有, 用旧版本会报错 fallback)
    with open(pred_path) as f:
        header = next(csv.reader(f))
    has_predrecall = all(pred_field(m) in header for m in methods)
    if not has_predrecall:
        # 旧 predictions, 没存 predicted recall, 退回 oracle pareto
        return {}

    # 收集每 query 的 predicted + actual + chosen QPS
    cells = defaultdict(list)  # (ds, sc) → [(predicted_dict, actual_dict, qps_dict)]
    with open(pred_path) as f:
        for r in csv.DictReader(f):
            try:
                qid = int(r["query_id"])
                ds = r["dataset"]
                sc = r["scenario"]
            except (KeyError, ValueError):
                continue
            pred = {m: float(r.get(pred_field(m), 0)) for m in methods}
            actual = {m: actual_recall.get((qid, ds, sc, m)) for m in methods}
            actual = {m: v for m, v in actual.items() if v is not None}
            qps_d = {m: method_best_qps.get((ds, sc, m)) for m in methods}
            qps_d = {m: q for m, q in qps_d.items() if q is not None}
            if not actual or not qps_d:
                continue
            cells[(ds, sc)].append((pred, actual, qps_d))

    result = defaultdict(list)
    for (ds, sc), queries in cells.items():
        for T in thresholds:
            recalls = []
            qps_list = []
            for pred, actual, qps_d in queries:
                # 用 predicted 筛, 在 actual 里有数据的 method 中
                candidates = [(m, qps_d[m]) for m in qps_d
                              if pred.get(m, 0) >= T and m in actual]
                if not candidates:
                    # fallback: 选 predicted recall 最高的 (model 默认 pick)
                    valid_pred = {m: pred.get(m, 0) for m in actual if m in qps_d}
                    if not valid_pred:
                        continue
                    chosen_m = max(valid_pred, key=valid_pred.get)
                else:
                    # max QPS in candidates
                    chosen_m = max(candidates, key=lambda x: x[1])[0]
                recalls.append(actual[chosen_m])
                qps_list.append(qps_d[chosen_m])
            if not recalls:
                continue
            mean_recall = sum(recalls) / len(recalls)
            harmonic_qps = len(qps_list) / sum(1.0 / q for q in qps_list)
            result[(ds, sc)].append((harmonic_qps, mean_recall))

    _ML_PARETO_CACHE = {"key": cache_key, "data": dict(result)}
    return _ML_PARETO_CACHE["data"]


def load_ml_pareto(dataset: str, scenario: str):
    """List of (qps, recall) — 真实 ML router Pareto (用 predicted recall threshold)."""
    data = load_ml_pareto_all()
    pts = data.get((dataset, scenario), [])
    return sorted(pts, key=lambda p: p[1])


def load_postfilter(dataset: str, scenario: str):
    """Returns list of (qps, recall) sorted by recall."""
    path = BASE_DIR / "faiss" / "results_postfilter" / dataset / scenario / "M=64_efc=400_result.csv"
    if not path.exists():
        return []
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                qps = float(row["QPS"])
                recall = float(row["Recall"])
                pts.append((qps, recall))
            except (ValueError, KeyError):
                continue
    return sorted(pts, key=lambda p: p[1])


def load_ung(dataset: str, scenario: str):
    """从 results_v2_grid/summary.csv 读 UNG 数据.

    取所有 (M, Lb) 配置中 max recall 最高的那组, 在那组里按 Ls 扫描.
    """
    path = BASE_DIR / "UNG-dev" / "results_v2_grid" / "summary.csv"
    if not path.exists():
        return []
    scen_ung = SCEN_UNG[scenario]

    # 收集所有有效行
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("dataset") != dataset:
                continue
            if r.get("scenario") != scen_ung:
                continue
            if r.get("status") != "success":
                continue
            try:
                rec = float(r.get("recall", "") or 0)
                qps = float(r.get("qps", "") or 0)
                M = int(r.get("M", 0))
                Lb = int(r.get("Lb", 0))
                Ls = int(r.get("Ls", 0))
            except ValueError:
                continue
            if rec <= 0 or qps <= 0:
                continue
            rows.append((M, Lb, Ls, rec, qps))

    if not rows:
        return []

    # 找 (M, Lb) 中 max recall 最高的 group
    by_mlb = defaultdict(list)
    for M, Lb, Ls, rec, qps in rows:
        by_mlb[(M, Lb)].append((Ls, rec, qps))
    best_key = max(by_mlb.keys(), key=lambda k: max(r[1] for r in by_mlb[k]))
    best_group = sorted(by_mlb[best_key], key=lambda x: x[0])  # by Ls
    return [(qps, rec) for _Ls, rec, qps in best_group]


def load_sieve(dataset: str, scenario: str):
    """Returns sorted (qps, recall) list, picking best (M, b, h) by max recall."""
    scen_sieve = SCEN_SIEVE[scenario]
    pattern = str(BASE_DIR / "SIEVE" / "results" / f"sieve_{dataset}_{scen_sieve}_M*.csv")
    files = glob.glob(pattern)
    if not files:
        return []

    best_pts = None
    best_max_recall = -1
    for f in files:
        pts = []
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    rec = float(r.get("recall_at_10", 0))
                    qps = float(r.get("qps", 0))
                except ValueError:
                    continue
                if rec > 0 and qps > 0:
                    pts.append((qps, rec))
        if not pts:
            continue
        max_r = max(p[1] for p in pts)
        if max_r > best_max_recall:
            best_max_recall = max_r
            best_pts = sorted(pts, key=lambda p: p[1])
    return best_pts or []


def load_acorn(dataset: str, scenario: str):
    """ACORN: 每个 (M, M_beta, gamma) 一个点 (recall, qps), 全画 scatter.

    Returns list of (qps, recall).
    """
    pts = []
    pattern = str(BASE_DIR / "ACORN" / "data" / f"param_search_{dataset}*" / "results" / dataset / "summary.csv")
    for path in glob.glob(pattern):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("scenario") != scenario:
                    continue
                if r.get("status") != "success":
                    continue
                try:
                    rec = float(r.get("recall@10", 0))
                    qps = float(r.get("qps", 0))
                except ValueError:
                    continue
                if rec > 0 and qps > 0:
                    pts.append((qps, rec))
    return pts


def load_filtered_vamana(dataset: str, scenario: str):
    """DiskANN/FilteredVamana 数据 (best R 配置 in summary.csv).

    summary.csv 字段: dataset, scenario, R, Lsearch, build_time_s, index_size_mb,
    recall@10, qps, latency_us, status
    scenario in summary uses 'containment'/'overlap'/'equality'.
    """
    path = BASE_DIR / "DiskANN" / "data_original" / "results" / dataset / "summary.csv"
    if not path.exists():
        return []
    scen_diskann = SCEN_UNG[scenario]  # same mapping as UNG (containment/overlap/equality)

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("scenario") != scen_diskann:
                continue
            if r.get("status") != "success":
                continue
            try:
                rec = float(r.get("recall@10", 0) or 0)
                qps = float(r.get("qps", 0) or 0)
                R = int(r.get("R", 0))
                Ls = int(r.get("Lsearch", 0))
            except ValueError:
                continue
            if rec <= 0 or qps <= 0:
                continue
            rows.append((R, Ls, rec, qps))

    if not rows:
        return []

    # 选 max recall 的 R, 在那 R 里按 Lsearch 扫
    by_r = defaultdict(list)
    for R, Ls, rec, qps in rows:
        by_r[R].append((Ls, rec, qps))
    best_r = max(by_r.keys(), key=lambda r: max(x[1] for x in by_r[r]))
    best_group = sorted(by_r[best_r], key=lambda x: x[0])
    return [(qps, rec) for _Ls, rec, qps in best_group]


def load_prefilter(dataset: str, scenario: str):
    """Pre-filter brute-force: single point per (ds, sc), recall=1.0 + QPS.

    summary.csv 字段: dataset, scenario, method, qps, recall@10, ...
    """
    path = BASE_DIR / "faiss" / "results_prefilter" / dataset / "summary.csv"
    if not path.exists():
        return []
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("scenario") != scenario:
                continue
            try:
                rec = float(r.get("recall@10", 0) or 0)
                qps = float(r.get("qps", 0) or 0)
            except ValueError:
                continue
            if rec > 0 and qps > 0:
                pts.append((qps, rec))
    return pts


def plot_one(ax, dataset: str, scenario: str):
    methods = [
        ("UNG", load_ung(dataset, scenario)),
        ("Post-filter", load_postfilter(dataset, scenario)),
        ("SIEVE", load_sieve(dataset, scenario)),
        ("FilteredVamana", load_filtered_vamana(dataset, scenario)),
        ("ACORN", load_acorn(dataset, scenario)),
        ("Pre-filter", load_prefilter(dataset, scenario)),
        ("ML Router", load_ml_pareto(dataset, scenario)),  # 真实 ML, 用 predicted recall 阈值
    ]

    for name, pts in methods:
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if name == "ACORN":
            ax.scatter(xs, ys, label=name, color=METHOD_COLORS[name],
                       marker=METHOD_MARKERS[name], s=20, alpha=0.7)
        elif name == "Pre-filter":
            ax.scatter(xs, ys, label=name, color=METHOD_COLORS[name],
                       marker=METHOD_MARKERS[name], s=120, alpha=0.9,
                       edgecolors="black", linewidths=0.5, zorder=5)
        elif name == "ML Router":
            # 真实 ML router (黑实线 + 黄边 X)
            ax.plot(xs, ys, label=name, color=METHOD_COLORS[name],
                    marker=METHOD_MARKERS[name], markersize=10, linewidth=2.5,
                    markeredgecolor="yellow", markeredgewidth=1.5, zorder=10)
        else:
            ax.plot(xs, ys, label=name, color=METHOD_COLORS[name],
                    marker=METHOD_MARKERS[name], markersize=4, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("QPS (log)", fontsize=8)
    ax.set_ylabel("recall@10", fontsize=8)
    ax.set_title(f"{dataset} / {scenario}", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)


def main():
    fig, axes = plt.subplots(len(V2_DATASETS), len(SCENARIOS),
                             figsize=(15, 5 * len(V2_DATASETS)),
                             squeeze=False)

    for i, ds in enumerate(V2_DATASETS):
        for j, sc in enumerate(SCENARIOS):
            plot_one(axes[i][j], ds, sc)

    # 加图例 (整个 figure 一次)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        # 兜底: 从其他 subplot 找
        for row in axes:
            for ax in row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
    fig.legend(handles, labels, loc="upper center", ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))

    fig.suptitle("V2 recall-QPS: 6 baselines + ML Router (MLP-Reg, T=0.5..0.99 recall thresholds)",
                 fontsize=12, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = BASE_DIR / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "v2_recall_qps.pdf"
    png_path = out_dir / "v2_recall_qps.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
