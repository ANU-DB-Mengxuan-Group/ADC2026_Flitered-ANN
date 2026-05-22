#!/usr/bin/env python3
"""Online routing functions for filtered ANN.

Optimized routing pipeline used by latency_benchmark.py:
  - Packed bitmap (uint8, np.packbits) for selectivity computation
    -> 8x smaller memory, 8x faster bitwise ops
  - Manual MLP forward (extracted from sklearn weights)
    -> bypasses sklearn predict() overhead (~130 us -> a few us)

Public API:
  compute_selectivity_packed(q_labels, scenario, packed_data)
  extract_mlp_weights(sklearn_mlp)
  predict_recalls_manual(x_scaled, manual_models)
  pick_method(predicted_recalls)
  lookup_config(method, dataset, scenario, config_table, T)
"""
from __future__ import annotations

import numpy as np


# ============================================================
# Selectivity (packed bitmap)
# ============================================================

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def _popcount_packed(packed_arr: np.ndarray) -> int:
    """Fast popcount on packed uint8 array.

    Prefers NumPy >= 2.0 native np.bitwise_count (SIMD); falls back to LUT.
    """
    if _HAS_BITWISE_COUNT:
        return int(np.bitwise_count(packed_arr).sum())
    return int(_POPCOUNT_LUT[packed_arr].sum())


def compute_selectivity_packed(q_labels, scenario, packed_data) -> float:
    """Selectivity using packed bitmap inverted index.

    AND  : popcount(intersect packed[l] for l in q_labels) / n_base
    OR   : popcount(union     packed[l] for l in q_labels) / n_base
    EQUAL: set_count[frozenset(q_labels)] / n_base
    """
    n_base = packed_data["n_base"]
    l2i = packed_data["label_to_idx"]
    packed = packed_data["packed_bitmaps"]

    if scenario == "and":
        if not all(l in l2i for l in q_labels):
            return 0.0
        idxs = [l2i[l] for l in q_labels]
        if len(idxs) == 1:
            return _popcount_packed(packed[idxs[0]]) / n_base
        result = packed[idxs[0]] & packed[idxs[1]]
        for i in idxs[2:]:
            result &= packed[i]
        return _popcount_packed(result) / n_base

    if scenario == "or":
        idxs = [l2i[l] for l in q_labels if l in l2i]
        if not idxs:
            return 0.0
        if len(idxs) == 1:
            return _popcount_packed(packed[idxs[0]]) / n_base
        result = packed[idxs[0]] | packed[idxs[1]]
        for i in idxs[2:]:
            result |= packed[i]
        return _popcount_packed(result) / n_base

    if scenario == "equal":
        return packed_data["set_count"].get(frozenset(q_labels), 0) / n_base

    raise ValueError(f"unknown scenario: {scenario}")


# Backward-compatible alias (some scripts still call this name)
compute_selectivity_bitmap = compute_selectivity_packed


# ============================================================
# Selectivity (Roaring bitmap, exact, faster on sparse labels)
# ============================================================

def convert_packed_to_roaring(packed_data: dict) -> dict:
    """Convert a packed-bitmap inverted index to a Roaring-bitmap index.

    Output dict is interface-compatible with packed_data:
      - n_base
      - label_to_idx   (unchanged)
      - roaring_bitmaps : list of BitMap, one per label index
      - set_count       (unchanged, used for Equality)
    """
    try:
        from pyroaring import BitMap
    except ImportError as e:
        raise ImportError(
            "pyroaring not installed; run `pip install pyroaring`"
        ) from e

    n_base = packed_data["n_base"]
    packed = packed_data["packed_bitmaps"]
    roaring = []
    for arr in packed:
        unpacked = np.unpackbits(arr)[:n_base]
        ones = np.flatnonzero(unpacked).astype(np.uint32)
        roaring.append(BitMap(ones))
    return {
        "n_base": n_base,
        "label_to_idx": packed_data["label_to_idx"],
        "roaring_bitmaps": roaring,
        "set_count": packed_data["set_count"],
    }


def compute_selectivity_roaring(q_labels, scenario, roaring_data) -> float:
    """Selectivity using Roaring-bitmap inverted index.

    Numerically identical to compute_selectivity_packed; faster on sparse
    labels because Roaring stores sparse chunks as sorted-int arrays
    and runs intersect / union in native C.
    """
    n_base = roaring_data["n_base"]
    l2i = roaring_data["label_to_idx"]
    bitmaps = roaring_data["roaring_bitmaps"]

    if scenario == "and":
        if not all(l in l2i for l in q_labels):
            return 0.0
        idxs = [l2i[l] for l in q_labels]
        if len(idxs) == 1:
            return len(bitmaps[idxs[0]]) / n_base
        result = bitmaps[idxs[0]] & bitmaps[idxs[1]]
        for i in idxs[2:]:
            result &= bitmaps[i]
        return len(result) / n_base

    if scenario == "or":
        idxs = [l2i[l] for l in q_labels if l in l2i]
        if not idxs:
            return 0.0
        if len(idxs) == 1:
            return len(bitmaps[idxs[0]]) / n_base
        result = bitmaps[idxs[0]] | bitmaps[idxs[1]]
        for i in idxs[2:]:
            result |= bitmaps[i]
        return len(result) / n_base

    if scenario == "equal":
        return roaring_data["set_count"].get(frozenset(q_labels), 0) / n_base

    raise ValueError(f"unknown scenario: {scenario}")


# ============================================================
# Manual MLP forward (bypass sklearn predict overhead)
# ============================================================

def extract_mlp_weights(sklearn_mlp) -> dict:
    """Extract weights & biases from a trained sklearn MLPRegressor.

    Returns dict with 'weights' (list of 2D ndarrays) and 'biases' (list of 1D).
    Activations are assumed: ReLU on hidden layers, identity on output
    (which is MLPRegressor's default).
    """
    return {
        "weights": [w.astype(np.float32) for w in sklearn_mlp.coefs_],
        "biases": [b.astype(np.float32) for b in sklearn_mlp.intercepts_],
    }


def mlp_forward_manual(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """Manual MLP forward. Supports batch input.

    Args:
        x: shape (B, in_dim) or (1, in_dim)
        weights: list of W matrices
        biases: list of b vectors
    Returns:
        ndarray shape (B, out_dim)
    """
    h = x
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1:   # ReLU on hidden, identity on output
            h = np.maximum(0.0, h, out=h)
    return h


def predict_recalls_manual(x_scaled: np.ndarray, manual_models: dict) -> dict:
    """{method: predicted_recall} using manual MLP forward.

    Args:
        x_scaled: shape (1, in_dim), scaled feature vector
        manual_models: {method: {'weights': [...], 'biases': [...]}}
    """
    return {
        m: float(mlp_forward_manual(x_scaled, md["weights"], md["biases"]).item())
        for m, md in manual_models.items()
    }


# ============================================================
# Feature vector / argmax / config lookup
# ============================================================

def build_feature_vector_minimal(selectivity: float, lid_mean: float) -> np.ndarray:
    return np.array([[selectivity, lid_mean]], dtype=np.float32)


def scale_manual(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Manual feature scaling (bypass sklearn scaler.transform overhead).

    Equivalent to StandardScaler.transform but skips input validation.
    """
    return (x - mean) / scale


def pick_method(predicted_recalls: dict) -> str:
    return max(predicted_recalls, key=predicted_recalls.get)


def lookup_config(method, dataset, scenario, config_table, T=0.7):
    candidates = config_table.get((dataset, scenario, method), [])
    if not candidates:
        return None
    above = [(c, r, q) for (c, r, q) in candidates if r >= T]
    if not above:
        above = candidates
    best = max(above, key=lambda x: x[2])
    return best[0]


# ============================================================
# Strict routing: method-level gate by predicted recall,
# then global max QPS across surviving (method, config) pairs.
# Falls back to argmax (Plain) when no method passes the gate.
# ============================================================

def pick_methods_strict(predicted_recalls: dict, T: float = 0.7) -> list:
    """Return list of methods whose predicted recall >= T."""
    return [m for m, r in predicted_recalls.items() if r >= T]


def lookup_config_strict(passing_methods, dataset, scenario, config_table, T=0.7):
    """For each passing method, take its fastest config with measured recall >= T,
    then return the global max-QPS pick across passing methods.

    Returns (method, config, recall, qps) tuple, or None if no candidate.
    """
    candidates = []
    for m in passing_methods:
        configs = config_table.get((dataset, scenario, m), [])
        above = [(c, r, q) for (c, r, q) in configs if r >= T]
        if not above:
            continue
        best_c, best_r, best_q = max(above, key=lambda x: x[2])
        candidates.append((m, best_c, best_r, best_q))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[3])


# ============================================================
# End-to-end (mostly for sanity testing, not timing)
# ============================================================

def route_minimal(q_labels, scenario, dataset, packed_data, lid_mean,
                  manual_models, scaler, config_table, T=0.7):
    """Plain (argmax-first) routing. Kept for backward compatibility / comparison."""
    sel = compute_selectivity_packed(q_labels, scenario, packed_data)
    x = build_feature_vector_minimal(sel, lid_mean)
    x_scaled = scaler.transform(x).astype(np.float32)
    preds = predict_recalls_manual(x_scaled, manual_models)
    method = pick_method(preds)
    config = lookup_config(method, dataset, scenario, config_table, T)
    return method, config, preds


def route_strict(q_labels, scenario, dataset, packed_data, lid_mean,
                 manual_models, scaler, config_table, T=0.7):
    """Strict routing: filter methods by predicted recall >= T,
    pick global max-QPS (method, config) among passing methods.
    Falls back to argmax if no method passes.
    """
    sel = compute_selectivity_packed(q_labels, scenario, packed_data)
    x = build_feature_vector_minimal(sel, lid_mean)
    x_scaled = scaler.transform(x).astype(np.float32)
    preds = predict_recalls_manual(x_scaled, manual_models)

    passing = pick_methods_strict(preds, T)
    if passing:
        pick = lookup_config_strict(passing, dataset, scenario, config_table, T)
        if pick is not None:
            method, config, _, _ = pick
            return method, config, preds, "strict"

    # Fallback: argmax (Plain)
    method = pick_method(preds)
    config = lookup_config(method, dataset, scenario, config_table, T)
    return method, config, preds, "fallback_argmax"


# ============================================================
# Legacy predict_recalls (kept for backward compatibility / sklearn-based timing)
# ============================================================

def predict_recalls(feature_vec, regressors: dict, scaler=None) -> dict:
    """Sklearn-based predict (slower; only used for baseline timing)."""
    x = feature_vec if scaler is None else scaler.transform(feature_vec)
    return {m: float(reg.predict(x)[0]) for m, reg in regressors.items()}
