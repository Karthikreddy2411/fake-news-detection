"""
src/metrics.py
─────────────────────────────────────────────────────────────────────────────
Fidelity metrics for XAI evaluation (Section VI-G).

Four metrics are computed for each (model, XAI method) pair:

  1. Δcomp  (Completeness) — Eq. (1)
     f(x) − f(x \ S_k)
     Remove top-k tokens; large drop → explanation identified key signal.

  2. Δsuff  (Sufficiency) — Eq. (2)
     f(x) − f(x | S_k)
     Keep only top-k tokens; small drop → core info is captured in S_k.

  3. AOPC   (Area Over Perturbation Curve) — Eq. (3)
     (1/m) Σ_{i=1}^{m} (p(0) − p(i))
     Average confidence drop as tokens are removed one-by-one (highest first).

  4. Flip@k — Eq. (4)
     min{i ≤ k : sign(f(x^(i))) ≠ sign(f(x^(0)))}
     Minimum tokens needed to flip the predicted label.

All perturbations replace selected positions with PAD (index 0),
consistent with the model's training distribution.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Callable
from tqdm import tqdm
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_tokens(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Replace positions `indices` with PAD. Shape preserved."""
    x_new = x.copy()
    x_new[indices] = C.PAD_IDX
    return x_new


def _keep_only(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Replace all positions EXCEPT `indices` with PAD."""
    x_new = np.full_like(x, C.PAD_IDX)
    x_new[indices] = x[indices]
    return x_new


def _top_k_indices(attributions: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """Return indices of the k highest |attribution| non-PAD positions."""
    non_pad = np.where(x != C.PAD_IDX)[0]
    if len(non_pad) == 0:
        return np.array([], dtype=int)
    sorted_idx = non_pad[np.argsort(-np.abs(attributions[non_pad]))]
    return sorted_idx[:min(k, len(sorted_idx))]


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_instance_metrics(
    model: tf.keras.Model,
    x: np.ndarray,
    attributions: np.ndarray,
    k: int = C.TOP_K,
    aopc_steps: int = C.AOPC_STEPS,
) -> dict:
    """
    Compute all four metrics for a single instance.

    Parameters
    ----------
    model        : compiled tf.keras.Model
    x            : padded token-ID sequence, shape (MAX_LEN,)
    attributions : attribution vector, shape (MAX_LEN,)
    k            : top-k tokens to use
    aopc_steps   : m in the AOPC formula

    Returns
    -------
    dict with keys: delta_comp, delta_suff, aopc, flip_at_k
    """

    def predict(seq):
        pred = model.predict(np.expand_dims(seq, 0), verbose=0)
        return float(pred[0, 0])

    p0 = predict(x)  # f(x) — original probability

    # ── Δcomp ─────────────────────────────────────────────────────────────────
    sk = _top_k_indices(attributions, k, x)
    x_removed = _remove_tokens(x, sk)
    p_removed  = predict(x_removed)
    delta_comp = p0 - p_removed

    # ── Δsuff ─────────────────────────────────────────────────────────────────
    x_kept  = _keep_only(x, sk)
    p_kept  = predict(x_kept)
    delta_suff = p0 - p_kept

    # ── AOPC ──────────────────────────────────────────────────────────────────
    # Remove tokens one by one in order of decreasing |attribution|
    top_m = _top_k_indices(attributions, aopc_steps, x)
    aopc_acc = 0.0
    x_cur = x.copy()
    for i, idx in enumerate(top_m[:aopc_steps]):
        x_cur[idx] = C.PAD_IDX
        p_i = predict(x_cur)
        aopc_acc += (p0 - p_i)
    aopc = aopc_acc / max(len(top_m[:aopc_steps]), 1)

    # ── Flip@k ────────────────────────────────────────────────────────────────
    label0 = int(p0 >= 0.5)
    flip_at_k = k  # sentinel: did not flip within k tokens
    x_cur2 = x.copy()
    for i, idx in enumerate(top_m[:k]):
        x_cur2[idx] = C.PAD_IDX
        p_i = predict(x_cur2)
        if int(p_i >= 0.5) != label0:
            flip_at_k = i + 1
            break

    return {
        "delta_comp": delta_comp,
        "delta_suff": delta_suff,
        "aopc":       aopc,
        "flip_at_k":  flip_at_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate over a test sample
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_xai_method(
    model: tf.keras.Model,
    explainer,
    X_test: np.ndarray,
    n_samples: int = C.EVAL_SAMPLE_N,
    method_name: str = "XAI",
) -> dict:
    """
    Run an XAI method over `n_samples` test articles and return averaged metrics.

    Parameters
    ----------
    model       : tf.keras.Model
    explainer   : SHAPExplainer / LIMEExplainer / IGExplainer instance
    X_test      : padded integer sequences, shape (N, MAX_LEN)
    n_samples   : how many instances to evaluate
    method_name : label for progress bar

    Returns
    -------
    dict: mean delta_comp, delta_suff, aopc, flip_at_k, time_per_sample
    """
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    results = []
    t_start = time.time()

    for i in tqdm(idx, desc=f"  [{method_name}]"):
        x = X_test[i]
        attributions = explainer.explain(x)
        m = compute_instance_metrics(model, x, attributions)
        results.append(m)

    elapsed = time.time() - t_start
    time_per = elapsed / len(idx)

    df = pd.DataFrame(results)
    return {
        "delta_comp": float(df["delta_comp"].mean()),
        "delta_suff": float(df["delta_suff"].mean()),
        "aopc":       float(df["aopc"].mean()),
        "flip_at_k":  float(df["flip_at_k"].mean()),
        "time_per_sample_s": time_per,
    }


def print_results_table(results: dict, model_name: str):
    """Pretty-print a results table matching Tables II/III in the paper."""
    print(f"\n  {'─'*70}")
    print(f"  Explanation Quality — {model_name}")
    print(f"  {'─'*70}")
    header = f"  {'Method':<20} {'Δcomp':>8} {'Δsuff':>8} {'AOPC':>8} {'Flip@k':>8} {'Time(s)':>9}"
    print(header)
    print(f"  {'─'*70}")
    for method, stats in results.items():
        print(
            f"  {method:<20}"
            f" {stats['delta_comp']:>8.6f}"
            f" {stats['delta_suff']:>8.6f}"
            f" {stats['aopc']:>8.6f}"
            f" {stats['flip_at_k']:>8.2f}"
            f" {stats['time_per_sample_s']:>9.3f}"
        )
    print(f"  {'─'*70}\n")
