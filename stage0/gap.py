"""
stage0/gap.py
=============
Computes the Stage 0 gate metric per layer:

    gap_l = ||H_loss_l - H_lay_l||_F  /  ||H_lay_l||_F

Where:
    H_lay_l  = N * Sigma_l  (GPTQ's Hessian, exact, from LayerStats)
    H_loss_l = true loss Hessian (estimated via L4 HVP, never materialized)

Also computes:
    trace(H_loss_l)  — total true curvature of layer l
    trace(H_lay_l)   — total curvature per GPTQ's approximation (= L1 rung score * N)

All three use Hutchinson's estimator:
    tr(A)    = E[v^T A v]      for Rademacher v ~ {+-1}^n
    ||A||_F² = E[||Av||²]      for Rademacher v ~ {+-1}^n

A small gap (< 0.1) confirms H_loss ≈ H_lay, validating the ladder design.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import LayerStats


def _hutchinson_trace(matvec, n_dim: int, n_probe: int = 30, seed: int = 0) -> float:
    """Estimates tr(A) = E[v^T A v] for Rademacher v."""
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_probe):
        v    = rng.choice([-1.0, 1.0], size=n_dim)
        acc += float(np.dot(v, matvec(v)))
    return acc / n_probe


def _hutchinson_frob_sq(matvec, n_dim: int, n_probe: int = 30, seed: int = 0) -> float:
    """Estimates ||A||_F² = E[||Av||²] for Rademacher v."""
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_probe):
        v   = rng.choice([-1.0, 1.0], size=n_dim)
        Av  = matvec(v)
        acc += float(np.dot(Av, Av))
    return acc / n_probe



def compute_hessian_gap(
    stats:   Dict[str, LayerStats],
    n_probe: int = 30,
    seed:    int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    For each layer with HVP attached, computes the relative Frobenius
    approximation error of Eq. 2 in the report:

        gap_l = ||H_loss_l - H_lay_l||_F / ||H_lay_l||_F

    where H_lay_l = N * Sigma_l is materialized exactly (no Hutchinson
    needed on that side — only H_loss is matvec-only).

    Returns
    -------
    gap_by_layer, trace_h_loss_by_layer, trace_n_sigma_by_layer
    Layers with hvp=None get nan.
    """
    gaps, tr_h_loss, tr_n_sigma = {}, {}, {}

    for layer_id, st in stats.items():
        n = st.weight_shape[1]   # d_in
        N = st.n_tokens

        # H_lay = N * Sigma is already a materialized array — compute
        # trace and Frobenius norm exactly, no Hutchinson needed here.
        H_lay  = st.sigma * N
        tr_ns  = float(np.trace(H_lay))
        fro_ns = float(np.linalg.norm(H_lay, "fro"))
        tr_n_sigma[layer_id] = tr_ns

        def h_lay_mv(v):  return H_lay @ v

        if st.hvp is None:
            gaps[layer_id]      = float('nan')
            tr_h_loss[layer_id] = float('nan')
            continue

        # H_loss is never materialized — only matvec access exists,
        # so trace and the numerator of the gap still need Hutchinson.
        tr_hl = _hutchinson_trace(st.hvp, n, n_probe, seed)
        tr_h_loss[layer_id] = tr_hl

        def diff_mv(v):  return st.hvp(v) - h_lay_mv(v)

        num_sq = _hutchinson_frob_sq(diff_mv, n, n_probe, seed)
        gap    = float(np.sqrt(num_sq) / max(fro_ns, 1e-12))

        gaps[layer_id] = gap
        print(f"  [gap] {layer_id:50s}  gap={gap:.4f}  "
              f"tr(H_loss)={tr_hl:.4f}  tr(H_lay)={tr_ns:.2f}")

    return gaps, tr_h_loss, tr_n_sigma




# def compute_hessian_gap(
#     stats:   Dict[str, LayerStats],
#     n_probe: int = 30,
#     seed:    int = 0,
# ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
#     """
#     For each layer with HVP attached, computes the structurally normalized gap:
#       - gap_by_layer           : ||H_loss/tr(H_loss) - H_lay/tr(H_lay)||_F
#       - trace_h_loss_by_layer  : tr(H_loss) estimated via Hutchinson
#       - trace_n_sigma_by_layer : tr(H_lay) = tr(N * Sigma) via Hutchinson

#     Returns
#     -------
#     gap_by_layer, trace_h_loss_by_layer, trace_n_sigma_by_layer
#     Layers with hvp=None get nan.
#     """
#     gaps, tr_h_loss, tr_n_sigma = {}, {}, {}

#     for layer_id, st in stats.items():
#         n = st.weight_shape[1]   # d_in
#         N = st.n_tokens

#         # 1. Define the layer approximation MV product
#         def h_lay_mv(v):  return (st.sigma * N) @ v

#         # 2. Estimate trace of H_lay
#         tr_ns = _hutchinson_trace(h_lay_mv, n, n_probe, seed)
#         tr_n_sigma[layer_id] = tr_ns

#         if st.hvp is None:
#             gaps[layer_id]      = float('nan')
#             tr_h_loss[layer_id] = float('nan')
#             continue

#         # 3. Estimate trace of H_loss
#         tr_hl = _hutchinson_trace(st.hvp, n, n_probe, seed)

#         # 4. Standardize scales using the estimated traces to fix the 1/M^2 mismatch
#         norm_hl = tr_hl if abs(tr_hl) > 1e-12 else 1.0
#         norm_ns = tr_ns if abs(tr_ns) > 1e-12 else 1.0

#         # This structural difference operator scales both matrices to an effective trace of 1.0
#         def normalized_diff_mv(v):  
#             return (st.hvp(v) / norm_hl) - (h_lay_mv(v) / norm_ns)

#         # 5. Compute Frobenius norm of the normalized mismatch landscape
#         num_sq = _hutchinson_frob_sq(normalized_diff_mv, n, n_probe, seed)
#         gap    = float(np.sqrt(num_sq))

#         gaps[layer_id] = gap
#         print(f"  [gap] {layer_id:50s}  gap={gap:.4f}  "
#               f"tr(H_loss)={tr_hl:.4f}  tr(H_lay)={tr_ns:.2f}")

#     return gaps, tr_h_loss, tr_n_sigma


