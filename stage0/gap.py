"""
stage0/gap.py — Step 3 of the Stage 0 pipeline.

Measures the residual of Eq. (2),  H_loss^(l) ~ N Sigma^(l), per layer.

WHAT IS COMPARED
----------------
  M     = sum_k [H_loss]_kk    (input-space diagonal blocks; see hvp.py)
  Sigma = (1/N) X X^T          computed on the SAME tokens as M

Under exact Kronecker structure with A = Sigma:   M = Tr(G) . Sigma.
So the claim has two separable parts, and we report them separately:

  SHAPE : does M point in the same direction as Sigma?
  SCALE : is the multiplier the one the theory predicts?

WHY A COSINE IS THE PRIMARY SHAPE METRIC
----------------------------------------
Expanding the Frobenius gap at normalization c:

    gap^2(c) = ||M||_F^2 / c^2  -  2 <M,Sigma>_F / c  +  ||Sigma||_F^2

Three scalars determine the gap at EVERY c. Estimate them once, evaluate any
normalization for free. Minimizing over c gives

    c_opt      = ||M||_F^2 / <M,Sigma>_F
    gap^2_min  = ||Sigma||_F^2 (1 - cos^2 theta),
    cos theta  = <M,Sigma>_F / (||M||_F ||Sigma||_F)

cos theta is scale-free BY CONSTRUCTION -- it needs no normalization choice,
so it cannot be contaminated by a wrong Tr(G), a mean-vs-sum convention, or a
factor of 2. It is the honest answer to "does Sigma have the right geometry?".
The normalization-dependent gaps are then reported as a SCALE question on top.

WHY THE SPLIT (TWO-INDEPENDENT-ESTIMATE) TRICK IS MANDATORY
-----------------------------------------------------------
M's action is estimated by a noisy probe. Noise does not cancel in a square:

    E||M_hat v||^2 = ||M v||^2 + Var(M_hat v)      <-- INFLATED

A naive gap therefore grows as you use FEWER probes -- i.e. the reported
"Eq. (2) residual" would partly be your own probe budget, and would shrink if
a reviewer asked you to probe harder. Using two INDEPENDENT estimates a, b of
Mv and taking <a,b> instead of ||a||^2 removes it exactly, since the two noise
terms are independent and zero-mean:

    E[<a,b>] = <E a, E b> = ||M v||^2                <-- UNBIASED at any noise

MATCHED DATA
------------
Sigma is recomputed here on exactly the HVP token subset. Comparing M (a few
thousand tokens) against a Sigma built from 65k tokens would make sampling
noise, not Eq. (2), the dominant term in the gap. Mismatched N is variance,
not scale; it cannot be normalized away.

OPEN CONVENTION QUESTION (do not let this block the run)
--------------------------------------------------------
Deriving M ~ Tr(G) . Sigma, the K-FAC swap  sum_n G_n (x) A_n ~ N Ghat (x) Ahat
leaves a factor N behind, giving M ~ N . Tr(G) . Sigma. The supervisor's email
writes the normalizer as Tr(G) with no N. We do NOT pick one: c_opt and c_trace
are measured from M itself and are immune, and we report c_measured/Tr(G) and
c_measured/(N Tr(G)) so the data says which convention is in force.
"""
from typing import Dict, Optional
import numpy as np
import torch
from transformers.pytorch_utils import Conv1D


# ─────────────────────────────────────────────────────────────────────────
# Matched Sigma — recomputed on the HVP token subset only
# ─────────────────────────────────────────────────────────────────────────

def matched_sigma(
    model,
    ids:        torch.Tensor,
    layer_ids,
    device:     str,
    batch_size: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Sigma = (1/N) X^T X on exactly `ids`. Mirrors stage0.capture.build_layer_stats
    but takes explicit token ids rather than a seed, and accumulates on-device
    in float64 (65k rank-1 updates in fp32 lose meaningful precision).
    """
    mods = dict(model.named_modules())
    acc: Dict[str, torch.Tensor] = {}
    cnt: Dict[str, int] = {}

    def make_hook(lid):
        def hook(module, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            if lid not in acc:
                acc[lid] = torch.zeros(x.shape[1], x.shape[1],
                                       dtype=torch.float64, device=x.device)
                cnt[lid] = 0
            acc[lid] += x.T @ x
            cnt[lid] += x.shape[0]
        return hook

    handles = [mods[lid].register_forward_hook(make_hook(lid)) for lid in layer_ids]
    with torch.no_grad():
        for s in range(0, ids.shape[0], batch_size):
            model(ids[s:s + batch_size].to(device))
    for h in handles:
        h.remove()

    return {lid: (acc[lid] / cnt[lid]).cpu().numpy() for lid in acc}, \
           {lid: cnt[lid] for lid in cnt}


# ─────────────────────────────────────────────────────────────────────────
# The gap
# ─────────────────────────────────────────────────────────────────────────

def compute_hessian_gap(
    stats,
    model,
    hvp_ids:    torch.Tensor,
    device:     str,
    n_v:        int = 30,
    n_trace:    int = 20,
    batch_size: int = 2,
    probe_seed: int = 0,
    verbose:    bool = True,
) -> Dict[str, dict]:
    """
    Per-layer measurement of the Eq. (2) residual.

    Parameters
    ----------
    n_v      : v-probes for the shape estimate. Each costs 2 hvp calls (split).
    n_trace  : DISJOINT v-probes for Tr(M). Held out so that c_trace is not
               fitted to the same noise the gap is measured against.

    Cost per layer = (2*n_v + n_trace) hvp calls, each of which is n_batches
    double-backward passes. Time ONE layer before launching all 48.

    Returns
    -------
    dict[layer_id] -> dict with keys:
      cos_theta      : <M,Sigma>/(||M|| ||Sigma||). PRIMARY. Scale-free shape.
                       1.0 = Sigma has exactly the right geometry.
      gap_opt        : sqrt(1 - cos^2). Minimum achievable relative gap.
      gap_trace      : relative gap at c = Tr(M)/Tr(Sigma).
      gap_theory     : relative gap at c = Tr(G). None if trace_g missing.
      c_opt          : ||M||^2/<M,Sigma>   -- least-squares best scalar
      c_trace        : Tr(M)/Tr(Sigma)     -- trace-matched scalar
      c_theory       : Tr(G)               -- what theory predicts
      scale_ratio    : c_opt / c_theory. ~1 => scaling confirmed.
      scale_ratio_N  : c_opt / (N * c_theory). The rival convention.
      g_bar          : Tr(G)/d_out. Eq.(2)'s G=I convention predicts 1.
      frac_neg_quad  : fraction of probes with v'Mv < 0. The true Hessian is
                       indefinite off a stationary point; Sigma is PSD by
                       construction. Large values mean part of the gap is this
                       structural mismatch, not spectrum disagreement. REPORT IT.
      tr_M, tr_sigma, fro_M, fro_sigma, n_tokens
      region         : 'attn' | 'ffn' | 'other', from LayerStats — raw metadata only.
                       NOTE: this is a coarser 2-way split than the c_attn/c_proj/c_fc
                       grouping used for the by-layer-type plots in run_stage0.py; the
                       two are not interchangeable (e.g. attn.c_proj is region='attn'
                       but layer-type='c_proj').
    """
    quant = [lid for lid, s in stats.items()
             if getattr(s, 'quantizable', True) and s.hvp is not None]

    if verbose:
        print(f"  Recomputing Sigma on {hvp_ids.shape[0]} matched sequences...")
    sig_matched, tok_counts = matched_sigma(model, hvp_ids, quant, device, batch_size)

    out: Dict[str, dict] = {}
    rng = np.random.default_rng(probe_seed)

    for lid in quant:
        s = stats[lid]
        S = sig_matched[lid]
        d_in = S.shape[0]
        d_out = s.weight_shape[0]

        # ---- exact, from the stored matrix; no need to estimate these ----
        tr_sigma  = float(np.trace(S))
        fro_sigma = float(np.linalg.norm(S, 'fro'))

        # ---- Tr(M) on a HELD-OUT probe set:  Tr(M) = E_v[v' M v] ----
        tr_M = 0.0
        n_neg = 0
        for _ in range(n_trace):
            v = rng.integers(0, 2, d_in).astype(np.float64) * 2 - 1
            q = float(v @ s.hvp(v, rng))
            tr_M += q
            n_neg += (q < 0)
        tr_M /= n_trace

        # ---- shape moments, split estimator ----
        # F_MM  -> ||M||_F^2       via E_v[<a,b>], a,b independent
        # F_MS  -> <M,Sigma>_F     via E_v[<a,Sv>]
        F_MM, F_MS = 0.0, 0.0
        for _ in range(n_v):
            v = rng.integers(0, 2, d_in).astype(np.float64) * 2 - 1
            a = s.hvp(v, rng)              # independent u-draw
            b = s.hvp(v, rng)              # independent u-draw  <-- the split
            Sv = S @ v
            F_MM += float(a @ b)                       # noise cancels: unbiased
            F_MS += float(a @ Sv + b @ Sv) / 2.0
        F_MM /= n_v
        F_MS /= n_v
        F_SS = fro_sigma ** 2                          # exact, lower variance

        fro_M = float(np.sqrt(max(F_MM, 0.0)))

        def rel_gap(c: float) -> float:
            """sqrt(||M/c - Sigma||_F^2) / ||Sigma||_F"""
            if c == 0 or not np.isfinite(c):
                return float('nan')
            g2 = F_MM / c**2 - 2.0 * F_MS / c + F_SS
            return float(np.sqrt(max(g2, 0.0)) / fro_sigma)   # clip: noise can dip <0

        cos_theta = F_MS / max(fro_M * fro_sigma, 1e-30)
        c_opt   = F_MM / F_MS if abs(F_MS) > 1e-30 else float('nan')
        c_trace = tr_M / tr_sigma if abs(tr_sigma) > 1e-30 else float('nan')
        c_th    = s.trace_g
        N       = tok_counts[lid]

        out[lid] = dict(
            cos_theta     = cos_theta,
            gap_opt       = float(np.sqrt(max(1.0 - cos_theta**2, 0.0))),
            gap_trace     = rel_gap(c_trace),
            gap_theory    = rel_gap(c_th) if c_th else None,
            c_opt         = c_opt,
            c_trace       = c_trace,
            c_theory      = c_th,
            scale_ratio   = (c_opt / c_th) if c_th else None,
            scale_ratio_N = (c_opt / (N * c_th)) if c_th else None,
            g_bar         = (c_th / d_out) if c_th else None,
            frac_neg_quad = n_neg / n_trace,
            tr_M          = tr_M,
            tr_sigma      = tr_sigma,
            fro_M         = fro_M,
            fro_sigma     = fro_sigma,
            n_tokens      = N,
            region        = s.region,
        )
        if verbose:
            print(f"    {lid:38s} cos={cos_theta:6.3f}  gap_opt={out[lid]['gap_opt']:.3f}  "
                  f"neg={out[lid]['frac_neg_quad']:.2f}")

    return out


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────

def report_gap(res: Dict[str, dict]) -> None:
    """
    Reads as:
      cos ~ 1, scale_ratio ~ 1   -> Eq. (2) holds. L1/GPTQ is well-founded.
      cos ~ 1, scale_ratio != 1  -> Sigma has the right GEOMETRY, wrong SCALE.
                                    If scale_ratio VARIES across layers, that is
                                    the real finding: output-side sensitivity
                                    that L1 structurally cannot see. If it is
                                    CONSTANT across layers, it is a units bug --
                                    check it before claiming anything.
      cos << 1                   -> Sigma has the wrong geometry. Check
                                    frac_neg_quad first: an indefinite M cannot
                                    match a PSD Sigma, and that is a statement
                                    about non-convergence, not about spectra.
    """
    print(f"\n  {'layer':38s} {'cos':>7s} {'gap_opt':>8s} {'gap_tr':>8s} "
          f"{'ratio':>9s} {'g_bar':>10s} {'neg':>6s}")
    for lid in sorted(res):
        r = res[lid]
        sr = f"{r['scale_ratio']:9.3e}" if r['scale_ratio'] is not None else f"{'--':>9s}"
        gb = f"{r['g_bar']:10.3e}"      if r['g_bar']       is not None else f"{'--':>10s}"
        print(f"  {lid:38s} {r['cos_theta']:7.3f} {r['gap_opt']:8.3f} "
              f"{r['gap_trace']:8.3f} {sr} {gb} {r['frac_neg_quad']:6.2f}")

    cos = np.array([r['cos_theta'] for r in res.values()])
    print(f"\n  cos theta : mean={cos.mean():.3f}  min={cos.min():.3f}  max={cos.max():.3f}")

    def _ratio_block(name: str, key: str):
        """Median + spread for a scale-ratio-like field, printed with no threshold or
        interpretation applied — this file does not pick a side on the N-factor
        convention (see the OPEN CONVENTION QUESTION in the module docstring); the
        raw numbers are reported for the reader to judge."""
        vals = [r[key] for r in res.values() if r[key] is not None]
        if not vals:
            return
        ra = np.array(vals)
        spread = ra.max() / max(ra.min(), 1e-30)
        print(f"  {name}: median={np.median(ra):.3e}  spread={spread:.1f}x")
        print("    -> spread ~1x  => CONSTANT: suspect a units/convention bug, not a result.")
        print("    -> spread >>1x => VARIES BY LAYER: real output-side sensitivity.")

    # Both N-factor conventions reported side by side, unchanged from before this pass —
    # scale_ratio_N was simply missing from this function; adding it doesn't change what
    # the file already does (measure both, assert neither).
    _ratio_block("scale_ratio  ", "scale_ratio")
    _ratio_block("scale_ratio_N", "scale_ratio_N")

    neg = np.array([r['frac_neg_quad'] for r in res.values()])
    print(f"\n  frac_neg_quad: mean={neg.mean():.3f}  min={neg.min():.3f}  max={neg.max():.3f}")

























# """
# stage0/gap.py
# =============
# Computes the Stage 0 gate metric per layer:

#     gap_l = ||H_loss_l - H_lay_l||_F  /  ||H_lay_l||_F

# Where:
#     H_lay_l  = N * Sigma_l  (GPTQ's Hessian, exact, from LayerStats)
#     H_loss_l = true loss Hessian (estimated via L4 HVP, never materialized)

# Also computes:
#     trace(H_loss_l)  — total true curvature of layer l
#     trace(H_lay_l)   — total curvature per GPTQ's approximation (= L1 rung score * N)

# All three use Hutchinson's estimator:
#     tr(A)    = E[v^T A v]      for Rademacher v ~ {+-1}^n
#     ||A||_F² = E[||Av||²]      for Rademacher v ~ {+-1}^n

# A small gap (< 0.1) confirms H_loss ≈ H_lay, validating the ladder design.
# """

# from __future__ import annotations
# import numpy as np
# from typing import Dict, Tuple

# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from gamma_bench.core.structures import LayerStats


# def _hutchinson_trace(matvec, n_dim: int, n_probe: int = 30, seed: int = 0) -> float:
#     """Estimates tr(A) = E[v^T A v] for Rademacher v."""
#     rng = np.random.default_rng(seed)
#     acc = 0.0
#     for _ in range(n_probe):
#         v    = rng.choice([-1.0, 1.0], size=n_dim)
#         acc += float(np.dot(v, matvec(v)))
#     return acc / n_probe


# def _hutchinson_frob_sq(matvec, n_dim: int, n_probe: int = 30, seed: int = 0) -> float:
#     """Estimates ||A||_F² = E[||Av||²] for Rademacher v."""
#     rng = np.random.default_rng(seed)
#     acc = 0.0
#     for _ in range(n_probe):
#         v   = rng.choice([-1.0, 1.0], size=n_dim)
#         Av  = matvec(v)
#         acc += float(np.dot(Av, Av))
#     return acc / n_probe



# def compute_hessian_gap(
#     stats:   Dict[str, LayerStats],
#     n_probe: int = 30,
#     seed:    int = 0,
# ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
#     """
#     For each layer with HVP attached, computes the relative Frobenius
#     approximation error of Eq. 2 in the report:

#         gap_l = ||H_loss_l - H_lay_l||_F / ||H_lay_l||_F

#     where H_lay_l = N * Sigma_l is materialized exactly (no Hutchinson
#     needed on that side — only H_loss is matvec-only).

#     Returns
#     -------
#     gap_by_layer, trace_h_loss_by_layer, trace_n_sigma_by_layer
#     Layers with hvp=None get nan.
#     """
#     gaps, tr_h_loss, tr_n_sigma = {}, {}, {}

#     for layer_id, st in stats.items():
#         n = st.weight_shape[1]   # d_in
#         N = st.n_tokens

#         # H_lay = N * Sigma is already a materialized array — compute
#         # trace and Frobenius norm exactly, no Hutchinson needed here.
#         H_lay  = st.sigma * N
#         tr_ns  = float(np.trace(H_lay))
#         fro_ns = float(np.linalg.norm(H_lay, "fro"))
#         tr_n_sigma[layer_id] = tr_ns

#         def h_lay_mv(v):  return H_lay @ v

#         if st.hvp is None:
#             gaps[layer_id]      = float('nan')
#             tr_h_loss[layer_id] = float('nan')
#             continue

#         # H_loss is never materialized — only matvec access exists,
#         # so trace and the numerator of the gap still need Hutchinson.
#         tr_hl = _hutchinson_trace(st.hvp, n, n_probe, seed)
#         tr_h_loss[layer_id] = tr_hl

#         def diff_mv(v):  return st.hvp(v) - h_lay_mv(v)

#         num_sq = _hutchinson_frob_sq(diff_mv, n, n_probe, seed)
#         gap    = float(np.sqrt(num_sq) / max(fro_ns, 1e-12))

#         gaps[layer_id] = gap
#         print(f"  [gap] {layer_id:50s}  gap={gap:.4f}  "
#               f"tr(H_loss)={tr_hl:.4f}  tr(H_lay)={tr_ns:.2f}")

#     return gaps, tr_h_loss, tr_n_sigma







