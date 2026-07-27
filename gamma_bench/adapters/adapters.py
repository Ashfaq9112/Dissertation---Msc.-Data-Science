"""
gamma_bench.adapters
====================

Every tested method is reduced to a small interface contract. The reuse-vs-
reimplement rule is encoded structurally:

  REUSE  (black box)  : run a baseline's own repo to get its native headline
                        number, then expose its effective AllocationMap.
  REIMPLEMENT (thin)  : the Hessian ladder L0-L4 and the Gamma-field, written
                        to ONE allocation interface over shared LayerStats.

Each method family supplies a small set of "hooks" -- the only method-specific
code the harness needs. Everything else (LayerStats, Quantizer, GeometryProbe,
results schema) is shared.

A method may produce results by TWO paths, and reports both where relevant:
  (a) native    : the method's own end-to-end pipeline (honest headline)
  (b) controlled: its AllocationMap -> shared Quantizer (allocation-isolating)
Reporting both pre-empts the "did you weaken the baselines?" objection.
"""

from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable
import numpy as np

from gamma_bench.core.structures import (
    LayerStats, AllocationMap, Array, GeometryProbe,
)


# ----------------------------------------------------------------------
# The universal adapter contract
# ----------------------------------------------------------------------
@runtime_checkable
class MethodAdapter(Protocol):
    """
    The contract EVERY method satisfies. Hooks are intentionally minimal.
    """
    name: str
    family: str          # "gamma" | "ladder" | "rotation" | "vq" | "activation"
    strategy: str        # "reuse" | "reimplement"

    # --- HOOK 1: allocation (the controlled path; required for the crux ablation)
    def allocate(self, stats: dict[str, LayerStats], budget: float
                 ) -> AllocationMap: ...

    # --- HOOK 2: weight pre-transform (identity unless the family rotates/scales)
    def pre_transform(self, weights: dict[str, Array]
                      ) -> tuple[dict[str, Array], dict]: ...

    # --- HOOK 3: native run (the reuse path; returns the method's own numbers)
    #     May be None for reimplemented methods that have no separate "native".
    def native_run(self, weights: dict[str, Array], budget: float
                   ) -> Optional[dict[str, Array]]: ...


# ======================================================================
# Family: GAMMA-FIELD  (reimplement -> reuse Gamma_prac estimators)
# ======================================================================
class GammaFieldAdapter:
    family = "gamma"
    strategy = "reimplement"

    def __init__(self, use_geometry_terms: bool = True, name: str = "gamma_lgeo"):
        # use_geometry_terms=False is the geometry-term ablation (Stage 2):
        # drops kappa_gen / lambda_CL / C(x), leaving only lambda_geo.
        self.use_geometry_terms = use_geometry_terms

        self.name = name

    def _lambda_geo(self, st: LayerStats) -> float:
    
        return float(np.exp(np.mean(np.log(st.spectrum())))
       
        

    def _geometry_modulation(self, st: LayerStats) -> float:
        """Hook point for kappa_gen / lambda_CL / C(x). Stubbed to 1.0 here;
        real Gamma_prac fills this. Ablation sets it identically to 1.0."""
        return 1.0 if not self.use_geometry_terms else 1.0  # TODO: Gamma_prac

    def allocate(self, stats, budget):
        score = {l: self._lambda_geo(st) * self._geometry_modulation(st)
                 for l, st in stats.items()}
        bits = _waterfill(score, stats, budget)
        return AllocationMap(bits, self.name, budget,
                             meta={"geometry_terms": self.use_geometry_terms})

    def pre_transform(self, weights):
        return weights, {}                      # gamma does not rotate weights

    def native_run(self, weights, budget):
        return None                             # no separate native pipeline


# ======================================================================
# Family: HESSIAN LADDER L0-L4  (reimplement to ONE interface)
# ======================================================================
class LadderAdapter:
    """
    Five backends over the SAME LayerStats. Writing them as one family is what
    turns 'Hessian baseline' from an ill-defined single point into a controlled
    approximation axis (L0 crude -> L4 near-true).
    """
    family = "ladder"
    strategy = "reimplement"

    def __init__(self, rung: str):
        assert rung in {"L0", "L1", "L2", "L3", "L4"}
        self.rung = rung
        self.name = f"ladder_{rung}"

    # ------------------------------------------------------------------
    # PER-COORDINATE sensitivity. Returns a length-d_in VECTOR, not a scalar.
    #
    # WHY VECTORS: collapsing each rung to a scalar destroys exactly what
    # distinguishes the rungs. L0/L1/L3 all have the same TRACE -- they differ
    # in STRUCTURE (diagonal vs full-Sigma vs Kronecker). Any trace-based
    # summary makes them algebraically identical:
    #     mean(diag(S)) == trace(S)/n == mean(eigvals(S))
    # The previous scalar implementation collapsed L0 == L1 == L3 exactly, and
    # L2 differed only by a hardcoded 1.5x region constant. See
    # `assert_rungs_discriminate` below, which now fails loudly on collapse.
    # ------------------------------------------------------------------
    def _sensitivity_vec(self, st: LayerStats) -> Array:
        S = st.sigma
        n = S.shape[0]

        if self.rung == "L0":
            # Magnitude / diagonal proxy: per-coordinate variance ONLY.
            # Ignores all off-diagonal (interaction) structure. This is the
            # AWQ-style activation-scale baseline.
            return np.clip(np.diag(S), 1e-12, None)

        if self.rung == "L1":
            # Empirical Fisher / OBS-GPTQ. The optimal-deletion sensitivity is
            #     s_i = 1 / [S^{-1}]_ii
            # which equals the SCHUR COMPLEMENT of coordinate i -- the residual
            # variance of coord i after regressing out all others. This differs
            # from L0 exactly when S has off-diagonal mass, which is the entire
            # content of "second-order beats magnitude".
            Sinv = np.linalg.pinv(S + 1e-10 * np.eye(n))
            return np.clip(1.0 / np.clip(np.diag(Sinv), 1e-12, None), 1e-12, None)

        if self.rung == "L2":
            # Region/block-structured second-order: L1 computed WITHIN blocks
            # (attention heads / FFN sub-blocks) rather than globally. Captures
            # within-block interaction, ignores cross-block. NOT a hardcoded
            # region multiplier -- that was a constant, not second-order structure.
            n_blocks = 12 if st.region == "attn" else 4
            bs = max(1, n // n_blocks)
            out = np.empty(n)
            for start in range(0, n, bs):
                end = min(start + bs, n)
                blk = S[start:end, start:end]
                blk_inv = np.linalg.pinv(blk + 1e-10 * np.eye(end - start))
                out[start:end] = 1.0 / np.clip(np.diag(blk_inv), 1e-12, None)
            return np.clip(out, 1e-12, None)

        if self.rung == "L3":
            # K-FAC: H ~= G (x) A, with A = Sigma the INPUT factor and G the
            # OUTPUT-gradient second moment. The per-coordinate input
            # sensitivity is diag(A), MODULATED by the measured mean output
            # curvature Tr(G)/d_out. That output-side term is information L0/L1
            # structurally CANNOT see -- it is the real reason L3 != L1.
            if st.g_bar is None:
                raise RuntimeError(
                    f"Layer {st.layer_id}: g_bar not computed. "
                    f"Run backward pass in build_layer_stats first."
                )
            g_bar = st.g_bar
            return np.clip(np.diag(S), 1e-12, None) * float(g_bar)

        if self.rung == "L4":
            # Near-true Hessian: per-coordinate Hutchinson DIAGONAL via HVP,
            # with a Rademacher contraction over output channels.
            # Raises RuntimeError if hvp not attached.
            return _hvp_diag_estimate(st)

        raise ValueError(self.rung)
    def _reduce_to_scalar(self, vec: Array) -> float:
        """Discrimination-preserving reduction: vector -> layer scalar via mean."""
        return float(np.mean(vec))

    def allocate(self, stats, budget):
        # Step 1: compute and STORE the full per-coordinate vectors.
        vecs = {l: self._sensitivity_vec(st) for l, st in stats.items()}
    
        # Step 2: reduce each vector to a layer scalar for water-filling.
        score = {l: self._reduce_to_scalar(v) for l, v in vecs.items()}
        bits = _waterfill(score, stats, budget)
        return AllocationMap(bits, self.name, budget, meta={"rung": self.rung})

    def pre_transform(self, weights):
        return weights, {}

    def native_run(self, weights, budget):
        return None



# ======================================================================
# Family: ROTATION  (reuse Q from QuaRot/SpinQuant; allocation on top is ours)
# ======================================================================
class RotationAdapter:
    """
    HOOK: supply the published rotation Q (Hadamard / learned). We apply it via
    pre_transform, then run an inner allocator on the rotated weights. This is
    exactly the H4-H6 interaction experiment: does rotation absorb, complement,
    or antagonize geometric allocation?
    """
    family = "rotation"
    strategy = "reuse"

    def __init__(self, q_provider, inner_allocator, name="quarot+gamma"):
        # q_provider(layer_id) -> orthogonal matrix Q   (from the reused repo)
        # inner_allocator: any MethodAdapter (e.g. GammaFieldAdapter)
        self.q_provider = q_provider
        self.inner = inner_allocator
        self.name = name

    def pre_transform(self, weights):
        rotated, applied = {}, {}
        for l, W in weights.items():
            Q = self.q_provider(l)
            rotated[l] = W @ Q
            applied[l] = Q
        return rotated, {"Q": applied}

    def allocate(self, stats, budget):
        # NOTE: stats here should be computed on ROTATED activations; the
        # harness recomputes LayerStats post-transform before calling allocate.
        amap = self.inner.allocate(stats, budget)
        amap.method = self.name
        amap.meta["rotation"] = True
        return amap

    def native_run(self, weights, budget):
        # The reused repo's own end-to-end QuaRot/SpinQuant result (rotation +
        # its own uniform quantization) -> honest baseline headline.
        return None   # wire to reused repo CLI / API


# ======================================================================
# Family: REUSED PTQ (GPTQ / AWQ / APTQ) and VQ (AQLM)  -- black boxes
# ======================================================================
class ReusedPTQAdapter:
    """
    For GPTQ / AWQ / APTQ / AQLM: run the published repo (native_run) for the
    honest headline, and -- where the method exposes one -- extract its effective
    AllocationMap to also route through the shared Quantizer (controlled path).

    HOOKS the integrator must provide per repo:
      - extract_alloc(weights, budget) -> dict[layer_id -> bits]
            (For AWQ: derive an effective bit/scale map. For AQLM: convert
             codebook config to matched EFFECTIVE bits -- the fiddly adapter.)
      - run_native(weights, budget)   -> dict[layer_id -> quantized weights]
    """
    family = "reused"
    strategy = "reuse"

    def __init__(self, name, extract_alloc, run_native):
        self.name = name
        self._extract = extract_alloc
        self._native = run_native

    def allocate(self, stats, budget):
        bits = self._extract({l: st.weight_shape for l, st in stats.items()},
                             budget)
        return AllocationMap(bits, self.name, budget, meta={"reused": True})

    def pre_transform(self, weights):
        return weights, {}

    def native_run(self, weights, budget):
        return self._native(weights, budget)


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
# def _waterfill(score: dict[str, float], stats, budget: float) -> dict[str, float]:
#     """Monotone allocation: more bits where score (sensitivity/geometry) is high.
#     Normalized so the size-weighted average equals `budget`. Skeleton form."""
#     layers = list(score)
#     s = np.array([score[l] for l in layers], dtype=float)
#     s = np.log1p(s - s.min() + 1e-9)                 # compress dynamic range
#     w = s / (s.sum() + 1e-12)
#     raw = budget + (w - w.mean()) * budget           # spread around budget
#     bits = np.clip(raw, 2.0, 8.0)
#     return {l: float(b) for l, b in zip(layers, bits)}

def _waterfill(score: dict[str, float], stats: dict[str, dict], budget: float,
                bmin: float = 1.0, bmax: float = 8.0,
                max_iter: int = 50) -> dict[str, float]:
    """Reverse water-filling bit allocation.

    Minimizes size-weighted quantization distortion under the high-rate
    approximation D_i ~ score_i * 2^(-2*R_i), subject to:
        sum(n_i * R_i) / sum(n_i) == budget   and   bmin <= R_i <= bmax
    `stats[l]["n_params"]` must give the parameter count for layer l —
    adjust the key if your stats schema differs.
    """
    layers = list(score)
    # log_s = np.log2(np.array([score[l] for l in layers], dtype=float) + 1e-12)
    log_s = np.log2(np.clip(np.array([score[l] for l in layers], dtype=float), 1e-12, None))
    # n = np.array([stats[l]["n_params"] for l in layers], dtype=float)
    n = np.array([stats[l].weight_shape[0] * stats[l].weight_shape[1] for l in layers], dtype=float)

    bits = np.full(len(layers), budget, dtype=float)
    free = np.ones(len(layers), dtype=bool)
    total_n = n.sum()

    for _ in range(max_iter):
        if not free.any():
            break
        committed = (n[~free] * bits[~free]).sum()
        remaining_total = budget * total_n - committed

        n_free = n[free]
        w = n_free / n_free.sum()
        mean_log_s = (w * log_s[free]).sum()
        K = remaining_total / n_free.sum() - 0.5 * mean_log_s

        bits[free] = K + 0.5 * log_s[free]

        over = free & (bits > bmax)
        under = free & (bits < bmin)
        if not (over.any() or under.any()):
            break
        bits[over] = bmax
        bits[under] = bmin
        free &= ~over & ~under

    bits = np.clip(bits, bmin, bmax)
    return {l: float(b) for l, b in zip(layers, bits)}





def _hvp_diag_estimate(st: LayerStats, n_probe: int = 64,
                       rng: Optional[np.random.Generator] = None) -> Array:
    """Hutchinson DIAGONAL estimate of the near-true Hessian via the HVP action.

        Returns a length-d_in vector: diag(H) ~= E[ v * (H v) ] with Rademacher v,
        since E[v_i v_j] = delta_ij.

        Raises RuntimeError if hvp is not attached or doesn't accept (v, u) signature.
    """
    n = st.weight_shape[1]
    if st.hvp is None:
        raise RuntimeError(
            f"Layer {st.layer_id}: hvp not attached. "
            f"Cannot compute L4 without HVP. Attach via attach_hvp() first."
        )

    rng = rng or np.random.default_rng()
    d_out = st.weight_shape[0]
    acc = np.zeros(n)
    for _ in range(n_probe):
        v = rng.choice([-1.0, 1.0], size=n)          # input-space probe
        u = rng.choice([-1.0, 1.0], size=d_out)      # OUTPUT-space Rademacher
        try:
            Hv = st.hvp(v, u)          # preferred signature: contracts with u
        except TypeError as e:
            raise RuntimeError(
            f"Layer {st.layer_id}: hvp must accept (v, u) signature "
            f"for correct output-side Rademacher contraction. Got: {e}"
        )
                       # legacy: all-ones broadcast (BIASED)
        acc += v * np.asarray(Hv).ravel()[:n]
    return np.clip(acc / n_probe, 1e-12, None)


# ----------------------------------------------------------------------
# GUARD: named-distinct estimators must compute distinct values
# ----------------------------------------------------------------------
def assert_rungs_discriminate(stats: dict[str, LayerStats],
                              rungs=("L0", "L1", "L2", "L3", "L4"),
                              tol: float = 1e-9) -> dict:
    """Two names computing one number is a DEFECT, not a finding.

    Run this at harness startup. It would have caught L0 == L1 == L3 the first
    time the ladder ran. Raises on collapse; returns the pairwise distances.
    """
    import itertools
    vecs = {r: {l: LadderAdapter(r)._sensitivity_vec(st)
                for l, st in stats.items()} for r in rungs}
    report, collapsed = {}, []
    for a, b in itertools.combinations(rungs, 2):
        d = max(float(np.max(np.abs(vecs[a][l] - vecs[b][l])))
                for l in stats)
        report[f"{a}_vs_{b}"] = d
        if d <= tol:
            collapsed.append((a, b))
    if collapsed:
        raise AssertionError(
            "LADDER COLLAPSE -- these rungs compute identical sensitivities and "
            f"are therefore ONE estimator under several names: {collapsed}. "
            "Any 'ladder-dependence' result from this configuration is invalid.")
    return report




# def _hvp_trace_estimate(st: LayerStats, n_probe: int = 50, seed: int = 0) -> float:
#     """Hutchinson trace estimate of the near-true Hessian via the HVP action,
#     rescaled to match L1's convention (unscaled trace(Sigma)/d_in)."""
#     if st.hvp is None:
#         raise ValueError(
#             f"L4 requires hvp on layer '{st.layer_id}', but it is None. "
#             f"Call stage0.hvp.attach_hvp(stats, model, input_ids, device) "
#             f"on this LayerStats dict before allocating with rung L4."
#         )
#     n = st.weight_shape[1]
#     rng = np.random.default_rng(seed)
#     acc = 0.0
#     for _ in range(n_probe):
#         v = rng.choice([-1.0, 1.0], size=n)
#         acc += float(v @ st.hvp(v))
#     raw = acc / (n_probe * n)
#     return max(raw / st.n_tokens, 1e-8)
