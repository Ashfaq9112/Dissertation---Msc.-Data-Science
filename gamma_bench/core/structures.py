"""
gamma_bench.core.structures
============================

The common spine. Every allocation method consumes the SAME LayerStats and
emits the SAME AllocationMap; every AllocationMap is realized by the SAME
Quantizer; every comparison is measured by the SAME GeometryProbe.

This is what makes the crux ablation a *controlled* experiment: the only thing
that varies between methods is the AllocationMap. Rounding, calibration data,
layer selection, and geometry measurement are held fixed.

Design rule encoded here:
  - LayerStats     : the single source of second-order information (Sigma, HVP)
  - AllocationMap  : the ONLY object an allocation method is allowed to vary
  - Quantizer      : fixed realization of an AllocationMap onto weights
  - GeometryProbe  : fixed measurement of CKA / k-NN overlap

Tensors are typed as `Array` (numpy here for a dependency-light skeleton; swap
for torch.Tensor in the real implementation -- the interfaces do not change).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Protocol, runtime_checkable
import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------
# 1. LayerStats : single source of second-order information
# ----------------------------------------------------------------------
@dataclass
class LayerStats:
    """
    Per-layer statistics consumed by EVERY allocation method.

    The same object backs the whole Hessian-approximation ladder (L0-L4) and
    the Gamma-field estimators, so that 'geometry vs second-order' is a fair
    test rather than a comparison of two different underlying matrices.

    Attributes
    ----------
    layer_id     : stable identifier, e.g. "h.5.mlp.c_fc"
    region       : "attn" | "ffn" | "other"  (for APTQ-style region analysis)
    weight_shape : (out_features, in_features)
    sigma        : Sigma = (1/N) X X^T, the empirical input second-moment (L1 object).
    n_tokens     : N used to form sigma (for matched-bits / variance accounting)
    hvp          : optional matrix-free Hessian-vector product action v -> H v.
                   Present only where the L4 near-true-Hessian oracle is built
                   (feasible at GPT-2 scale). None elsewhere.
    eig_cache    : optional cached spectrum of sigma (filled lazily by estimators)
    """
    layer_id: str
    region: str
    weight_shape: tuple[int, int]
    sigma: Array
    n_tokens: int
    hvp: Optional[Callable[[Array], Array]] = None
    eig_cache: Optional[Array] = None

    def spectrum(self) -> Array:
        """Eigenvalues of sigma (cached). Basis-invariant quantities use this."""
        if self.eig_cache is None:
            # symmetric -> eigvalsh; clip tiny negatives from numerical noise
            w = np.linalg.eigvalsh(self.sigma)
            self.eig_cache = np.clip(w, a_min=1e-12, a_max=None)
        return self.eig_cache

    def diag(self) -> Array:
        """Per-coordinate (basis-DEPENDENT) variance -- the L0/L1 diagonal object."""
        return np.clip(np.diag(self.sigma), 1e-12, None)


# ----------------------------------------------------------------------
# 2. AllocationMap : the ONLY thing an allocation method varies
# ----------------------------------------------------------------------
@dataclass
class AllocationMap:
    """
    Output of any allocation method: layer/region -> bits.

    `bits` may be fractional during optimization; `method` and `meta` record
    provenance so results can be grouped/filtered downstream.
    """
    bits: dict[str, float]                 # layer_id -> bits-per-weight
    method: str                            # "gamma_lgeo", "ladder_L1", "awq", ...
    budget: float                          # target average bits-per-weight B
    meta: dict = field(default_factory=dict)

    def effective_bits(self, sizes: dict[str, int]) -> float:
        """Size-weighted average bits = the matched-bits comparison axis."""
        num = sum(self.bits[l] * sizes[l] for l in self.bits)
        den = sum(sizes[l] for l in self.bits)
        return num / den


# ----------------------------------------------------------------------
# 3. Quantizer : fixed realization (held constant across all methods)
# ----------------------------------------------------------------------
@runtime_checkable
class Quantizer(Protocol):
    """
    Applies an AllocationMap to weights. ONE implementation is shared across
    every method so that differences in results are attributable to allocation,
    not to rounding. Swapping the quantizer is a global, declared change.
    """
    def apply(self, weights: dict[str, Array], alloc: AllocationMap
              ) -> dict[str, Array]: ...


class UniformGridQuantizer:
    """Reference symmetric per-tensor uniform quantizer to `bits` levels."""
    def apply(self, weights, alloc):
        out = {}
        for layer_id, W in weights.items():
            b = alloc.bits[layer_id]
            levels = max(2, int(round(2 ** b)))
            scale = np.max(np.abs(W)) / (levels / 2 - 1 + 1e-12)
            q = np.clip(np.round(W / (scale + 1e-12)), -(levels//2), levels//2 - 1)
            out[layer_id] = q * scale
        return out


# ----------------------------------------------------------------------
# 4. GeometryProbe : fixed measurement (shared by every method)
# ----------------------------------------------------------------------
class GeometryProbe:
    """
    Computes the primary geometry-preservation outcomes from cached activations.
    Identical for every method so the geometry comparison is apples-to-apples.
    """

    @staticmethod
    def linear_cka(H1: Array, H2: Array) -> float:
        """Layer-wise CKA between original (H1) and quantized (H2) activations.
        H: (n_tokens, hidden)."""
        H1 = H1 - H1.mean(0, keepdims=True)
        H2 = H2 - H2.mean(0, keepdims=True)
        hsic = np.linalg.norm(H2.T @ H1, 'fro') ** 2
        n1 = np.linalg.norm(H1.T @ H1, 'fro')
        n2 = np.linalg.norm(H2.T @ H2, 'fro')
        return float(hsic / (n1 * n2 + 1e-12))

    @staticmethod
    def knn_overlap(H1: Array, H2: Array, k: int = 10) -> float:
        """Neighborhood preservation = mean Jaccard overlap of k-NN sets.
        FIXED definition (trustworthiness/continuity family). State k, point
        set, layer, corpus wherever this is reported."""
        def knn(H):
            # (n,n) sq distances; argsort drop self
            d = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            return np.argsort(d, axis=1)[:, :k]
        a, b = knn(H1), knn(H2)
        jac = [len(set(ai) & set(bi)) / len(set(ai) | set(bi))
               for ai, bi in zip(a, b)]
        return float(np.mean(jac))
