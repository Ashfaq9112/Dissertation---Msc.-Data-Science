"""
gamma_bench.results.schema
==========================

THE COMMON OUTPUT SHELL. Every experiment -- every method x model x bit-width x
seed -- writes one RunRecord with an identical schema. Statistics and
visualizations are built ONLY from RunRecords, so adding a method or a stage
never changes downstream tooling.

The schema is deliberately keyed so that the pre-registered hypotheses
(H1-H6) and the operator tiers (T1-T3) can be evaluated by simple groupbys:

  H1/H2/H3  : group by (model, budget); compare method='gamma_*' vs 'ladder_L0..L4'
  H4/H5/H6  : compare rotation+uniform vs rotation+gamma (meta['rotation'])
  T1        : capacity fields (footprint, max_batch, tokens_per_gpu)
  T2        : roofline fields (arith_intensity, bound_regime, energy_model)
  T3        : regress task_acc on cka / knn across budgets
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json, hashlib, datetime


@dataclass
class PerfMetrics:
    ppl_wikitext2: Optional[float] = None
    ppl_c4: Optional[float] = None
    zeroshot: dict[str, float] = field(default_factory=dict)   # task -> acc
    effective_bits: Optional[float] = None
    compression_ratio: Optional[float] = None


@dataclass
class GeometryMetrics:
    cka_by_layer: dict[str, float] = field(default_factory=dict)
    knn_overlap_by_layer: dict[str, float] = field(default_factory=dict)
    cka_mean: Optional[float] = None
    knn_mean: Optional[float] = None


@dataclass
class OperatorMetrics:
    # --- T1 measured ---
    footprint_bytes: Optional[int] = None
    max_batch: Optional[int] = None
    tokens_per_gpu: Optional[float] = None
    # --- T2 modeled (roofline; ALWAYS carries its assumptions) ---
    arith_intensity: Optional[float] = None          # FLOP / byte
    bound_regime: Optional[str] = None               # "memory" | "compute"
    energy_per_token_model: Optional[float] = None    # under kernel_assumption
    kernel_assumption: Optional[str] = None           # the explicit label
    # --- T3 projected (filled by analysis, not by a single run) ---
    usable_bits_at_quality_floor: Optional[float] = None


@dataclass
class HessianLadderProbe:
    """Stage-0/1 bonus measurement: how good is H ~= Sigma on real models."""
    h_minus_sigma_norm: Optional[float] = None        # ||H - Sigma|| normalized
    rung: Optional[str] = None
    # Cross-layer summary of stage0/gap.py::compute_hessian_gap's per-layer output.
    # cos_theta is the scale-free PRIMARY answer to "does Sigma have the right geometry";
    # scale_ratio/scale_ratio_N are the two rival N-factor conventions, kept side by side
    # so the open convention question is settled empirically, not assumed (see gap.py
    # module docstring); frac_neg_quad flags when a low cos_theta is actually just
    # Hessian-indefinite-vs-Sigma-PSD structural mismatch rather than a real gap.
    cos_theta_mean: Optional[float] = None
    cos_theta_min: Optional[float] = None
    scale_ratio_median: Optional[float] = None
    scale_ratio_N_median: Optional[float] = None
    frac_neg_quad_mean: Optional[float] = None


@dataclass
class RunRecord:
    # --- identity / grouping keys ---
    run_id: str
    stage: int
    model: str
    method: str
    family: str
    budget: float                  # target average bits
    seed: int
    rotation: bool = False
    geometry_terms: bool = True
    # --- payloads ---
    perf: PerfMetrics = field(default_factory=PerfMetrics)
    geom: GeometryMetrics = field(default_factory=GeometryMetrics)
    operator: OperatorMetrics = field(default_factory=OperatorMetrics)
    hessian_probe: HessianLadderProbe = field(default_factory=HessianLadderProbe)
    alloc_bits: dict[str, float] = field(default_factory=dict)
    sensitivity_scores: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat())

    @staticmethod
    def make_id(model, method, budget, seed, rotation, geometry_terms) -> str:
        key = f"{model}|{method}|{budget}|{seed}|{rotation}|{geometry_terms}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ----------------------------------------------------------------------
# append-only results store: one JSONL file, one line per RunRecord
# ----------------------------------------------------------------------
class ResultsStore:
    def __init__(self, path: str):
        self.path = path

    def append(self, rec: RunRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def load(self) -> list[dict]:
        try:
            with open(self.path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
