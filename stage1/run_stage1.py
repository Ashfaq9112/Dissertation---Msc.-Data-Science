"""
stage1/run_stage1.py
=====================
Stage 1 orchestrator -- the crux ablation. Run from the project root:

    python stage1/run_stage1.py

Pipeline (per model in {gpt2, gpt2-medium}, per seed in {0,1,2}):
  1. Build real LayerStats from C4 (stage0.capture -- unchanged, reused)
  2. Attach L4 HVP (stage0.hvp -- unchanged, reused)
  3. Attach L3 trace(G) (stage1.kfac -- new)
  4. Run Gamma-field + Ladder L0-L4 over the SAME LayerStats (six methods)
  5. For each method: allocate -> quantize a copy of the real model via the
     controlled path (stage1.quantize_controlled) -> measure CKA/kNN against
     the untouched original (stage0.baselines.measure_geometry, unchanged) ->
     evaluate perplexity (stage0.baselines.evaluate_perplexity, unchanged)
  6. Write one RunRecord per (model, seed, method) to JSONL
  7. Build the crux ladder table + figure from records alone
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma_bench.core.structures import UniformGridQuantizer
from gamma_bench.adapters.adapters import GammaFieldAdapter, LadderAdapter
from gamma_bench.results.schema import (
    RunRecord, PerfMetrics, GeometryMetrics,
    HessianLadderProbe, ResultsStore,
)
from gamma_bench.results import analysis

from stage0.capture   import build_layer_stats
from stage0.hvp       import attach_hvp
from stage0.baselines import measure_geometry, evaluate_perplexity
from stage1.kfac                import attach_kfac_trace
from stage1.quantize_controlled import apply_controlled_quantization

# -- Config ------------------------------------------------------------------
CFG = dict(
    models      = ['gpt2'],
    # 'gpt2-medium'
    device      = 'cuda' if torch.cuda.is_available() else 'cpu',
    n_samples   = 512,
    seq_len     = 128,
    batch_size  = 8,
    budget_bits = 4.0,
    seeds       = [0],
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'stage1'),
)

STORE_PATH = os.path.join(CFG['results_dir'], 'stage1_results.jsonl')
FIG_DIR    = os.path.join(CFG['results_dir'], 'figures')


def build_methods():
    return [GammaFieldAdapter(use_geometry_terms=True)] + \
           [LadderAdapter(r) for r in ("L0", "L1", "L2", "L3", "L4")]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    store = ResultsStore(STORE_PATH)
    open(STORE_PATH, 'w').close()   # fresh run

    quantizer = UniformGridQuantizer()
    budget = CFG['budget_bits']

    for model_name in CFG['models']:
        for seed in CFG['seeds']:
            print(f"\n{'=' * 60}\n{model_name}  seed={seed}\n{'=' * 60}")

            print("[1/4] Building LayerStats from C4...")
            stats, model, tokenizer, input_ids = build_layer_stats(
                model_name=model_name, n_samples=CFG['n_samples'],
                seq_len=CFG['seq_len'], batch_size=CFG['batch_size'],
                device=CFG['device'], seed=seed,
            )
            print(f"      {len(stats)} layers found.")

            print("[2/4] Attaching L4 HVP...")
            attach_hvp(stats, model, input_ids, CFG['device'],
                       n_batches=10, batch_size=2)

            print("[3/4] Attaching L3 trace(G)...")
            attach_kfac_trace(stats, model, input_ids, CFG['device'],
                               n_batches=10, batch_size=8, seed=seed)

            print("[4/4] Running controlled path per method...")
            sizes = {l: st.weight_shape[0] * st.weight_shape[1] for l, st in stats.items()}
            layer_ids = list(stats.keys())
            ppl_orig = evaluate_perplexity(model, tokenizer, CFG['device'])
            print(f"      Original PPL (WikiText-2): {ppl_orig:.2f}")

            for mth in build_methods():
                alloc = mth.allocate(stats, budget)
                model_q = apply_controlled_quantization(model, alloc, quantizer, CFG['device'])

                cka, knn = measure_geometry(
                    model, model_q, input_ids, layer_ids, CFG['device'], seed=seed,
                )
                ppl_q = evaluate_perplexity(model_q, tokenizer, CFG['device'])
                eff = alloc.effective_bits(sizes)

                rec = RunRecord(
                    run_id=RunRecord.make_id(
                        model_name, mth.name, budget, seed, False,
                        getattr(mth, "use_geometry_terms", True),
                    ),
                    stage=1, model=model_name, method=mth.name,
                    family=mth.family, budget=budget, seed=seed,
                    rotation=False,
                    geometry_terms=getattr(mth, "use_geometry_terms", True),
                    perf=PerfMetrics(ppl_wikitext2=ppl_q, effective_bits=eff),
                    geom=GeometryMetrics(
                        cka_by_layer=cka, knn_overlap_by_layer=knn,
                        cka_mean=float(np.mean(list(cka.values()))) if cka else None,
                        knn_mean=float(np.mean(list(knn.values()))) if knn else None,
                    ),
                    hessian_probe=HessianLadderProbe(rung=getattr(mth, "rung", None)),
                    alloc_bits=alloc.bits,
                    meta={"ppl_original": ppl_orig},
                )
                store.append(rec)
                print(f"      [{mth.name:12s}] eff_bits={eff:.2f}  "
                      f"PPL={ppl_q:.2f}  CKA={rec.geom.cka_mean:.4f}  "
                      f"kNN={rec.geom.knn_mean:.4f}")

                del model_q
                torch.cuda.empty_cache()

    # -- crux ladder table + figure, per model --
    recs = store.load()
    if recs:
        df = analysis.to_frame(recs)
        for model_name in CFG['models']:
            crux = analysis.crux_ladder_table(df, model_name, budget)
            print(f"\n=== Crux ladder table -- {model_name} ===")
            print(crux[["method", "cka_mean_mean", "cka_mean_std",
                        "knn_mean_mean", "ppl_wikitext2_mean"]].to_string(index=False))
            fig = analysis.plot_ladder_geometry(crux)
            fig.savefig(os.path.join(FIG_DIR, f'stage1_crux_{model_name}.png'), dpi=120)
            plt.close(fig)

    print(f"\nDone.\n  Results -> {STORE_PATH}\n  Figures -> {FIG_DIR}/")


if __name__ == '__main__':
    main()