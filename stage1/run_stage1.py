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
import os, sys , psutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma_bench.core.structures import UniformGridQuantizer
from gamma_bench.adapters.adapters import GammaFieldAdapter, LadderAdapter, assert_rungs_discriminate
from gamma_bench.results.schema import (
    RunRecord, PerfMetrics, GeometryMetrics,
    HessianLadderProbe, ResultsStore,
)
from gamma_bench.results import analysis

from stage0.capture   import build_layer_stats
from stage0.hvp       import attach_hvp
from stage0.baselines import measure_geometry, evaluate_perplexity
from stage1_analysis import run_stage1_analysis
# from stage1.kfac                import attach_kfac_trace
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
    seeds       = [0,1,2],
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'stage1'),
)

STORE_PATH = os.path.join(CFG['results_dir'], 'stage1_results.jsonl')
FIG_DIR    = os.path.join(CFG['results_dir'], 'figures')


def build_methods():
    return [GammaFieldAdapter(use_geometry_terms=False)] + \
           [LadderAdapter(r) for r in ("L0", "L1", "L2", "L3", "L4")]

def validate_rungs(stats: Dict[str, LayerStats]):
    """
    Run all five rungs on every layer. Check:
    1. Each rung produces a vector of the right length (d_in)
    2. No NaN/Inf values
    3. Rungs produce DIFFERENT vectors (not collapsed)
    4. g_bar is real (not None)
    5. hvp is attached
    """
    rungs = ["L0", "L1", "L2", "L3", "L4"]
    
    for layer_id, st in stats.items():
        print(f"\n--- {layer_id} ---")
        print(f"  shape: {st.weight_shape}, region: {st.region}")
        print(f"  g_bar: {st.g_bar}")
        print(f"  hvp attached: {st.hvp is not None}")
        
        vecs = {}
        for rung in rungs:
            adapter = LadderAdapter(rung)
            vec = adapter._sensitivity_vec(st)
            
            d_in = st.weight_shape[1]
            assert vec.shape == (d_in,), f"{rung}: expected ({d_in},), got {vec.shape}"
            assert np.all(np.isfinite(vec)), f"{rung}: contains NaN/Inf"
            assert np.all(vec > 0), f"{rung}: contains non-positive values"
            
            vecs[rung] = vec
            print(f"  {rung}: min={vec.min():.6e}  mean={vec.mean():.6e}  max={vec.max():.6e}")
        
        # Check rungs are distinct
        for r1, r2 in [("L0", "L1"), ("L1", "L2"), ("L0", "L3"), ("L3", "L4")]:
            corr = np.corrcoef(vecs[r1], vecs[r2])[0, 1]
            ratio = np.mean(vecs[r1]) / np.mean(vecs[r2])
            print(f"  {r1} vs {r2}: corr={corr:.4f}, mean_ratio={ratio:.4f}")
        
        # L0 vs L3: should differ by g_bar scaling
        if st.g_bar is not None:
            expected_ratio = 1.0 / st.g_bar
            actual_ratio = np.mean(vecs["L0"]) / np.mean(vecs["L3"])
            print(f"  L0/L3 ratio={actual_ratio:.4f}, expected 1/g_bar={expected_ratio:.4f}")





def mem_gb(): return psutil.Process(os.getpid()).memory_info().rss / 1e9




def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    store = ResultsStore(STORE_PATH)
    open(STORE_PATH, 'w').close()   # fresh run

    quantizer = UniformGridQuantizer()
    budget = CFG['budget_bits']

    for model_name in CFG['models']:
        for seed in CFG['seeds']:
            print(f"\n{'=' * 60}\n{model_name}  seed={seed}\n{'=' * 60}")

            print("[1/3] Building LayerStats from C4...")
            stats, model, tokenizer, input_ids = build_layer_stats(
                model_name=model_name, n_samples=CFG['n_samples'],
                seq_len=CFG['seq_len'], batch_size=CFG['batch_size'],
                device=CFG['device'], seed=seed,
            )
            print(f"      {len(stats)} layers found.")
            print(f"[after build_layer_stats] RAM: {mem_gb():.2f} GB")
            print("2/3 Attaching L4 HVP...")
            attach_hvp(stats, model, input_ids, CFG['device'],
                       n_batches=20, batch_size=2)
            print(f"[after attach_hvp] RAM: {mem_gb():.2f} GB")
            # print("Validating ladder rung discrimination...")
            # rung_report = assert_rungs_discriminate(stats)
            # for pair, dist in rung_report.items():
            #     if isinstance(dist, bool):
            #         print(f"  {pair}: {dist}")
            #     else:
            #         print(f"  {pair}: max_abs_diff = {dist:.6e}")
            # print("Ladder rungs validated — no forbidden collapses.\n")
            print("Running controlled path per method...")
            sizes = {l: st.weight_shape[0] * st.weight_shape[1] for l, st in stats.items()}
            layer_ids = list(stats.keys())
            ppl_orig = evaluate_perplexity(model, tokenizer, CFG['device'])
            print(f"      Original PPL (WikiText-2): {ppl_orig:.2f}")

            for mth in build_methods():
                print("Allocating Bits")
                alloc = mth.allocate(stats, budget)
                print(f"[{mth.name} after allocate] RAM: {mem_gb():.2f} GB")
                print("Quantizing the model")
                model_q = apply_controlled_quantization(model, alloc, quantizer, CFG['device'])
                print(f"[{mth.name} after quantize] RAM: {mem_gb():.2f} GB")
                print("Saving the  model")
                save_dir = os.path.join(CFG['results_dir'],f"seed{seed}", model_name.replace("/", "_"), mth.name, f"budget_{budget}")
                os.makedirs(save_dir, exist_ok=True)
                model_q.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print("Model Saved")
                print("Measuring Geometic methods")
                model_q.to('cpu')
                torch.cuda.empty_cache()
                geometry_layer_ids = [l for l in layer_ids if not l.endswith('lm_head')]
                cka, knn = measure_geometry(
                    model, model_q, input_ids, geometry_layer_ids, CFG['device'], seed=seed,
                )
                print("Evaluating Perplexity")
                ppl_q = evaluate_perplexity(model_q, tokenizer, CFG['device'])
                # eff = alloc.effective_bits(sizes)
                sensitivity = alloc.meta.get("sensitivity", {})
                print("Storing records")
                rec = RunRecord(
                    run_id=RunRecord.make_id(
                        model_name, mth.name, budget, seed, False,
                        getattr(mth, "use_geometry_terms", False),
                    ),
                    stage=1, model=model_name, method=mth.name,
                    family=mth.family, budget=budget, seed=seed,
                    rotation=False,
                    geometry_terms=getattr(mth, "use_geometry_terms", True),
                    perf=PerfMetrics(ppl_wikitext2=ppl_q),
                    geom=GeometryMetrics(
                        cka_by_layer=cka, knn_overlap_by_layer=knn,
                        cka_mean=float(np.mean(list(cka.values()))) if cka else None,
                        knn_mean=float(np.mean(list(knn.values()))) if knn else None,
                    ),
                    hessian_probe=HessianLadderProbe(rung=getattr(mth, "rung", None)),
                    alloc_bits=alloc.bits,
                    sensitivity_scores=sensitivity,
                    meta={"ppl_original": ppl_orig},
                )
                store.append(rec)
                print(f"      [{mth.name:12s}]  "
                      f"PPL={ppl_q:.2f}  CKA={rec.geom.cka_mean:.4f}  "
                      f"kNN={rec.geom.knn_mean:.4f}")

                del model_q
                torch.cuda.empty_cache()

    # -- crux ladder table + figure, per model --
    print('Running Analysis')
    recs = store.load()
    run_stage1_analysis(
        store=store,
        recs=recs,
        cfg=CFG,
        store_path=STORE_PATH,
        fig_dir=FIG_DIR,
    )

    print(f"\nDone.\n  Results -> {STORE_PATH}\n  Figures -> {FIG_DIR}/")


if __name__ == '__main__':
    main()