"""
stage0/run_stage0.py
====================
Stage 0 orchestrator. Run from inside gamma_bench/ project root:

    python stage0/run_stage0.py

Pipeline:
  1. Build real LayerStats from GPT-2 Small + C4
  2. Attach L4 HVP to LayerStats
  3. Compute gap ||H_loss - H_lay|| per layer + traces -> save in meta + .npy
  4. Reproduce GPTQ + AWQ (native path) using same calibration data
  5. Measure CKA + kNN using professor's GeometryProbe
  6. Write RunRecords to JSONL (gap data stored in meta field)
  7. Generate figures: pareto + per-layer gap bar chart
"""

from __future__ import annotations
import os, sys, pickle
import numpy as np
import copy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma_bench.results.schema import (
    RunRecord, PerfMetrics, GeometryMetrics,
    HessianLadderProbe, ResultsStore,
)
from gamma_bench.results import analysis

from stage0.capture   import build_layer_stats
from stage0.hvp       import attach_hvp
from stage0.gap       import compute_hessian_gap
from stage0.baselines import run_gptq, measure_geometry, evaluate_perplexity

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    model_name  = 'gpt2',
    device      = 'cuda' if torch.cuda.is_available() else 'cpu',
    n_samples   = 512,
    seq_len     = 128,
    batch_size  = 8,
    budget_bits = 4.0,
    seeds       = [0, 1, 2],
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'stage0'),
)

STORE_PATH = os.path.join(CFG['results_dir'], 'stage0_results.jsonl')
FIG_DIR    = os.path.join(CFG['results_dir'], 'figures')


def _save_matrices(stats, seed: int, results_dir: str) -> str:
    """Save H_lay (= N * Sigma) per layer as .npy files. Returns directory path."""
    matrices_dir = os.path.join(results_dir, 'matrices', f'seed{seed}')
    os.makedirs(matrices_dir, exist_ok=True)
    for layer_id, st in stats.items():
        h_lay = st.sigma * st.n_tokens          # H_lay = N * Sigma  (exact)
        fname = layer_id.replace('.', '_') + '_hlay.npy'
        np.save(os.path.join(matrices_dir, fname), h_lay)
    return matrices_dir

def _sanitise(d: dict) -> dict:
    """Replace float nan with None so JSON stays valid."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in d.items()}


def _plot_gap_by_layer(gaps: dict, seed: int, fig_dir: str):
    """Bar chart of normalized gap per layer — the Stage 0 gate figure."""
    layers = list(gaps.keys())
    values = [gaps[l] if not np.isnan(gaps[l]) else 0.0 for l in layers]
    short  = ['.'.join(l.split('.')[-2:]) for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.4), 4))
    ax.bar(range(len(layers)), values, color='steelblue', alpha=0.8)
    ax.axhline(0.1, color='red', linestyle='--', lw=1, label='threshold 0.1')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('||H_loss - H_lay||_F  /  ||H_lay||_F')
    ax.set_title(f'Stage 0 — Hessian gap per layer (seed={seed})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stage0_gap_seed{seed}.png'), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Stage 0  —  GPT-2 Small baseline reproduction")
    print("=" * 60)

    store = ResultsStore(STORE_PATH)
    open(STORE_PATH, 'w').close()   # fresh run

    for seed in CFG['seeds']:
        print(f"\n{'─' * 55}  seed={seed}")

        # 1. Real LayerStats
        print("\n[1/4] Building LayerStats from C4 activations...")
        stats, model, tokenizer, input_ids = build_layer_stats(
            model_name  = CFG['model_name'],
            n_samples   = CFG['n_samples'],
            seq_len     = CFG['seq_len'],
            batch_size  = CFG['batch_size'],
            device      = CFG['device'],
            seed        = seed,
        )
        print(f"      {len(stats)} layers found.")

        
        # 2. Attach L4 HVP
        print("\n[2/4] Attaching L4 HVP...")
        attach_hvp(stats, model, input_ids, CFG['device'], n_batches=10, batch_size=2)

        # 3. Hessian gap + traces
        print("\n[3/4] Computing hessian gap per layer...")
        gaps, tr_h_loss, tr_n_sigma = compute_hessian_gap(stats, n_probe=30, seed=seed)

        gap_values = [v for v in gaps.values() if not np.isnan(v)]
        gap_mean   = float(np.mean(gap_values))
        gap_std    = float(np.std(gap_values))
        print(f"      Mean gap = {gap_mean:.4f} +/- {gap_std:.4f}")

        # save H_lay matrices as .npy
        matrices_dir = _save_matrices(stats, seed, CFG['results_dir'])

        # gap bar chart
        _plot_gap_by_layer(gaps, seed, FIG_DIR)

        # 4. GPTQ + AWQ native path
        print("\n[4/4] Reproducing GPTQ and AWQ...")
        layer_ids = list(stats.keys())
        ppl_orig  = evaluate_perplexity(model, tokenizer, CFG['device'])
        print(f"      Original GPT-2 PPL (WikiText-2): {ppl_orig:.2f}")

        for method, run_fn in [('gptq', run_gptq)]:
            model_q, ppl_q = run_fn(
                CFG['model_name'], CFG['device'], input_ids, bits=4
            )
            CKPT_DIR = os.path.join(CFG['results_dir'], 'checkpoints', f'seed{seed}')
            os.makedirs(CKPT_DIR, exist_ok=True)
            
            model_q.save_pretrained(os.path.join(CKPT_DIR, 'gptq_model'))
            from auto_gptq import AutoGPTQForCausalLM as _AGPTQ
            model_q = _AGPTQ.from_quantized(
                os.path.join(CKPT_DIR, 'gptq_model'), device='cuda:0'
            )

            # strip hvp closures — can't pickle local functions
            stats_pkl = {}
            for lid, st in stats.items():
                st_copy = copy.copy(st)
                st_copy.hvp = None
                stats_pkl[lid] = st_copy
            
            with open(os.path.join(CKPT_DIR, 'checkpoint.pkl'), 'wb') as f:
                pickle.dump({
                    'stats':       stats_pkl,
                    'gaps':        gaps,
                    'tr_h_loss':   tr_h_loss,
                    'tr_n_sigma':  tr_n_sigma,
                    'gap_mean':    gap_mean,
                    'gap_std':     gap_std,
                    'input_ids':   input_ids.cpu(),
                    'layer_ids':   layer_ids,
                    'ppl_q':       ppl_q,
                    'ppl_orig':    ppl_orig,
                }, f)
            print(f"Checkpoint saved to {CKPT_DIR}")
            cka, knn = {}, {}
            if model_q is not None:
                print("Calling measure_geometry")
                cka, knn = measure_geometry(model, model_q, input_ids, layer_ids, CFG['device'], seed=seed)
            print(f"  CKA mean: {np.mean(list(cka.values())):.4f}  kNN mean: {np.mean(list(knn.values())):.4f}")

            print("Running RunRecord")
            rec = RunRecord(
                run_id = RunRecord.make_id('gpt2', 'gptq', 4.0, seed, False, True),
                stage=0, model='gpt2', method='gptq', family='reused', budget=4.0, seed=seed,
                perf=PerfMetrics(ppl_wikitext2=ppl_q, effective_bits=4.0),
                geom=GeometryMetrics(
                    cka_by_layer=cka, knn_overlap_by_layer=knn,
                    cka_mean=float(np.mean(list(cka.values()))) if cka else None,
                    knn_mean=float(np.mean(list(knn.values()))) if knn else None,
                ),
                hessian_probe=HessianLadderProbe(h_minus_sigma_norm=gap_mean),
                meta={
                    'gap_by_layer':          _sanitise(gaps),
                    'gap_mean':              gap_mean,
                    'gap_std':               gap_std,
                    'trace_h_loss_by_layer': _sanitise(tr_h_loss),
                    'trace_n_sigma_by_layer':_sanitise(tr_n_sigma),
                    'matrices_dir':          matrices_dir,
                    'native_path':           True,
                    'ppl_original':          ppl_orig,
                },
            )
            store.append(rec)
            print(f"  Record written: PPL={ppl_q:.2f}")
            
            del model, model_q
            torch.cuda.empty_cache()

    # figures from records
    recs = store.load()
    if recs:
        df  = analysis.to_frame(recs)
        fig = analysis.plot_pareto(df, CFG['model_name'])
        fig.savefig(os.path.join(FIG_DIR, 'stage0_pareto.png'), dpi=120)
        plt.close('all')

    print(f"\nDone.")
    print(f"  Results  -> {STORE_PATH}")
    print(f"  Figures  -> {FIG_DIR}/")
    print(f"  Matrices -> {CFG['results_dir']}/matrices/")


if __name__ == '__main__':
    main()
