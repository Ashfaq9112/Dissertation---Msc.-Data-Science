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
  7. Generate figures: pareto, per-seed gap-metric plots, cross-seed by-layer-type plots
     (gap metrics + CKA/kNN, ported from stage0_analysis.ipynb) — all raw values, no
     pass/fail thresholds baked in; interpretation happens later, separately.
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
from stage0.gap       import compute_hessian_gap, report_gap
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
    """Replace float nan with None (recursively) so JSON stays valid. gap_res is now
    {layer: {field: value}}, not {layer: value} — recurse into nested dicts."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return None if isinstance(v, float) and np.isnan(v) else v
    return {k: clean(v) for k, v in d.items()}


def _layer_type_groups(layer_ids):
    """Same grouping stage0_analysis.ipynb already uses for CKA/kNN — reused here so the
    new gap-metric plots are visually comparable to the existing geometry ones. NOT the
    same as LayerStats.region (attn/ffn/other): this splits c_proj out as its own group
    regardless of which submodule it's under."""
    return {
        'c_attn': [l for l in layer_ids if l.endswith('attn.c_attn')],
        'c_proj': [l for l in layer_ids if l.endswith('attn.c_proj') or l.endswith('mlp.c_proj')],
        'c_fc':   [l for l in layer_ids if l.endswith('mlp.c_fc')],
    }


# ── Per-seed gap-metric plots — raw values only, no pass/fail lines ────────────────────

def _plot_shape(gap_res: dict, seed: int, fig_dir: str):
    """cos_theta (shape match) and frac_neg_quad (structural indefiniteness), same
    x-axis, so the two can be visually compared per layer."""
    layers = sorted(gap_res.keys())
    cos    = [gap_res[l]['cos_theta'] for l in layers]
    neg    = [gap_res[l]['frac_neg_quad'] for l in layers]
    short  = ['.'.join(l.split('.')[-2:]) for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(layers) * 0.4), 7), sharex=True)
    ax1.bar(range(len(layers)), cos, color='steelblue', alpha=0.8)
    ax1.set_ylabel('cos_theta')
    ax1.set_title(f'Stage 0 — shape match per layer (seed={seed})')

    ax2.bar(range(len(layers)), neg, color='indianred', alpha=0.8)
    ax2.set_ylabel('frac_neg_quad')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(short, rotation=45, ha='right', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stage0_shape_seed{seed}.png'), dpi=120)
    plt.close(fig)


def _plot_scale(gap_res: dict, seed: int, fig_dir: str):
    """scale_ratio vs scale_ratio_N per layer — both N-factor conventions shown side by
    side, neither preferred (see gap.py's OPEN CONVENTION QUESTION)."""
    layers = sorted(gap_res.keys())
    sr     = [gap_res[l]['scale_ratio']   for l in layers]
    srN    = [gap_res[l]['scale_ratio_N'] for l in layers]
    short  = ['.'.join(l.split('.')[-2:]) for l in layers]

    x = np.arange(len(layers))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.5), 4))
    ax.bar(x - w / 2, [v if v is not None else np.nan for v in sr],  width=w,
           label='scale_ratio',   color='steelblue', alpha=0.8)
    ax.bar(x + w / 2, [v if v is not None else np.nan for v in srN], width=w,
           label='scale_ratio_N', color='darkorange', alpha=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('scale ratio (log)')
    ax.set_title(f'Stage 0 — N-factor convention comparison per layer (seed={seed})')
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stage0_scale_seed{seed}.png'), dpi=120)
    plt.close(fig)


def _plot_gap_opt(gap_res: dict, seed: int, fig_dir: str):
    """gap_opt = sqrt(1 - cos_theta^2), the minimum achievable relative gap per layer."""
    layers = sorted(gap_res.keys())
    values = [gap_res[l]['gap_opt'] for l in layers]
    short  = ['.'.join(l.split('.')[-2:]) for l in layers]

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.4), 4))
    ax.bar(range(len(layers)), values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('gap_opt = sqrt(1 - cos_theta^2)')
    ax.set_title(f'Stage 0 — minimum achievable relative gap per layer (seed={seed})')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stage0_gap_opt_seed{seed}.png'), dpi=120)
    plt.close(fig)


# ── Cross-seed "by layer-type" plots, built once from all records ──────────────────────
# Same grouping + subplot layout as stage0_analysis.ipynb's existing CKA/kNN figures
# (Analysis 2 / Analysis 3 cells), ported here so they're generated automatically
# instead of requiring a manual notebook run, and extended with the same layout for
# the new gap metrics.

def _plot_gap_by_layer_type(recs: list, fig_dir: str):
    layer_ids = list(recs[0]['meta']['gap_by_layer'].keys())
    groups = _layer_type_groups(layer_ids)
    group_names = {'c_attn': 'c_attn (attn input proj)', 'c_proj': 'c_proj (output projections)',
                   'c_fc': 'c_fc (MLP expand)'}

    def vals(l, key):
        return [r['meta']['gap_by_layer'][l][key] for r in recs
                if r['meta']['gap_by_layer'].get(l, {}).get(key) is not None]

    cos_mean = {l: np.mean(vals(l, 'cos_theta')) for l in layer_ids}
    cos_std  = {l: np.std(vals(l, 'cos_theta'))  for l in layer_ids}
    sr_mean  = {l: (np.mean(vals(l, 'scale_ratio')) if vals(l, 'scale_ratio') else np.nan) for l in layer_ids}
    sr_std   = {l: (np.std(vals(l, 'scale_ratio'))  if vals(l, 'scale_ratio') else np.nan) for l in layer_ids}

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for row, key in enumerate(['c_attn', 'c_proj', 'c_fc']):
        group_ids = groups[key]
        short_labels = ['.'.join(l.split('.')[-3:]) for l in group_ids]
        x = np.arange(len(group_ids))

        axes[row, 0].bar(x, [cos_mean[l] for l in group_ids], yerr=[cos_std[l] for l in group_ids],
                          color='steelblue', alpha=0.8, capsize=3)
        axes[row, 0].set_ylabel('cos_theta')
        axes[row, 0].set_title(f'{group_names[key]} — cos_theta')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)

        axes[row, 1].bar(x, [sr_mean[l] for l in group_ids], yerr=[sr_std[l] for l in group_ids],
                          color='darkorange', alpha=0.8, capsize=3)
        axes[row, 1].set_yscale('log')
        axes[row, 1].set_ylabel('scale_ratio (log)')
        axes[row, 1].set_title(f'{group_names[key]} — scale_ratio')
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)

    plt.suptitle('Stage 0 — Hessian-gap shape/scale by layer type (mean ± std across seeds)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stage0_gap_by_layer_type.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_gap_layer_type_groups(recs: list, fig_dir: str):
    layer_ids = list(recs[0]['meta']['gap_by_layer'].keys())
    groups = _layer_type_groups(layer_ids)
    labels = {'c_attn': 'c_attn\n(Q,K,V projection)', 'c_proj': 'c_proj\n(output projection)',
              'c_fc': 'c_fc\n(MLP expand)'}

    results = {}
    for key, ids in groups.items():
        cos_vals, sr_vals = [], []
        for r in recs:
            for l in ids:
                d = r['meta']['gap_by_layer'].get(l)
                if d is None:
                    continue
                cos_vals.append(d['cos_theta'])
                if d['scale_ratio'] is not None:
                    sr_vals.append(d['scale_ratio'])
        results[key] = dict(
            cos_mean=np.mean(cos_vals), cos_std=np.std(cos_vals),
            sr_mean=(np.mean(sr_vals) if sr_vals else np.nan),
            sr_std=(np.std(sr_vals) if sr_vals else np.nan),
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    keys = ['c_attn', 'c_proj', 'c_fc']
    names = [labels[k] for k in keys]
    x = np.arange(len(keys))

    ax1.bar(x, [results[k]['cos_mean'] for k in keys], yerr=[results[k]['cos_std'] for k in keys],
            color=['steelblue', 'tomato', 'mediumseagreen'], alpha=0.85, capsize=5, width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel('cos_theta (mean ± std)')
    ax1.set_title('cos_theta by layer type')

    ax2.bar(x, [results[k]['sr_mean'] for k in keys], yerr=[results[k]['sr_std'] for k in keys],
            color=['steelblue', 'tomato', 'mediumseagreen'], alpha=0.85, capsize=5, width=0.5)
    ax2.set_yscale('log')
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel('scale_ratio (mean ± std, log)')
    ax2.set_title('scale_ratio by layer type')

    plt.suptitle('Stage 0 — Hessian-gap shape/scale by layer type (all seeds pooled)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stage0_gap_layer_type_groups.png'), dpi=120)
    plt.close(fig)


def _plot_geometry_by_layer_type(recs: list, fig_dir: str):
    """Ported verbatim (layout + values) from stage0_analysis.ipynb's 'Analysis 2' cell."""
    layer_ids = list(recs[0]['geom']['cka_by_layer'].keys())
    groups = _layer_type_groups(layer_ids)
    group_names = {'c_attn': 'c_attn (attn input proj)', 'c_proj': 'c_proj (output projections)',
                   'c_fc': 'c_fc (MLP expand)'}

    cka_mean = {l: np.mean([r['geom']['cka_by_layer'].get(l, np.nan) for r in recs]) for l in layer_ids}
    knn_mean = {l: np.mean([r['geom']['knn_overlap_by_layer'].get(l, np.nan) for r in recs]) for l in layer_ids}
    cka_std  = {l: np.std([r['geom']['cka_by_layer'].get(l, np.nan) for r in recs]) for l in layer_ids}
    knn_std  = {l: np.std([r['geom']['knn_overlap_by_layer'].get(l, np.nan) for r in recs]) for l in layer_ids}

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for row, key in enumerate(['c_attn', 'c_proj', 'c_fc']):
        group_ids = groups[key]
        short_labels = ['.'.join(l.split('.')[-3:]) for l in group_ids]
        x = np.arange(len(group_ids))

        axes[row, 0].bar(x, [cka_mean[l] for l in group_ids], yerr=[cka_std[l] for l in group_ids],
                          color='steelblue', alpha=0.8, capsize=3)
        axes[row, 0].axhline(1.0, color='green', lw=0.8, linestyle='--')
        axes[row, 0].set_ylim(0.85, 1.02)
        axes[row, 0].set_ylabel('CKA')
        axes[row, 0].set_title(f'{group_names[key]} — CKA')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)

        axes[row, 1].bar(x, [knn_mean[l] for l in group_ids], yerr=[knn_std[l] for l in group_ids],
                          color='darkorange', alpha=0.8, capsize=3)
        axes[row, 1].set_ylim(0, 1.0)
        axes[row, 1].set_ylabel('kNN overlap')
        axes[row, 1].set_title(f'{group_names[key]} — kNN overlap')
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)

    plt.suptitle('GPTQ 4-bit geometry preservation by layer type (mean ± std across seeds)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stage0_cka_knn_by_layer_type.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_geometry_layer_type_groups(recs: list, fig_dir: str):
    """Ported verbatim (layout + values) from stage0_analysis.ipynb's 'Analysis 3' cell."""
    layer_ids = list(recs[0]['geom']['cka_by_layer'].keys())
    groups = _layer_type_groups(layer_ids)
    labels = {'c_attn': 'c_attn\n(Q,K,V projection)', 'c_proj': 'c_proj\n(output projection)',
              'c_fc': 'c_fc\n(MLP expand)'}

    results = {}
    for key, ids in groups.items():
        cka_vals, knn_vals = [], []
        for r in recs:
            cka_vals += [r['geom']['cka_by_layer'][l] for l in ids if l in r['geom']['cka_by_layer']]
            knn_vals += [r['geom']['knn_overlap_by_layer'][l] for l in ids if l in r['geom']['knn_overlap_by_layer']]
        results[key] = {
            'cka_mean': np.mean(cka_vals), 'cka_std': np.std(cka_vals),
            'knn_mean': np.mean(knn_vals), 'knn_std': np.std(knn_vals),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    keys  = ['c_attn', 'c_proj', 'c_fc']
    names = [labels[k] for k in keys]
    x = np.arange(len(keys))

    ax1.bar(x, [results[k]['cka_mean'] for k in keys], yerr=[results[k]['cka_std'] for k in keys],
            color=['steelblue', 'tomato', 'mediumseagreen'], alpha=0.85, capsize=5, width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0.88, 1.02)
    ax1.set_ylabel('CKA (mean ± std)')
    ax1.set_title('CKA by layer type')
    ax1.axhline(1.0, color='black', lw=0.8, linestyle='--')

    ax2.bar(x, [results[k]['knn_mean'] for k in keys], yerr=[results[k]['knn_std'] for k in keys],
            color=['steelblue', 'tomato', 'mediumseagreen'], alpha=0.85, capsize=5, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('kNN overlap (mean ± std)')
    ax2.set_title('kNN overlap by layer type')

    plt.suptitle('GPTQ 4-bit — geometry damage by layer type (all seeds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stage0_cka_knn_layer_type_groups.png'), dpi=120)
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
        hvp_ids = attach_hvp(stats, model, input_ids, CFG['device'], n_batches=10, batch_size=2)

        # 3. Hessian gap per layer (cos_theta, scale_ratio, scale_ratio_N, frac_neg_quad, ...)
        print("\n[3/4] Computing hessian gap per layer...")
        gap_res = compute_hessian_gap(
            stats, model, hvp_ids, CFG['device'],
            n_v=30, n_trace=20, batch_size=2, probe_seed=seed,
        )
        report_gap(gap_res)

        cos_vals = [r['cos_theta'] for r in gap_res.values()]
        gap_vals = [r['gap_opt']   for r in gap_res.values()]
        sr_vals  = [r['scale_ratio']   for r in gap_res.values() if r['scale_ratio']   is not None]
        srN_vals = [r['scale_ratio_N'] for r in gap_res.values() if r['scale_ratio_N'] is not None]
        neg_vals = [r['frac_neg_quad'] for r in gap_res.values()]

        gap_mean = float(np.mean(gap_vals))
        gap_std  = float(np.std(gap_vals))
        print(f"      Mean gap_opt = {gap_mean:.4f} +/- {gap_std:.4f}  "
              f"(mean cos_theta = {np.mean(cos_vals):.4f})")

        # save H_lay matrices as .npy
        matrices_dir = _save_matrices(stats, seed, CFG['results_dir'])

        # per-seed gap-metric plots — raw values, no threshold lines
        _plot_shape(gap_res, seed, FIG_DIR)
        _plot_scale(gap_res, seed, FIG_DIR)
        _plot_gap_opt(gap_res, seed, FIG_DIR)

        # 4. GPTQ + AWQ native path
        print("\n[4/4] Reproducing GPTQ ")
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
                    'gap_res':     gap_res,
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
                # All fields below are recorded values (means/medians/mins of what
                # compute_hessian_gap measured) — no thresholds, flags, or verdicts are
                # applied anywhere in this pipeline. Interpretation happens later,
                # separately, from these recorded numbers.
                hessian_probe=HessianLadderProbe(
                    h_minus_sigma_norm=gap_mean,
                    cos_theta_mean=float(np.mean(cos_vals)),
                    cos_theta_min=float(np.min(cos_vals)),
                    scale_ratio_median=float(np.median(sr_vals)) if sr_vals else None,
                    scale_ratio_N_median=float(np.median(srN_vals)) if srN_vals else None,
                    frac_neg_quad_mean=float(np.mean(neg_vals)),
                ),
                meta={
                    'gap_by_layer': _sanitise(gap_res),  # full per-layer detail, all 16 fields + region
                    'gap_mean':     gap_mean,
                    'gap_std':      gap_std,
                    'matrices_dir': matrices_dir,
                    'native_path':  True,
                    'ppl_original': ppl_orig,
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

        # cross-seed "by layer-type" plots — gap metrics + CKA/kNN (the latter ported
        # from stage0_analysis.ipynb so it's generated automatically here too)
        _plot_gap_by_layer_type(recs, FIG_DIR)
        _plot_gap_layer_type_groups(recs, FIG_DIR)
        _plot_geometry_by_layer_type(recs, FIG_DIR)
        _plot_geometry_layer_type_groups(recs, FIG_DIR)

    print(f"\nDone.")
    print(f"  Results  -> {STORE_PATH}")
    print(f"  Figures  -> {FIG_DIR}/")
    print(f"  Matrices -> {CFG['results_dir']}/matrices/")


if __name__ == '__main__':
    main()
