"""
stage0/capture.py
=================
Captures layer-wise input activations from any HuggingFace causal LM on C4
calibration data and builds LayerStats objects (professor's core structure).

Reusable: pass any model_name (gpt2, gpt2-medium, meta-llama/Llama-2-7b, etc.)
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import LayerStats


def _classify_region(layer_name: str) -> str:
    """Map layer name to 'attn' | 'ffn' | 'other'."""
    if 'attn' in layer_name:
        return 'attn'
    if 'mlp' in layer_name or 'fc' in layer_name:
        return 'ffn'
    return 'other'


def load_calibration_data(
    tokenizer,
    n_samples:  int = 512,
    seq_len:    int = 128,
    seed:       int = 0,
) -> torch.Tensor:
    """
    Returns (n_samples, seq_len) token ids from C4 validation split.
    Fixed seed ensures every method sees the exact same calibration data.
    Reusable across all stages.
    """
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    samples = []
    for item in dataset:
        enc = tokenizer(item['text'], return_tensors='pt',
                        truncation=True, max_length=seq_len)
        if enc['input_ids'].shape[1] == seq_len:
            samples.append(enc['input_ids'])
        if len(samples) >= n_samples:
            break

    return torch.cat(samples, dim=0)  # (n_samples, seq_len)


def build_layer_stats(
    model_name:  str         = 'gpt2',
    n_samples:   int         = 512,
    seq_len:     int         = 128,
    batch_size:  int         = 8,
    device:      str         = 'cuda',
    seed:        int         = 0,
    dtype:       torch.dtype = torch.float32,
) -> Tuple[Dict[str, LayerStats], AutoModelForCausalLM, AutoTokenizer]:
    """
    Main entry point. Runs model on C4, computes Sigma = (1/N) X X^T per layer,
    returns LayerStats for every nn.Linear. hvp is None at this stage.

    Note: Sigma here IS H_lay / N exactly (professor's Eq.1).
          H_lay = X X^T = N * Sigma — stored as LayerStats.sigma * n_tokens.

    Returns
    -------
    stats     : dict[layer_id -> LayerStats]
    model     : loaded model (reused by hvp.py and baselines.py)
    tokenizer : reused by downstream steps
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,attn_implementation="eager")
    model = model.to(device).eval()

    input_ids = load_calibration_data(tokenizer, n_samples, seq_len, seed).to(device)

    # accumulate input activations X per Linear layer
    sigma_sum:  Dict[str, np.ndarray] = {}
    token_count: Dict[str, int]       = {}
    
    def make_hook(layer_id: str):
        def hook(module, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu().numpy()
            if layer_id not in sigma_sum:
                sigma_sum[layer_id]   = np.zeros((x.shape[1], x.shape[1]))
                token_count[layer_id] = 0
            sigma_sum[layer_id]   += x.T @ x        # running sum of X.T @ X
            token_count[layer_id] += len(x)
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            model(input_ids[start : start + batch_size])

    for h in handles:
        h.remove()

    # build LayerStats from accumulated X
    stats: Dict[str, LayerStats] = {}
    modules_dict = dict(model.named_modules())

    for layer_id in sigma_sum:
        N     = token_count[layer_id]
        sigma = sigma_sum[layer_id] / N          # (1/N) X^T X = Sigma
        mod   = modules_dict[layer_id]
        is_conv1d = isinstance(mod, Conv1D)
        d_out = mod.nf if is_conv1d else mod.out_features
        d_in  = sigma.shape[0]

        stats[layer_id] = LayerStats(
            layer_id     = layer_id,
            region       = _classify_region(layer_id),
            weight_shape = (d_out, d_in),
            weight_is_transposed = is_conv1d,
            sigma        = sigma,
            n_tokens     = N,
        )

    return stats, model, tokenizer, input_ids
