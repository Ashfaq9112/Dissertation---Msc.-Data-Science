"""
stage0/baselines.py
===================
Reproduces GPTQ and AWQ perplexity on GPT-2 Small + WikiText-2 (native path)
and measures CKA + kNN between original and quantized model activations.

IMPORTANT: run_gptq and run_awq accept our C4 input_ids so both methods use
the exact same calibration data as capture.py — ensuring H_lay is comparable.

WIRE-UP-NEEDED sections are filled once AutoGPTQ and AutoAWQ are installed
on the GPU workstation.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import GeometryProbe


def evaluate_perplexity(
    model:      AutoModelForCausalLM,
    tokenizer:  AutoTokenizer,
    device:     str,
    seq_len:    int = 512,
    n_samples:  int = 128,
) -> float:
    """
    Perplexity on WikiText-2 test set.
    Reusable for original model, GPTQ model, AWQ model — any causal LM.
    """
    from datasets import load_dataset
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(data['text'])
    ids  = tokenizer(text, return_tensors='pt')['input_ids'][0]

    nlls = []
    model.eval()
    with torch.no_grad():
        for start in range(0, min(len(ids) - seq_len, n_samples * seq_len), seq_len):
            chunk = ids[start : start + seq_len].unsqueeze(0).to(device)
            nlls.append(model(chunk, labels=chunk).loss.item())

    return float(np.exp(np.mean(nlls)))


def _capture_output_activations(
    model:      AutoModelForCausalLM,
    input_ids:  torch.Tensor,
    layer_ids:  list,
    device:     str,
    batch_size: int = 8,
    max_tokens: int = 512,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Captures output activations per layer for CKA/kNN comparison.
    Subsamples to max_tokens to keep kNN memory manageable.
    """
    activations: Dict[str, list] = {}
    print("Entering output activation")

    def make_hook(lid):
        def hook(module, inp, out):
            x = out.detach().float().reshape(-1, out.shape[-1])
            activations.setdefault(lid, []).append(x.cpu())
        return hook

    modules = dict(model.named_modules())
    handles = [modules[lid].register_forward_hook(make_hook(lid))
               for lid in layer_ids if lid in modules]

    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            model(input_ids[start : start + batch_size].to(device))

    for h in handles:
        h.remove()

    result = {}
    for lid, chunks in activations.items():
        arr = torch.cat(chunks, dim=0).numpy()
        if len(arr) > max_tokens:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(arr), max_tokens, replace=False)
            arr = arr[idx]
        result[lid] = arr
    print("Completed Activation output")
    return result


def measure_geometry(
    model_orig: AutoModelForCausalLM,
    model_q:    AutoModelForCausalLM,
    input_ids:  torch.Tensor,
    layer_ids:  list,
    device:     str,
    k_nn:       int = 10,
    seed=0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    probe = GeometryProbe()

    # H1: original model
    H1 = _capture_output_activations(model_orig, input_ids[:64], layer_ids, 'cuda', seed=seed)

    # AutoGPTQ wraps model with 'model.' prefix — detect and adapt
    q_modules   = set(dict(model_q.named_modules()).keys())
    layer_ids_q = ['model.' + lid if 'model.' + lid in q_modules else lid
                   for lid in layer_ids]
    H2_raw = _capture_output_activations(model_q, input_ids[:64], layer_ids_q, 'cuda', seed=seed)
    H2 = {k[len('model.'):] if k.startswith('model.') else k: v
          for k, v in H2_raw.items()}

    def _knn_efficient(H, k):
        """||a-b||² = ||a||² + ||b||² - 2a·b  — avoids (n,n,d) intermediate."""
        sq = (H ** 2).sum(axis=1)
        d  = sq[:, None] + sq[None, :] - 2 * (H @ H.T)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    cka, knn = {}, {}
    for lid in layer_ids:
        if lid in H1 and lid in H2:
            cka[lid] = probe.linear_cka(H1[lid], H2[lid])
            a = _knn_efficient(H1[lid], k=10)
            b = _knn_efficient(H2[lid], k=10)
            jac = [len(set(ai) & set(bi)) / len(set(ai) | set(bi))
                   for ai, bi in zip(a, b)]
            knn[lid] = float(np.mean(jac))

    return cka, knn


def run_gptq(
    model_name: str,
    device:     str,
    input_ids:  torch.Tensor,
    bits:       int = 4,
    group_size: int = 128,
) -> Tuple[Optional[AutoModelForCausalLM], Optional[float]]:
    """
    Runs GPTQ quantization using our C4 calibration data.
    Uses AutoGPTQ with standard GPTQ config (4-bit, group_size=128, damp=0.01).
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = BaseQuantizeConfig(
        bits         = bits,
        group_size   = group_size,
        damp_percent = 0.01,
        desc_act     = False,

    )

    model_q = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)

    # pass our calibration data so H_lay stays consistent with capture.py
    calib = [
        {
            'input_ids':      input_ids[i].cpu().unsqueeze(0),
            'attention_mask': torch.ones(1, input_ids.shape[1], dtype=torch.long),
        }
        for i in range(min(128, len(input_ids)))
    ]

    model_q.quantize(calib)
    model_q = model_q.to(device)

    ppl = evaluate_perplexity(model_q, tokenizer, device)
    print(f"  [GPTQ] PPL = {ppl:.2f}")
    return model_q, ppl


# def run_awq(
#     model_name: str,
#     device:     str,
#     input_ids:  torch.Tensor,
#     bits:       int = 4,
#     group_size: int = 128,
# ) -> Tuple[Optional[AutoModelForCausalLM], Optional[float]]:
#     """
#     Runs AWQ quantization using our C4 calibration data.
#     Decodes our input_ids back to text since AWQ expects string inputs.
#     """
#     from awq import AutoAWQForCausalLM

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model_q = AutoAWQForCausalLM.from_pretrained(model_name, safetensors=True)

#     quant_config = {
#         'zero_point':  True,
#         'q_group_size': group_size,
#         'w_bit':        bits,
#         'version':      'GEMM',
#     }

#     # AWQ expects text strings — decode our token ids back to text
#     calib_texts = [
#         tokenizer.decode(input_ids[i], skip_special_tokens=True)
#         for i in range(min(128, len(input_ids)))
#     ]

#     model_q.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
#     model_q = model_q.to(device)

#     ppl = evaluate_perplexity(model_q, tokenizer, device)
#     print(f"  [AWQ]  PPL = {ppl:.2f}")
#     return model_q, ppl
