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
    seq_len:    int = 128,
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

import gc
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

def _get_single_layer_activations(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_id: str,
    device: str,
    max_tokens: int = 512,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Captures activations for ONLY ONE specified layer to prevent memory accumulation."""
    modules = dict(model.named_modules())
    if layer_id not in modules:
        return None

    collected_chunks = []

    def hook(module, inp, out):
        # Detach and flatten immediately
        x = out.detach().float().reshape(-1, out.shape[-1]).cpu()
        collected_chunks.append(x)

    handle = modules[layer_id].register_forward_hook(hook)

    with torch.no_grad():
        # Process input_ids in small batches
        batch_size = 4
        for start in range(0, len(input_ids), batch_size):
            model(input_ids[start : start + batch_size].to(device))

    handle.remove()

    if not collected_chunks:
        return None

    # Concatenate only for this single layer
    arr = torch.cat(collected_chunks, dim=0).numpy()
    del collected_chunks

    # Subsample tokens
    if len(arr) > max_tokens:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(arr), max_tokens, replace=False)
        arr = arr[idx]

    return arr




def measure_geometry(
    model_orig, model_q, input_ids, layer_ids,
    device='cuda', seed=0, k=10, max_tokens=512,
):
    cka_scores = {}
    knn_scores = {}
    truncated_input_ids = input_ids[:16, :512]
    q_modules = set(dict(model_q.named_modules()).keys())

    print("  Phase 1: Collecting original activations")
    orig_acts = {}
    for lid in layer_ids:
        X = _get_single_layer_activations(
            model_orig, truncated_input_ids, lid, device,
            max_tokens=max_tokens, seed=seed,
        )
        if X is not None:
            orig_acts[lid] = X

    print("  Phase 2: Swapping models")
    model_orig.to('cpu')
    torch.cuda.empty_cache()
    model_q.to(device)

    print("  Phase 3: Computing metrics")
    for lid in layer_ids:
        if lid not in orig_acts:
            continue
        lid_q = 'model.' + lid if 'model.' + lid in q_modules else lid
        Y = _get_single_layer_activations(
            model_q, truncated_input_ids, lid_q, device,
            max_tokens=max_tokens, seed=seed,
        )
        if Y is None:
            continue
        cka_scores[lid] = _linear_cka(orig_acts[lid], Y)
        knn_scores[lid] = _knn_overlap(orig_acts[lid], Y, k=k, seed=seed)
        del Y
        gc.collect()

    print("  Phase 4: Restoring original model")
    del orig_acts
    model_q.to(device)
    model_orig.to(device)
    torch.cuda.empty_cache()

    return cka_scores, knn_scores





# def measure_geometry(
#     model_orig: torch.nn.Module,
#     model_q: torch.nn.Module,
#     input_ids: torch.Tensor,
#     layer_ids: List[str],
#     device: str = 'cuda',
#     seed: int = 0,
#     k: int = 10,
#     max_tokens: int = 512,
# ) -> Tuple[Dict[str, float], Dict[str, float]]:
#     """
#     Computes CKA & kNN layer-by-layer on demand.
#     Memory footprint remains constant (~O(1) layer in RAM) regardless of model depth.
#     """
#     cka_scores = {}
#     knn_scores = {}

#     # 1. Truncate tokens upfront to avoid evaluating thousands of unneeded tokens
#     # Keep max sequence length to 512 per sample
#     truncated_input_ids = input_ids[:16, :512] 

#     q_modules = set(dict(model_q.named_modules()).keys())
#     model_q.to('cpu')
#     torch.cuda.empty_cache()
#     # 2. Iterate layer by layer
#     for lid in layer_ids:
#         # Determine quantized layer name
#         lid_q = 'model.' + lid if 'model.' + lid in q_modules else lid

#         # Extract X (Original Layer Activation)
#         X = _get_single_layer_activations(
#             model_orig, truncated_input_ids, lid, device, max_tokens=max_tokens, seed=seed
#         )
#         if X is None:
#             continue

#         # Extract Y (Quantized Layer Activation)
#         Y = _get_single_layer_activations(
#             model_q, truncated_input_ids, lid_q, device, max_tokens=max_tokens, seed=seed
#         )
#         if Y is None:
#             del X
#             continue

#         # Compute metrics for this single layer
#         cka_scores[lid] = _linear_cka(X, Y)
#         knn_scores[lid] = _knn_overlap(X, Y, k=k, seed=seed)

#         # Clear memory immediately before proceeding to next layer
#         del X, Y
#         gc.collect()

#     return cka_scores, knn_scores
# def _capture_output_activations(
#     model:      AutoModelForCausalLM,
#     input_ids:  torch.Tensor,
#     layer_ids:  list,
#     device:     str,
#     batch_size: int = 8,
#     max_tokens: int = 512,
#     seed: int = 0,
# ) -> Dict[str, np.ndarray]:
#     """
#     Captures output activations per layer for CKA/kNN comparison.
#     Subsamples to max_tokens to keep kNN memory manageable.
#     returns a dict like this {"transformer.h.0":  np.array of shape (512, 768),"transformer.h.1":  np.array of shape (512, 768),}
#     """
#     activations: Dict[str, list] = {}
#     print("Entering output activation")

#     def make_hook(lid):
#         def hook(module, inp, out):
#             x = out.detach().float().reshape(-1, out.shape[-1])
#             activations.setdefault(lid, []).append(x.cpu())
#         return hook

#     modules = dict(model.named_modules())
#     handles = [modules[lid].register_forward_hook(make_hook(lid))
#                for lid in layer_ids if lid in modules]

#     with torch.no_grad():
#         for start in range(0, len(input_ids), batch_size):
#             model(input_ids[start : start + batch_size].to(device))

#     for h in handles:
#         h.remove()

#     result = {}
#     for lid, chunks in activations.items():
#         arr = torch.cat(chunks, dim=0).numpy()
#         if len(arr) > max_tokens:
#             rng = np.random.default_rng(seed)
#             idx = rng.choice(len(arr), max_tokens, replace=False)
#             arr = arr[idx]
#         result[lid] = arr
#     print("Completed Activation output")
#     return result



# def measure_geometry(
#     model_orig: AutoModelForCausalLM,
#     model_q:    AutoModelForCausalLM,
#     input_ids:  torch.Tensor,
#     layer_ids:  list,
#     device:     str = 'cuda',
#     seed:       int = 0,
#     k:          int = 10,
# ) -> Tuple[Dict[str, float], Dict[str, float]]:
#     """
#     Compares representational geometry between an original and quantized model
#     using two complementary metrics:
#       - CKA   (global geometry): do layers organise all tokens similarly?
#       - kNN overlap (local geometry): does each token keep its nearest neighbors?

#     Returns:
#         cka_scores : {layer_id: float in [0,1]}
#         knn_scores : {layer_id: float in [0,1]}
#     """

#     # ── Step 1: Capture activations from the original model ──────────────
#     # Result: dict of {layer_id: np.array of shape (512, hidden_dim)}
#     # Each row is one token's activation vector at that layer.
#     H1 = _capture_output_activations(
#         model_orig, input_ids[:64], layer_ids, device, seed=seed
#     )

#     # ── Step 2: Capture activations from the quantized model ─────────────
#     # AutoGPTQ wraps the model and prepends 'model.' to all module names,
#     # so "transformer.h.0" becomes "model.transformer.h.0".
#     # We detect this and adapt the layer IDs, then strip the prefix back
#     # so both dictionaries use the same keys for comparison.
#     q_modules   = set(dict(model_q.named_modules()).keys())
#     layer_ids_q = [
#         'model.' + lid if 'model.' + lid in q_modules else lid
#         for lid in layer_ids
#     ]
#     H2_raw = _capture_output_activations(
#         model_q, input_ids[:64], layer_ids_q, device, seed=seed
#     )
#     H2 = {
#         k_name[len('model.'):] if k_name.startswith('model.') else k_name: v
#         for k_name, v in H2_raw.items()
#     }

#     # ── Step 3: Compute metrics layer by layer ───────────────────────────
#     cka_scores = {}
#     knn_scores = {}

#     for lid in layer_ids:
#         if lid not in H1 or lid not in H2:
#             continue

#         X = H1[lid]  # shape (512, 768) — original model's activations
#         Y = H2[lid]  # shape (512, 768) — quantized model's activations

#         cka_scores[lid] = _linear_cka(X, Y)
#         knn_scores[lid] = _knn_overlap(X, Y, k=k, seed=seed)

#     return cka_scores, knn_scores


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two activation matrices.

    X: (N, p)  — N tokens, p-dimensional activations from layer A
    Y: (N, q)  — N tokens, q-dimensional activations from layer B

    Steps:
    1. Compute Gram (kernel) matrices:  K = XX^T,  L = YY^T
       Both are (N, N). Entry (i,j) = dot product between token i and token j.
       This encodes pairwise geometric relationships.

    2. Center the kernel matrices using H = I - (1/N) * 11^T
       K_c = HKH,  L_c = HLH
       This removes the mean so similarity isn't driven by average activation magnitude.

    3. CKA = trace(K_c @ L_c) / sqrt(trace(K_c @ K_c) * trace(L_c @ L_c))
       Numerator: how much the two centered kernels agree.
       Denominator: normalises to [0, 1].

    Equivalent shortcut (avoids forming N×N matrices explicitly):
       We can use the HSIC formulation with Frobenius norms directly.
    """
    n = X.shape[0]

    # Center the data by subtracting the mean token representation
    # This is equivalent to centering the kernel matrices but cheaper:
    # H @ X = X - mean(X), and (HXX^TH) = (X - mean)(X - mean)^T
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute the key dot products using the trace-of-product identity:
    # trace(K_c @ L_c) = trace(XX^T @ YY^T) = ||X^T Y||_F^2
    # This avoids ever forming the (N, N) kernel matrices.
    # X^T Y has shape (p, q), which is (768, 768) — much smaller than (512, 512)
    # when p and q are comparable to N. But even when N < p, this works fine.
    XtY = X.T @ Y                          # (p, q)
    XtX = X.T @ X                          # (p, p)
    YtY = Y.T @ Y                          # (q, q)

    hsic_xy = np.sum(XtY ** 2)             # = trace(K_c @ L_c)
    hsic_xx = np.sum(XtX ** 2)             # = trace(K_c @ K_c)
    hsic_yy = np.sum(YtY ** 2)             # = trace(L_c @ L_c)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    return float(hsic_xy / denominator)


def _knn_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
    seed: int = 0,
) -> float:
    """
    kNN Overlap: for each token, find its k nearest neighbors in X and in Y,
    then measure how many neighbors are shared.

    X: (N, p)  — original activations
    Y: (N, q)  — quantized activations
    k: number of neighbors to compare

    Steps:
    1. Compute pairwise squared distances using the expansion:
       ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
       This avoids creating an (N, N, p) intermediate tensor.

    2. For each token, find the indices of its k closest tokens.

    3. Compare the two neighbor sets and compute the fraction of overlap.

    Returns: average overlap across all tokens, in [0, 1].
    """
    neighbors_orig  = _knn_efficient(X, k)   # (N, k) indices
    neighbors_quant = _knn_efficient(Y, k)   # (N, k) indices

    n = X.shape[0]
    overlap = 0.0
    for i in range(n):
        # Set intersection: how many of token i's neighbors are the same?
        shared = len(set(neighbors_orig[i]) & set(neighbors_quant[i]))
        overlap += shared / k

    return float(overlap / n)


def _knn_efficient(H: np.ndarray, k: int) -> np.ndarray:
    """
    Finds k nearest neighbors for each token using squared Euclidean distance.

    H: (N, d)  — activation matrix

    The distance expansion trick:
       ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b

    - sq = ||a||^2 for each token            → shape (N,)
    - sq[:, None] + sq[None, :]              → (N, N) pairwise sum of squared norms
    - 2 * (H @ H.T)                          → (N, N) pairwise dot products
    - Subtract to get (N, N) distance matrix

    fill_diagonal(inf): a token shouldn't be its own neighbor.
    argsort picks the k smallest distances per row.

    Returns: (N, k) array of neighbor indices.
    """
    sq = (H ** 2).sum(axis=1)                        # (N,)
    d  = sq[:, None] + sq[None, :] - 2 * (H @ H.T)  # (N, N)
    np.fill_diagonal(d, np.inf)                       # exclude self
    return np.argsort(d, axis=1)[:, :k]               # (N, k)

# def measure_geometry(
#     model_orig: AutoModelForCausalLM,
#     model_q:    AutoModelForCausalLM,
#     input_ids:  torch.Tensor,
#     layer_ids:  list,
#     device:     str,
#     k_nn:       int = 10,
#     seed=0,
# ) -> Tuple[Dict[str, float], Dict[str, float]]:
#     probe = GeometryProbe()

#     # H1: original model
#     H1 = _capture_output_activations(model_orig, input_ids[:64], layer_ids, 'cuda', seed=seed)

#     # AutoGPTQ wraps model with 'model.' prefix — detect and adapt
#     q_modules   = set(dict(model_q.named_modules()).keys())
#     layer_ids_q = ['model.' + lid if 'model.' + lid in q_modules else lid
#                    for lid in layer_ids]
#     H2_raw = _capture_output_activations(model_q, input_ids[:64], layer_ids_q, 'cuda', seed=seed)
#     H2 = {k[len('model.'):] if k.startswith('model.') else k: v
#           for k, v in H2_raw.items()}

#     def _knn_efficient(H, k):
#         """||a-b||² = ||a||² + ||b||² - 2a·b  — avoids (n,n,d) intermediate."""
#         sq = (H ** 2).sum(axis=1)
#         d  = sq[:, None] + sq[None, :] - 2 * (H @ H.T)
#         np.fill_diagonal(d, np.inf)
#         return np.argsort(d, axis=1)[:, :k]

#     cka, knn = {}, {}
#     for lid in layer_ids:
#         if lid in H1 and lid in H2:
#             cka[lid] = probe.linear_cka(H1[lid], H2[lid])
#             a = _knn_efficient(H1[lid], k=10)
#             b = _knn_efficient(H2[lid], k=10)
#             jac = [len(set(ai) & set(bi)) / len(set(ai) | set(bi))
#                    for ai, bi in zip(a, b)]
#             knn[lid] = float(np.mean(jac))

#     return cka, knn


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


