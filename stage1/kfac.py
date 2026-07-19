"""
stage1/kfac.py
==============
Attaches the L3 K-FAC-style curvature signal to LayerStats.

L3's sensitivity needs a second matrix beyond Sigma = A (the input second
moment already captured by stage0.capture): G = E[g g^T], the covariance of
the loss gradient w.r.t. each layer's OWN output. The K-FAC approximation
says H_W ~= A (kron) G, and because eig(A kron G) is every pairwise product
of eig(A) and eig(G):

    mean_eig(A kron G) = mean_eig(A) * mean_eig(G)
                        = [trace(A)/d_in] * [trace(G)/d_out]

So L3 only ever needs trace(G) = E[||g||^2] -- never the full G matrix, never
an eigendecomposition of it. That's why this is a single ordinary backward
pass (register_full_backward_hook + loss.backward()), NOT the double-autograd
trick stage0.hvp needs: hvp.py differentiates in WEIGHT space (where Conv1D's
transposed layout matters); this differentiates in OUTPUT space, where
Conv1D and nn.Linear look identical, so no Conv1D-specific handling is needed.

Reusable: same contract as stage0.hvp.attach_hvp -- takes the stats dict,
model, and input_ids that stage0.capture.build_layer_stats already returns,
and attaches .trace_g in place. Any future stage that wants the L3 rung
imports this the same way stage1 imports stage0.hvp.
"""

from __future__ import annotations
import torch
from typing import Dict
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import LayerStats


def attach_kfac_trace(
    stats:      Dict[str, LayerStats],
    model:      AutoModelForCausalLM,
    input_ids:  torch.Tensor,
    device:     str,
    n_batches:  int = 10,
    batch_size: int = 8,
    seed:       int = 0,
) -> None:
    """
    Attaches trace(G) = E[||dL/d(layer output)||^2] to each LayerStats
    in-place. Only layers already present in `stats` (i.e. every nn.Linear /
    Conv1D that stage0.capture.build_layer_stats captured) are hooked, so
    this stays in lockstep with Sigma.

    Model is left in whatever mode capture.py already put it in (eval --
    dropout off, matching the deterministic-Sigma convention). eval() does
    NOT disable autograd, so loss.backward() still works correctly here.
    """
    modules_dict = dict(model.named_modules())
    target_ids = [
        lid for lid in stats
        if lid in modules_dict and isinstance(modules_dict[lid], (torch.nn.Linear, Conv1D))
    ]

    g_sq_sum:    Dict[str, float] = {lid: 0.0 for lid in target_ids}
    token_count: Dict[str, int]   = {lid: 0   for lid in target_ids}

    def make_hook(layer_id: str):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            g = g.detach().float().reshape(-1, g.shape[-1])
            m = g.shape[0]                                  # tokens in this batch
            g_sq_sum[layer_id]    += float((g ** 2).sum().item()) * (m ** 2)
            token_count[layer_id] += m
        return hook

    handles = [
        modules_dict[lid].register_full_backward_hook(make_hook(lid))
        for lid in target_ids
    ]

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(input_ids), generator=gen)[: n_batches * batch_size]

    for start in range(0, len(indices), batch_size):
        batch = input_ids[indices[start : start + batch_size]].to(device)
        model.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()
    model.zero_grad(set_to_none=True)

    for h in handles:
        h.remove()

    for layer_id in target_ids:
        n = token_count[layer_id]
        stats[layer_id].trace_g = (g_sq_sum[layer_id] / n) if n > 0 else None
        print(f"  [kfac] attached -> {layer_id:50s}  trace(G)={stats[layer_id].trace_g}")