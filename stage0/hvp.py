"""
stage0/hvp.py  (revised — weight-space double autograd)
=============
Attaches the L4 HVP to LayerStats.

ROOT CAUSE OF PRIOR BUG (input-activation approach):
  The old code differentiated w.r.t. the layer INPUT x_l.
  This introduced (1/BT)^3 scaling (loss averaged, grad averaged, hvp_val averaged)
  with BT=256 tokens → result was ~6e-8 of the true value → printed as 0.00.

CORRECT APPROACH (vectozavr/llm-hessian, arXiv:2504.04520):
  Differentiate w.r.t. WEIGHT PARAMETERS W_l.
  Uses monkey-patching so loss = f(W_param), then double autograd on W_param.
  Only one (1/BT) factor from the averaged loss → no cancellation.

WHAT H_loss MEANS HERE:
  H_loss_W = (1/d_out) * sum_j  d²L/dw_j²   in (d_in, d_in) space
           ≈ Sigma * trace(H_out)/d_out
           = H_lay * (trace(H_out)/d_out) / n_tokens

  The gap ||H_loss - H_lay||_F / ||H_lay||_F ≈ |trace(H_out)/d_out/n_tokens - 1|
  which will be < 1 and vary between layers — that IS the signal.

SCALE CONVENTION:
  We multiply hvp by n_tokens_calib so that
      hvp(v) ≈ n_tokens * Sigma @ v * (trace(H_out)/d_out)
      h_lay_mv(v) = n_tokens * Sigma @ v
  making both sides comparable for the gap formula in gap.py.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Callable, Optional
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import LayerStats


def _make_layer_hvp(
    model:         AutoModelForCausalLM,
    layer_id:      str,
    input_ids:     torch.Tensor,
    device:        str,
    n_batches:     int   = 10,
    batch_size:    int   = 2,
    scale_factor:  float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns  hvp: v (numpy, d_in,) -> H_loss @ v (numpy, d_in,)

    Algorithm
    ---------
    For each mini-batch:
      1. Clone W_param = layer.weight  (requires_grad=True, frozen reference)
      2. Monkey-patch layer.forward to use W_param instead of layer.weight
      3. loss = model(batch, labels=batch).loss
      4. g = dL/dW_param              (first grad, create_graph=True)
      5. v_full = v broadcast over output dim (same shape as W_param)
         gv    = (g * v_full).sum()   (scalar)
      6. Hv_W = d(gv)/dW_param        (second grad, weight-space HVP)
      7. sum Hv_W over output dim → (d_in,)
    Average over batches, multiply by scale_factor.

    Math note
    ---------
    Hv_W.sum(-1) ≈ (x x^T) @ v * trace(H_out) / d_out * BT_factor
    where H_out = d²L/dy² at this layer's output.
    With scale_factor = n_tokens_calib / n_tokens_per_batch, the result
    matches the scale of H_lay = n_tokens_calib * Sigma used in gap.py.
    """
    target    = dict(model.named_modules())[layer_id]
    is_conv1d = isinstance(target, Conv1D)
    indices   = torch.randperm(len(input_ids))[:n_batches * batch_size]

    # Belt-and-suspenders: disable non-differentiable SDPA backends
    # (attn_implementation='eager' in capture.py already handles this;
    #  these globals are extra safety for double autograd)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    def hvp(v_np: np.ndarray) -> np.ndarray:
        v       = torch.tensor(v_np, dtype=torch.float32, device=device)
        hvp_acc = torch.zeros_like(v)
        count   = 0

        for start in range(0, len(indices), batch_size):
            batch = input_ids[indices[start : start + batch_size]].to(device)

            # ── 1. Differentiable weight copy ────────────────────────────
            W_param = target.weight.detach().clone().requires_grad_(True)
            # Conv1D:  W shape (d_in, d_out)   y = x @ W + b
            # Linear:  W shape (d_out, d_in)   y = W x + b

            bias_val: Optional[torch.Tensor] = (
                target.bias.detach()
                if hasattr(target, 'bias') and target.bias is not None
                else None
            )

            # ── 2. Monkey-patch layer.forward to use W_param ─────────────
            def _make_fwd(Wp, bv, conv1d):
                def _fwd(self, x_in):
                    if conv1d:
                        sz = x_in.size()[:-1] + (Wp.shape[1],)
                        y  = torch.addmm(bv, x_in.view(-1, x_in.size(-1)), Wp)
                        return y.view(*sz)
                    else:
                        return F.linear(x_in, Wp, bv)
                return _fwd

            orig_fwd      = target.forward
            target.forward = _make_fwd(W_param, bias_val, is_conv1d).__get__(
                target, type(target)
            )

            # ── 3. Forward pass ───────────────────────────────────────────
            try:
                loss = model(batch, labels=batch).loss
            finally:
                target.forward = orig_fwd   # always restore

            # ── 4. First grad (keep graph for 2nd pass) ───────────────────
            g = torch.autograd.grad(loss, W_param, create_graph=True)[0]
            if g.abs().sum() == 0:
                print(f"!!! WARNING: Zero gradients for weight param in {layer_id} !!!")
            # g shape: (d_in, d_out) for Conv1D | (d_out, d_in) for Linear

            # ── 5. v_full: broadcast v over output dimension ──────────────
            # Conv1D: v (d_in,) → (d_in, d_out)   each col = v
            # Linear: v (d_in,) → (d_out, d_in)   each row = v
            if is_conv1d:
                v_full = v.unsqueeze(1).expand_as(W_param)   # (d_in, d_out)
            else:
                v_full = v.unsqueeze(0).expand_as(W_param)   # (d_out, d_in)

            gv = (g * v_full).sum()   # scalar

            # ── 6. Second grad = HVP in weight space ──────────────────────
            Hv_W = torch.autograd.grad(gv, W_param)[0]
            # Hv_W shape: same as W_param

            # ── 7. Reduce: sum over output dim → (d_in,) ─────────────────
            if is_conv1d:
                hvp_acc += Hv_W.sum(dim=-1).detach()   # sum over d_out
            else:
                hvp_acc += Hv_W.sum(dim=0).detach()    # sum over d_out

            count += 1

        result = (hvp_acc / max(count, 1)) * scale_factor
        return result.cpu().numpy()

    return hvp


def attach_hvp(
    stats:      Dict[str, LayerStats],
    model:      AutoModelForCausalLM,
    input_ids:  torch.Tensor,
    device:     str,
    n_batches:  int = 10,
    batch_size: int = 2,
) -> None:
    """
    Attaches L4 HVP callable to each LayerStats in-place.

    scale_factor = n_tokens_calib / n_tokens_hvp_per_batch
    This puts hvp(v) on the same scale as h_lay_mv(v) = n_tokens * Sigma @ v
    used in gap.py, so the gap formula ||H_loss - H_lay||_F / ||H_lay||_F
    measures the structural/scale difference, not just an arbitrary unit choice.

    Concretely:
        n_tokens_calib  : from LayerStats (e.g. 512 samples × 128 tok = 65 536)
        n_tokens_hvp    : n_batches × batch_size × seq_len (e.g. 10×2×128 = 2 560)
        scale_factor    : 65 536 / 2 560 ≈ 25.6  (varies by layer if n_tokens differs)
    """
    seq_len       = input_ids.shape[1]
    n_tok_per_run = n_batches * batch_size * seq_len

    for layer_id, st in stats.items():
        scale = float(st.n_tokens) / n_tok_per_run if n_tok_per_run > 0 else 1.0
        stats[layer_id].hvp = _make_layer_hvp(
            model, layer_id, input_ids, device,
            n_batches, batch_size, scale_factor=scale,
        )
        print(f"  [hvp] attached → {layer_id:50s}  scale={scale:.1f}")
