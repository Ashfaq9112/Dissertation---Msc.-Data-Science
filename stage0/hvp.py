
from typing import Dict, Optional
import numpy as np
import torch
import os, sys
from stage0.capture import LayerStats  # or wherever it's defined
from transformers.pytorch_utils import Conv1D
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def attach_hvp(
    stats: Dict[str, LayerStats],
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: str = 'cuda',
    n_batches: int = 20,
    batch_size: int = 2,
):
    """
    For each layer, attach a function st.hvp(v, u) -> Hv that computes
    the Hessian-vector product via backprop, contracted over output
    channels by Rademacher vector u.
    
    H ≈ G ⊗ A  (K-FAC structure)
    
    The hvp is computed by:
    1. Forward pass on calibration data
    2. Compute loss
    3. Get gradient of loss w.r.t. layer weights (first backward)
    4. Contract: dot the gradient with the probe direction
    5. Second backward to get the HVP
    """
    modules_dict = dict(model.named_modules())
    
    # Pre-select calibration batches (same data for all layers → fair)
    cal_batches = []
    for i in range(n_batches):
        start = i * batch_size
        if start + batch_size > len(input_ids):
            break
        cal_batches.append(input_ids[start : start + batch_size])
    
    for layer_id, st in stats.items():
        mod = modules_dict[layer_id]
        d_out, d_in = st.weight_shape
        is_conv1d = st.weight_is_transposed
        
        def _make_hvp(module, layer_id, d_out, d_in, is_conv1d):
            def hvp_fn(v_np, u_np):
                """
                v_np: (d_in,) input-space Rademacher probe
                u_np: (d_out,) output-space Rademacher probe
                Returns: (d_in,) Hessian-vector product
                """
                v = torch.tensor(v_np, dtype=torch.float32, device=device)
                u = torch.tensor(u_np, dtype=torch.float32, device=device)
                
                acc = torch.zeros(d_in, device=device)
                total_tokens = 0
                
                for batch in cal_batches:
                    model.zero_grad()
                    
                    logits = model(batch).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    
                    # W shape: Conv1D stores (d_in, d_out), Linear stores (d_out, d_in)
                    W = module.weight  # requires_grad=True by default
                    
                    # First backward: get gradient of loss w.r.t. W
                    grad_W = torch.autograd.grad(
                        loss, W, create_graph=True
                    )[0]  # same shape as W
                    
                    # Contract with probes:
                    # For Linear: W is (d_out, d_in), grad_W is (d_out, d_in)
                    #   We want: sum over output dim of u[j] * grad_W[j, :] dot v
                    #   = u^T @ grad_W @ v  (but grad_W is a matrix, this gives scalar)
                    # For Conv1D: W is (d_in, d_out), so grad_W is (d_in, d_out)
                    
                    if is_conv1d:
                        # grad_W: (d_in, d_out)
                        # scalar = v^T @ grad_W @ u
                        scalar = torch.dot(v, grad_W @ u)
                    else:
                        # grad_W: (d_out, d_in)
                        # scalar = u^T @ grad_W @ v
                        scalar = torch.dot(u, grad_W @ v)
                    
                    # Second backward: d(scalar)/dW gives the HVP direction
                    hvp_W = torch.autograd.grad(
                        scalar, W
                    )[0]  # same shape as W
                    
                    # Extract the input-dimension component
                    # For Linear (d_out, d_in): contract output dim with u
                    # For Conv1D (d_in, d_out): contract output dim with u
                    if is_conv1d:
                        # hvp_W: (d_in, d_out) → sum over d_out with u
                        Hv = hvp_W @ u   # (d_in,)
                    else:
                        # hvp_W: (d_out, d_in) → sum over d_out with u
                        Hv = u @ hvp_W   # (d_in,)
                    
                    n_tok = batch.shape[0] * (batch.shape[1] - 1)
                    acc += Hv.detach() * n_tok
                    total_tokens += n_tok
                
                return (acc / total_tokens).cpu().numpy()
            
            return hvp_fn
        
        st.hvp = _make_hvp(mod, layer_id, d_out, d_in, is_conv1d)
        print(f"  HVP attached: {layer_id}")




























# def attach_g_diagnostics(
#     stats:      Dict[str, "LayerStats"],
#     model,
#     input_ids:  torch.Tensor,
#     device:     str,
#     n_batches:  int = 10,
#     batch_size: int = 2,
# ) -> None:
#     """
#     Fills stats[lid].trace_g and stats[lid].sum_g.

#     G is the output-gradient second moment:  G = (1/N) sum_n g_n g_n^T
#     with g_n = dL/dy_n the gradient w.r.t. the layer's OUTPUT. Therefore

#         Tr(G)  = (1/N) sum_n ||g_n||^2          <- correct contraction
#         1'G1   = (1/N) sum_n (1^T g_n)^2        <- what broadcasting picks up

#     Both are expectations of per-token scalars, so a backward hook on
#     grad_output gives them for free. No d_out x d_out matrix is ever formed
#     (c_fc would be 3072x3072 per layer).

#     ON LOSS SCALING: the absolute scale of both depends on whether the loss
#     is mean- or sum-reduced over tokens. We do NOT try to correct for it,
#     because the diagnostic that matters is the RATIO sum_g / trace_g, in
#     which any global scaling cancels exactly. Absolute Tr(G) is used only
#     within-layer, never compared across different loss conventions.
#     """
#     mods = dict(model.named_modules())
#     sq_norm: Dict[str, float] = {}   # sum_n ||g_n||^2
#     sq_sum:  Dict[str, float] = {}   # sum_n (1^T g_n)^2
#     counts:  Dict[str, int]   = {}

#     def make_hook(lid):
#         def hook(module, grad_input, grad_output):
#             g = grad_output[0].detach()                       # (B, S, d_out)
#             g = g.reshape(-1, g.shape[-1]).double()           # (n_tok, d_out)
#             sq_norm[lid] = sq_norm.get(lid, 0.0) + g.pow(2).sum().item()
#             sq_sum[lid]  = sq_sum.get(lid, 0.0) + g.sum(dim=1).pow(2).sum().item()
#             counts[lid]  = counts.get(lid, 0) + g.shape[0]
#         return hook

#     handles = [mods[lid].register_full_backward_hook(make_hook(lid)) for lid in stats]

#     model.zero_grad(set_to_none=True)
#     for b in range(n_batches):
#         ids = input_ids[b * batch_size:(b + 1) * batch_size].to(device)

#         logits = model(ids).logits[:, :-1, :]        # causal LM: drop last position
#         B, S, V = logits.shape

#         with torch.no_grad():                        # y ~ p_model(y|x), NOT the data labels
#             p = torch.softmax(logits, dim=-1).reshape(-1, V)
#             y_sampled = torch.multinomial(p, 1).squeeze(1)

#         loss = torch.nn.functional.cross_entropy(    # SUM, not mean
#             logits.reshape(-1, V), y_sampled, reduction='sum')
#         loss.backward()
#         model.zero_grad(set_to_none=True)

#     for h in handles:
#         h.remove()

#     for lid, s in stats.items():
#         if lid not in counts:
#             continue                      # layer never fired (e.g. skipped)
#         N = counts[lid]
#         s.trace_g = sq_norm[lid] / N
#         s.sum_g   = sq_sum[lid] / N


# # ─────────────────────────────────────────────────────────────────────────
# # Part 2 — the HVP probe: action of M = sum_k [H]_kk
# # ─────────────────────────────────────────────────────────────────────────

# def _make_probe_fn(model, mod, input_ids, device, n_batches, batch_size,
#                    d_out, d_in, is_transposed):
#     """
#     Returns  probe(v, rng) -> ndarray (d_in,), an UNBIASED estimate of M @ v
#     from a SINGLE Rademacher draw. Average many calls to reduce variance.

#     NOT a deterministic linear operator (u is redrawn each call). Do not feed
#     to Lanczos/CG, which assume linearity. Average explicitly instead.
#     """
#     W = mod.weight

#     def probe(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
#         v_t = torch.as_tensor(v, dtype=W.dtype, device=device)
#         u_np = rng.integers(0, 2, size=d_out).astype(np.float64) * 2.0 - 1.0
#         u_t = torch.as_tensor(u_np, dtype=W.dtype, device=device)

#         # probe is ALWAYS built in logical (d_out, d_in) space...
#         V_logical = torch.outer(u_t, v_t)
#         # ...and transposed only where it touches the stored parameter.
#         P = V_logical.T.contiguous() if is_transposed else V_logical

#         acc = torch.zeros(d_in, dtype=W.dtype, device=device)
#         for b in range(n_batches):
#             ids = input_ids[b * batch_size:(b + 1) * batch_size].to(device)
#             out = model(ids, labels=ids)                    # REAL labels — see below
#             n_tok = ids.shape[0] * (ids.shape[1] - 1)
#             loss = out.loss * n_tok                         # mean -> sum
#             g = torch.autograd.grad(loss, W, create_graph=True)[0]
#             # second backward of the scalar <g, P>  ==  H vec(P)
#             Hv = torch.autograd.grad((g * P).sum(), W, retain_graph=False)[0]
#             Hv_logical = Hv.T if is_transposed else Hv        # back to (d_out, d_in)
#             acc += u_t @ Hv_logical                           # contract output index
#         return (acc / n_batches).detach().cpu().numpy()

#     return probe


# def attach_hvp(
#     stats:      Dict[str, "LayerStats"],
#     model,
#     input_ids:  torch.Tensor,
#     device:     str,
#     n_batches:  int = 10,
#     batch_size: int = 2,
# ) -> torch.Tensor:
#     """
#     Attaches stats[lid].hvp = probe(v, rng) for every quantizable layer, and
#     fills trace_g / sum_g.

#     RETURNS the exact token subset used, so that Sigma can be recomputed on
#     MATCHED data. This matters: comparing M (estimated on n_batches*batch_size
#     *seq_len tokens) against N*Sigma (65,536 tokens) makes sampling noise the
#     dominant term in the gap. Mismatched N is variance, not scale -- it cannot
#     be normalized away.
#     """
#     mods = dict(model.named_modules())
#     hvp_ids = input_ids[:n_batches * batch_size]

#     for lid, s in stats.items():
#         if not getattr(s, 'quantizable', True):
#             continue                       # lm_head: weight-tied to wte, Eq.(2) N/A
#         mod = mods[lid]
#         d_out, d_in = s.weight_shape
#         is_t = s.weight_is_transposed
#         assert is_t == isinstance(mod, Conv1D), f"{lid}: transpose flag disagrees"
#         expect = (d_in, d_out) if is_t else (d_out, d_in)
#         assert tuple(mod.weight.shape) == expect, \
#             f"{lid}: weight {tuple(mod.weight.shape)} != {expect}"

#         s.hvp = _make_probe_fn(model, mod, hvp_ids, device,
#                                n_batches, batch_size, d_out, d_in, is_t)

#     attach_g_diagnostics(stats, model, hvp_ids, device, n_batches, batch_size)
#     return hvp_ids


# # ─────────────────────────────────────────────────────────────────────────
# # Part 3 — the diagnostic the supervisor asked for, per layer
# # ─────────────────────────────────────────────────────────────────────────

# def report_g_ratio(stats) -> Dict[str, float]:
#     """
#     ratio = 1'G1 / Tr(G) per layer.

#     Reading it:
#       ratio ~ 1        G effectively diagonal; broadcast was harmless here.
#       ratio >> 1       output channels coherent; broadcast INFLATES this layer.
#       ratio ~ 0        channels cancel; broadcast COLLAPSES this layer to
#                        near-zero sensitivity for reasons unrelated to curvature.
#       non-monotone in depth  => L4's layer ranking was tracking G's coherence,
#                                not curvature. Pins the anomaly to the projection.

#     Note 1'G1 <= d_out * Tr(G) by Cauchy-Schwarz, and 1'G1 >= 0 since G is PSD
#     by construction (it is a Gram matrix), so ratio is in [0, d_out].
#     """
#     print(f"\n  {'layer':42s} {'Tr(G)':>12s} {'1_G_1':>12s} {'ratio':>9s}")
#     out = {}
#     for lid in sorted(stats):
#         s = stats[lid]
#         if s.trace_g is None:
#             continue
#         r = s.sum_g / max(s.trace_g, 1e-30)
#         out[lid] = r
#         print(f"  {lid:42s} {s.trace_g:12.4e} {s.sum_g:12.4e} {r:9.3f}")
#     return out

















# """
# stage0/hvp.py  (revised — weight-space double autograd)
# =============
# Attaches the L4 HVP to LayerStats.

# ROOT CAUSE OF PRIOR BUG (input-activation approach):
#   The old code differentiated w.r.t. the layer INPUT x_l.
#   This introduced (1/BT)^3 scaling (loss averaged, grad averaged, hvp_val averaged)
#   with BT=256 tokens → result was ~6e-8 of the true value → printed as 0.00.

# CORRECT APPROACH (vectozavr/llm-hessian, arXiv:2504.04520):
#   Differentiate w.r.t. WEIGHT PARAMETERS W_l.
#   Uses monkey-patching so loss = f(W_param), then double autograd on W_param.
#   Only one (1/BT) factor from the averaged loss → no cancellation.

# WHAT H_loss MEANS HERE:
#   H_loss_W = (1/d_out) * sum_j  d²L/dw_j²   in (d_in, d_in) space
#            ≈ Sigma * trace(H_out)/d_out
#            = H_lay * (trace(H_out)/d_out) / n_tokens

#   The gap ||H_loss - H_lay||_F / ||H_lay||_F ≈ |trace(H_out)/d_out/n_tokens - 1|
#   which will be < 1 and vary between layers — that IS the signal.

# SCALE CONVENTION:
#   We multiply hvp by n_tokens_calib so that
#       hvp(v) ≈ n_tokens * Sigma @ v * (trace(H_out)/d_out)
#       h_lay_mv(v) = n_tokens * Sigma @ v
#   making both sides comparable for the gap formula in gap.py.
# """

# from __future__ import annotations
# import torch
# import torch.nn.functional as F
# import numpy as np
# from typing import Dict, Callable, Optional
# from transformers import AutoModelForCausalLM
# from transformers.pytorch_utils import Conv1D

# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from gamma_bench.core.structures import LayerStats


# def _make_layer_hvp(
#     model:         AutoModelForCausalLM,
#     layer_id:      str,
#     input_ids:     torch.Tensor,
#     device:        str,
#     n_batches:     int   = 10,
#     batch_size:    int   = 2,
#     scale_factor:  float = 1.0,
# ) -> Callable[[np.ndarray], np.ndarray]:
#     """
#     Returns  hvp: v (numpy, d_in,) -> H_loss @ v (numpy, d_in,)

#     Algorithm
#     ---------
#     For each mini-batch:
#       1. Clone W_param = layer.weight  (requires_grad=True, frozen reference)
#       2. Monkey-patch layer.forward to use W_param instead of layer.weight
#       3. loss = model(batch, labels=batch).loss
#       4. g = dL/dW_param              (first grad, create_graph=True)
#       5. v_full = v broadcast over output dim (same shape as W_param)
#          gv    = (g * v_full).sum()   (scalar)
#       6. Hv_W = d(gv)/dW_param        (second grad, weight-space HVP)
#       7. sum Hv_W over output dim → (d_in,)
#     Average over batches, multiply by scale_factor.

#     Math note
#     ---------
#     Hv_W.sum(-1) ≈ (x x^T) @ v * trace(H_out) / d_out * BT_factor
#     where H_out = d²L/dy² at this layer's output.
#     With scale_factor = n_tokens_calib / n_tokens_per_batch, the result
#     matches the scale of H_lay = n_tokens_calib * Sigma used in gap.py.
#     """
#     target    = dict(model.named_modules())[layer_id]
#     is_conv1d = isinstance(target, Conv1D)
#     indices   = torch.randperm(len(input_ids))[:n_batches * batch_size]

#     # Belt-and-suspenders: disable non-differentiable SDPA backends
#     # (attn_implementation='eager' in capture.py already handles this;
#     #  these globals are extra safety for double autograd)
#     torch.backends.cuda.enable_flash_sdp(False)
#     torch.backends.cuda.enable_math_sdp(True)
#     torch.backends.cuda.enable_mem_efficient_sdp(False)

#     def hvp(v_np: np.ndarray) -> np.ndarray:
#         v       = torch.tensor(v_np, dtype=torch.float32, device=device)
#         hvp_acc = torch.zeros_like(v)
#         count   = 0

#         for start in range(0, len(indices), batch_size):
#             batch = input_ids[indices[start : start + batch_size]].to(device)

#             # ── 1. Differentiable weight copy ────────────────────────────
#             W_param = target.weight.detach().clone().requires_grad_(True)
#             # Conv1D:  W shape (d_in, d_out)   y = x @ W + b
#             # Linear:  W shape (d_out, d_in)   y = W x + b

#             bias_val: Optional[torch.Tensor] = (
#                 target.bias.detach()
#                 if hasattr(target, 'bias') and target.bias is not None
#                 else None
#             )

#             # ── 2. Monkey-patch layer.forward to use W_param ─────────────
#             def _make_fwd(Wp, bv, conv1d):
#                 def _fwd(self, x_in):
#                     if conv1d:
#                         sz = x_in.size()[:-1] + (Wp.shape[1],)
#                         y  = torch.addmm(bv, x_in.view(-1, x_in.size(-1)), Wp)
#                         return y.view(*sz)
#                     else:
#                         return F.linear(x_in, Wp, bv)
#                 return _fwd

#             orig_fwd      = target.forward
#             target.forward = _make_fwd(W_param, bias_val, is_conv1d).__get__(
#                 target, type(target)
#             )

#             # ── 3. Forward pass ───────────────────────────────────────────
#             try:
#                 loss = model(batch, labels=batch).loss
#             finally:
#                 target.forward = orig_fwd   # always restore

#             # ── 4. First grad (keep graph for 2nd pass) ───────────────────
#             g = torch.autograd.grad(loss, W_param, create_graph=True)[0]
#             if g.abs().sum() == 0:
#                 print(f"!!! WARNING: Zero gradients for weight param in {layer_id} !!!")
#             # g shape: (d_in, d_out) for Conv1D | (d_out, d_in) for Linear

#             # ── 5. v_full: broadcast v over output dimension ──────────────
#             # Conv1D: v (d_in,) → (d_in, d_out)   each col = v
#             # Linear: v (d_in,) → (d_out, d_in)   each row = v
#             if is_conv1d:
#                 v_full = v.unsqueeze(1).expand_as(W_param)   # (d_in, d_out)
#             else:
#                 v_full = v.unsqueeze(0).expand_as(W_param)   # (d_out, d_in)

#             gv = (g * v_full).sum()   # scalar

#             # ── 6. Second grad = HVP in weight space ──────────────────────
#             Hv_W = torch.autograd.grad(gv, W_param)[0]
#             # Hv_W shape: same as W_param

#             # ── 7. Reduce: sum over output dim → (d_in,) ─────────────────
#             if is_conv1d:
#                 hvp_acc += Hv_W.sum(dim=-1).detach()   # sum over d_out
#             else:
#                 hvp_acc += Hv_W.sum(dim=0).detach()    # sum over d_out

#             count += 1

#         result = (hvp_acc / max(count, 1)) * scale_factor
#         return result.cpu().numpy()

#     return hvp


# def attach_hvp(
#     stats:      Dict[str, LayerStats],
#     model:      AutoModelForCausalLM,
#     input_ids:  torch.Tensor,
#     device:     str,
#     n_batches:  int = 10,
#     batch_size: int = 2,
# ) -> None:
#     """
#     Attaches L4 HVP callable to each LayerStats in-place.

#     scale_factor = n_tokens_calib / n_tokens_hvp_per_batch
#     This puts hvp(v) on the same scale as h_lay_mv(v) = n_tokens * Sigma @ v
#     used in gap.py, so the gap formula ||H_loss - H_lay||_F / ||H_lay||_F
#     measures the structural/scale difference, not just an arbitrary unit choice.

#     Concretely:
#         n_tokens_calib  : from LayerStats (e.g. 512 samples × 128 tok = 65 536)
#         n_tokens_hvp    : n_batches × batch_size × seq_len (e.g. 10×2×128 = 2 560)
#         scale_factor    : 65 536 / 2 560 ≈ 25.6  (varies by layer if n_tokens differs)
#     """
#     seq_len       = input_ids.shape[1]
#     n_tok_per_run = n_batches * batch_size * seq_len

#     for layer_id, st in stats.items():
#         scale = float(st.n_tokens) / n_tok_per_run if n_tok_per_run > 0 else 1.0
#         stats[layer_id].hvp = _make_layer_hvp(
#             model, layer_id, input_ids, device,
#             n_batches, batch_size, scale_factor=scale,
#         )
#         print(f"  [hvp] attached → {layer_id:50s}  scale={scale:.1f} hvp={stats[layer_id].hvp}")
