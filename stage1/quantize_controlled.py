"""
stage1/quantize_controlled.py
==============================
Realizes the "controlled path": takes any AllocationMap (Gamma-field, any
Ladder rung, or later a reused method's extracted map) and applies it to a
REAL model's weights via the SAME UniformGridQuantizer every method shares,
producing a quantized model whose activations can be measured against the
original with the SAME GeometryProbe used everywhere else.

This file exists so quantization arithmetic is never duplicated:
UniformGridQuantizer.apply() in structures.py is called exactly as-is, on
whatever weight tensor a layer happens to have (Conv1D's (d_in, d_out) or
nn.Linear's (d_out, d_in) -- quantization is elementwise, so orientation
never matters here). Everything below is model plumbing: finding the right
modules, copying the model so the original stays untouched for comparison,
and writing the quantized arrays back.

Reusable beyond Stage 1: any later stage doing a controlled-path comparison
(bit-width sweep, SOTA controlled path, rotation + allocation) calls this the
same way.
"""

from __future__ import annotations
import copy
import torch
import numpy as np
from typing import Dict
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gamma_bench.core.structures import AllocationMap, UniformGridQuantizer


def apply_controlled_quantization(
    model:      AutoModelForCausalLM,
    alloc:      AllocationMap,
    quantizer:  UniformGridQuantizer,
    device:     str,
) -> AutoModelForCausalLM:
    """
    Returns a NEW model (the original is left untouched) with every layer
    named in alloc.bits quantized in place per alloc, using `quantizer` --
    the exact same Quantizer object every other method in the panel uses.
    """
    model_q = copy.deepcopy(model)
    modules_dict = dict(model_q.named_modules())

    # 1. extract real weights for every targeted layer
    weights: Dict[str, np.ndarray] = {}
    for layer_id in alloc.bits:
        module = modules_dict.get(layer_id)
        if module is None or not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        weights[layer_id] = module.weight.detach().cpu().float().numpy()

    # 2. pure quantization math -- unmodified shared Quantizer, array in/array out
    quantized = quantizer.apply(weights, alloc)

    # 3. write the quantized arrays back into the copied model's parameters
    for layer_id, w_q in quantized.items():
        module = modules_dict[layer_id]
        orig_dtype = module.weight.dtype
        with torch.no_grad():
            module.weight.copy_(
                torch.from_numpy(w_q).to(dtype=orig_dtype, device=module.weight.device)
            )

    return model_q.to(device).eval()