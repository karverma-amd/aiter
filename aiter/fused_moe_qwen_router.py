# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next MoE: fold the router GEMM+top-k and the shared-expert gate into
the AITER fused-MoE call.

The stock ``aiter.fused_moe`` requires precomputed ``topk_weight``/``topk_ids``
and can only weight a shared expert by a constant scalar. This module adds the
two tensors it cannot ingest for Qwen3-Next:

  - ``router_weight`` ``[E, H]``  -- the top-k gate projection (Gap 2a / 2b).
  - ``shared_gate_weight`` ``[1, H]`` -- the per-token sigmoid gate (Gap 1).

Semantics match transformers ``Qwen3NextTopKRouter`` / ``Qwen3NextSparseMoeBlock``:

    logits = hidden @ router_weight.T                 # [M, E]
    probs  = softmax(logits, dim=-1, dtype=fp32)
    w, ids = topk(probs, top_k)                        # [M, top_k]
    if norm_topk_prob: w = w / w.sum(-1, keepdim=True) # renorm over selected k
    routed = fused_moe(hidden, w1, w2, w, ids, ...)
    shared = sigmoid(hidden @ shared_gate_weight.T) * shared_expert(hidden)
    out    = routed + shared
"""

from collections.abc import Callable

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.topk import topk_gating


def qwen_router_topk(
    hidden_states: torch.Tensor,  # [M, H]
    router_weight: torch.Tensor,  # [E, H]
    top_k: int,
    *,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k routing. Returns ``(topk_weight [M, k] fp32, topk_ids [M, k] i32)``.

    ``fused=False`` (Gap 2a): router GEMM materializes ``[M, E]`` logits, then the
    fused ``topk_gating`` softmax kernel selects the top-k. ``fused=True``
    (Gap 2b): a single Triton kernel computes the logits and top-k without
    writing the ``[M, E]`` logits to HBM.
    """
    M, H = hidden_states.shape
    E = router_weight.shape[0]
    assert router_weight.shape == (
        E,
        H,
    ), f"router_weight must be [E, H]; got {tuple(router_weight.shape)} vs H={H}"

    if fused:
        from aiter.ops.triton.moe.moe_routing.fused_router import fused_router_topk

        return fused_router_topk(
            hidden_states,
            router_weight,
            top_k,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )

    # Gap 2a: compose the existing tested primitives.
    logits = torch.matmul(hidden_states, router_weight.transpose(0, 1))
    topk_weight = torch.empty(
        (M, top_k), dtype=dtypes.fp32, device=hidden_states.device
    )
    topk_ids = torch.empty((M, top_k), dtype=dtypes.i32, device=hidden_states.device)
    # softmax scoring is normalized over all E; the top-k renorm below is separate.
    topk_gating(
        topk_weight,
        topk_ids,
        logits,
        correction_bias=None,
        need_renorm=False,
        routed_scaling_factor=routed_scaling_factor,
        score_func="softmax",
    )
    if norm_topk_prob:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return topk_weight, topk_ids


def shared_expert_gate(
    hidden_states: torch.Tensor,  # [M, H]
    shared_gate_weight: torch.Tensor,  # [1, H]
) -> torch.Tensor:
    """Per-token sigmoid gate for the shared expert. Returns ``[M, 1]`` in the
    hidden dtype (Gap 1). Replaces the constant ``share_expert_score`` used by
    ``fused_moe_dp_share_expert``."""
    H = hidden_states.shape[1]
    assert shared_gate_weight.shape == (
        1,
        H,
    ), f"shared_gate_weight must be [1, H]; got {tuple(shared_gate_weight.shape)}"
    gate = torch.matmul(
        hidden_states.to(dtypes.fp32),
        shared_gate_weight.to(dtypes.fp32).transpose(0, 1),
    )
    return torch.sigmoid(gate).to(hidden_states.dtype)


def qwen_router_moe(
    hidden_states: torch.Tensor,  # [M, H]
    w1: torch.Tensor,  # [E, inter_dim*2, H]
    w2: torch.Tensor,  # [E, H, inter_dim]
    router_weight: torch.Tensor,  # [E, H]
    *,
    top_k: int,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    quant_type: QuantType = QuantType.per_1x128,
    activation: ActivationType = ActivationType.Silu,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    shared_gate_weight: torch.Tensor | None = None,  # [1, H]
    shared_expert: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fused_router: bool = False,
    **fused_moe_kwargs,
) -> torch.Tensor:
    """End-to-end Qwen3-Next MoE block: router top-k -> routed experts (+ gated
    shared expert). ``shared_expert`` returns the *ungated* shared output; the
    per-token sigmoid gate is applied here so the shared MLP's own quantization
    stays outside AITER."""
    topk_weight, topk_ids = qwen_router_topk(
        hidden_states,
        router_weight,
        top_k,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        fused=fused_router,
    )

    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=activation,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        **fused_moe_kwargs,
    )

    if shared_gate_weight is not None:
        assert (
            shared_expert is not None
        ), "shared_gate_weight requires a shared_expert callable"
        gate = shared_expert_gate(hidden_states, shared_gate_weight)
        out = out + gate * shared_expert(hidden_states)
    return out
