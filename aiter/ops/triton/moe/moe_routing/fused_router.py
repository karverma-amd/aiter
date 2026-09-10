# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.fused_router import (
    _default_config,
    _fused_router_topk_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_router_topk(
    hidden_states: torch.Tensor,  # [M, H]
    router_weight: torch.Tensor,  # [E, H]
    top_k: int,
    *,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused router GEMM + softmax + top-k (Gap 2b).

    Returns ``(topk_weight [M, top_k] fp32, topk_ids [M, top_k] i32)``, matching
    ``fused_moe_qwen_router.qwen_router_topk``. The ``[M, E]`` logits stay in
    registers -- never materialized to HBM -- which is the win over the Gap 2a
    matmul+``topk_gating`` composition. Numerically equals
    ``fused_router_topk_torch``; validate on-device before trusting.
    """
    _LOGGER.info(
        f"FUSED_ROUTER_TOPK: x={tuple(hidden_states.shape)} "
        f"w={tuple(router_weight.shape)} top_k={top_k}"
    )
    x = hidden_states.view(-1, hidden_states.shape[-1])
    M, K = x.shape
    E, Kw = router_weight.shape
    assert K == Kw, f"hidden K={K} != router_weight K={Kw}"

    topk_weight = torch.empty((M, top_k), device=x.device, dtype=torch.float32)
    topk_ids = torch.empty((M, top_k), device=x.device, dtype=torch.int32)

    cfg = _default_config(M)
    block_m = cfg.pop("BLOCK_M")
    block_k = cfg.pop("BLOCK_K")
    block_n = triton.next_power_of_2(E)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _fused_router_topk_kernel[grid](
        x,
        router_weight,
        topk_weight,
        topk_ids,
        M,
        E,
        K,
        routed_scaling_factor,
        x.stride(0),
        x.stride(1),
        # router_weight is [E, K]; address it as [K, E] via swapped strides.
        router_weight.stride(1),
        router_weight.stride(0),
        topk_weight.stride(0),
        topk_weight.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        TOPK=top_k,
        TOPK_PAD=triton.next_power_of_2(top_k),
        NORM_TOPK=norm_topk_prob,
        **cfg,
    )
    return topk_weight, topk_ids


def fused_router_topk_torch(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for :func:`fused_router_topk` -- mirrors ``Qwen3NextTopKRouter``."""
    logits = torch.nn.functional.linear(hidden_states, router_weight)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weight, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        weight = weight / weight.sum(dim=-1, keepdim=True)
    weight = weight * routed_scaling_factor
    return weight.to(torch.float32), ids.to(torch.int32)
