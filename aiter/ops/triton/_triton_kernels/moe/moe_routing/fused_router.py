# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton._triton_kernels.moe.moe_routing.topk import (
    fpval_to_key,
    key_to_fpval,
)

_fused_router_topk_repr = make_kernel_repr(
    "_fused_router_topk_kernel",
    ["BLOCK_M", "BLOCK_K", "BLOCK_N", "TOPK", "NORM_TOPK"],
)


@triton.jit(repr=_fused_router_topk_repr)
def _fused_router_topk_kernel(
    X_ptr,  # [M, K] hidden states
    W_ptr,  # router_weight, addressed as [K, N] via swapped strides
    topk_weights_ptr,  # [M, TOPK] fp32
    topk_ids_ptr,  # [M, TOPK] i32
    M,
    N,
    K,
    routed_scaling_factor,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_tw_m,
    stride_tw_n,
    stride_ti_m,
    stride_ti_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,  # next_pow2(TOPK); tl.arange requires a power-of-2 length
    NORM_TOPK: tl.constexpr,
):
    """Fused router GEMM + softmax + top-k for Qwen3-Next.

    One program handles BLOCK_M tokens against all N experts (BLOCK_N >= N).
    Logits are formed in registers and reduced in-kernel, so the [M, N] router
    logits are never written to HBM (the Gap 2b fold). Matches
    ``Qwen3NextTopKRouter``: full softmax over N, top-k, optional renorm.

    NOTE: at Qwen3-Next decode/prefill shapes this fused single launch does not
    beat the two-launch Gap 2a path (tuned rocBLAS GEMM + ASM ``topk_gating``);
    the router GEMM is tiny so the ~[M,E] HBM round-trip saved is dwarfed by the
    untuned-Triton-vs-tuned-ASM gap. Kept behind ``fused_router=False`` as a
    correct, reusable building block; see the module benchmark.
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_topk = tl.arange(0, TOPK_PAD)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_topk = offs_topk < TOPK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K
        X_ptrs = X_ptr + (
            offs_m[:, None] * stride_xm + offs_k_iter[None, :] * stride_xk
        )
        W_ptrs = W_ptr + (
            offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        acc = tl.dot(x, w, acc=acc)

    # Full softmax over the N valid experts (invalid lanes -> 0 probability).
    acc = tl.where(mask_n[None, :], acc, float("-inf"))
    row_max = tl.max(acc, axis=1)[:, None]
    probs = tl.exp(acc - row_max)
    probs = probs / tl.sum(probs, axis=1)[:, None]

    # Top-k by probability == top-k by logit (softmax is monotone): pack each
    # expert as ``(fpval_to_key(prob) << 16) | idx`` so one ``tl.topk`` pulls the
    # largest probs with the expert id in the low bits (same packing trick as
    # ``streaming_topk``). The recovered value is the fp32 prob bit-for-bit, so
    # weights match the reference exactly.
    packed = (fpval_to_key(probs.to(tl.uint32, bitcast=True)).to(tl.uint64) << 16) | (
        offs_n[None, :].to(tl.uint64)
    )
    top = tl.topk(packed, TOPK_PAD, dim=1)  # [BLOCK_M, TOPK_PAD] largest, bitonic
    top = tl.flip(tl.sort(top, dim=1), 1)  # ascending sort then flip -> descending
    ids_buf = (top & 0xFFFF).to(tl.int32)
    wts_buf = key_to_fpval((top >> 16).to(tl.uint32)).to(tl.float32, bitcast=True)
    # Keep only the true top-TOPK (TOPK_PAD == next_pow2(TOPK) >= TOPK); zero the
    # padding lanes before the renorm so they do not pollute the denominator.
    keep = offs_topk[None, :] < TOPK
    wts_buf = tl.where(keep, wts_buf, 0.0)
    ids_buf = tl.where(keep, ids_buf, 0)

    if NORM_TOPK:
        wts_buf = wts_buf / tl.sum(wts_buf, axis=1)[:, None]
    wts_buf = wts_buf * routed_scaling_factor

    tw_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_tw_m
        + offs_topk[None, :] * stride_tw_n
    )
    ti_ptrs = (
        topk_ids_ptr + offs_m[:, None] * stride_ti_m + offs_topk[None, :] * stride_ti_n
    )
    store_mask = mask_m[:, None] & mask_topk[None, :]
    tl.store(tw_ptrs, wts_buf, mask=store_mask)
    tl.store(ti_ptrs, ids_buf, mask=store_mask)


def _default_config(M: int) -> dict:
    # Single-tile kernel: BLOCK_N == next_pow2(E) covers all experts in one block.
    # An N-tiled online-softmax variant was tried to grow BLOCK_M but was slower at
    # every M (the extra per-tile bitonic merges cost more than they save), so the
    # simple single-tile form is kept.
    del M
    return {"BLOCK_M": 16, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2}
