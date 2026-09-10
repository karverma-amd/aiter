# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Op test for the Qwen3-Next fused-MoE router extension
(``aiter/fused_moe_qwen_router.py``): the router GEMM+softmax+top-k
(``qwen_router_topk``, Gap 2a compose vs Gap 2b fused kernel), the per-token
shared-expert sigmoid gate (``shared_expert_gate``), and the end-to-end
``qwen_router_moe`` orchestrator against a torch reference.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import QuantType, dtypes
from aiter.fused_moe import torch_moe
from aiter.fused_moe_qwen_router import (
    qwen_router_moe,
    qwen_router_topk,
    shared_expert_gate,
)
from aiter.int4_utils import *  # noqa: F401,F403
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe.moe_routing.fused_router import fused_router_topk_torch
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def _dense(weight, ids, n_experts):
    """Scatter (weight, id) into a dense [M, E] map for an order-invariant compare."""
    d = torch.zeros(
        (weight.shape[0], n_experts), dtype=dtypes.fp32, device=weight.device
    )
    d.scatter_(1, ids.long(), weight.to(dtypes.fp32))
    return d


@benchmark()
def test_qwen_router_topk(token_num, model_dim, n_experts, top_k, dtype):
    """Router matmul + softmax + top-k. Two candidates against the torch reference:
    ``gap2a`` = matmul + tuned ``topk_gating``; ``gap2b`` = single fused Triton
    kernel (logits never hit HBM). Correctness is tie-invariant (sorted top-k
    weights; ids compared as a dense scatter)."""
    hidden = torch.randn((token_num, model_dim), dtype=dtype) * 0.1
    router_weight = torch.randn((n_experts, model_dim), dtype=dtype) * 0.05

    ref_w, ref_ids = fused_router_topk_torch(hidden, router_weight, top_k)
    ref_sorted = torch.sort(ref_w, dim=-1).values
    ref_dense = _dense(ref_w, ref_ids, n_experts)

    candidates = {
        "gap2a": lambda: qwen_router_topk(hidden, router_weight, top_k, fused=False),
        "gap2b": lambda: qwen_router_topk(hidden, router_weight, top_k, fused=True),
    }

    # router GEMM roofline: [M,H] x [H,E] -> [M,E]
    flops = 2 * token_num * model_dim * n_experts
    nbytes = (
        token_num * model_dim + n_experts * model_dim + token_num * n_experts
    ) * hidden.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out_w, out_ids), us = run_perftest(fn)
        # tie-invariant weight check + dense id/weight map check
        err = checkAllclose(
            ref_sorted,
            torch.sort(out_w, dim=-1).values,
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: topk weights",
        )
        checkAllclose(
            ref_dense,
            _dense(out_w, out_ids, n_experts),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0.02,  # rare exact-tie rows may pick a different equal expert
            msg=f"{name}: topk dense map",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_qwen_router_moe(token_num, model_dim, inter_dim, n_experts, top_k, dtype):
    """End-to-end block: router top-k -> routed experts + gated shared expert, vs a
    torch reference. Validates the ``qwen_router_moe`` wiring. Uses per_Token i8
    (the reference path ``torch_moe`` validates exactly); the orchestrator is
    quant-agnostic, so the wiring proof carries over to the model's per_1x128 fp8."""
    q_dtype = dtypes.i8
    quant_type = QuantType.per_Token
    hidden = torch.randn((token_num, model_dim), dtype=dtype) * 0.1
    router_weight = torch.randn((n_experts, model_dim), dtype=dtype) * 0.05
    shared_gate_weight = torch.randn((1, model_dim), dtype=dtype) * 0.05
    w1 = torch.randn((n_experts, inter_dim * 2, model_dim), dtype=dtype) / 10.0
    w2 = torch.randn((n_experts, model_dim, inter_dim), dtype=dtype) / 10.0
    # a stand-in ungated shared expert (its own compute stays outside AITER)
    shared_w = torch.randn((model_dim, model_dim), dtype=dtype) * 0.02

    def shared_expert(x):
        return x @ shared_w.T

    torch_quant = aiter.get_torch_quant(quant_type)
    w1_q, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
    w2_q, w2_scale = torch_quant(w2, quant_dtype=q_dtype)

    topk_w, topk_ids = fused_router_topk_torch(hidden, router_weight, top_k)
    ref_routed = torch_moe(hidden, w1_q, w2_q, topk_w, topk_ids, w1_scale, w2_scale)
    gate = torch.sigmoid(hidden.float() @ shared_gate_weight.float().T).to(dtype)
    ref = ref_routed + gate * shared_expert(hidden)

    w1_s = shuffle_weight(w1_q, layout=(16, 16))
    w2_s = shuffle_weight(w2_q, layout=(16, 16))

    out, us = run_perftest(
        qwen_router_moe,
        hidden,
        w1_s,
        w2_s,
        router_weight,
        top_k=top_k,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        shared_gate_weight=shared_gate_weight,
        shared_expert=shared_expert,
    )
    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, msg="qwen_router_moe")

    # shared-gate sub-check (Gap 1)
    g = shared_expert_gate(hidden, shared_gate_weight)
    checkAllclose(
        gate.float(), g.float(), rtol=1e-2, atol=1e-2, msg="shared_expert_gate"
    )
    return {"gfx": get_gfx(), "us": us, "err": err}


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("qwen_router_moe unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-t",
        "--tokenNum",
        type=int,
        nargs="*",
        default=[1, 8, 16, 32, 64, 128, 256],
        help="number of tokens, e.g. -t 1 8 64",
    )
    parser.add_argument(
        "-dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(2048, 512)],
        help="(model_dim, inter_dim), e.g. -dim 2048,512",
    )
    parser.add_argument("-e", "--expert", type=int, nargs="*", default=[512])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[10])
    args = parser.parse_args()

    # Table 1: router top-k (gap2a vs gap2b)
    for dtype in args.dtype:
        rows = []
        for (model_dim, _inter), E, k, M in itertools.product(
            args.dim, args.expert, args.topk, args.tokenNum
        ):
            rows.append(test_qwen_router_topk(M, model_dim, E, k, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "qwen_router_topk summary (markdown):\n%s", df.to_markdown(index=False)
        )

    # Table 2: end-to-end qwen_router_moe wiring
    for dtype in args.dtype:
        rows = []
        for (model_dim, inter_dim), E, k, M in itertools.product(
            args.dim, args.expert, args.topk, args.tokenNum
        ):
            rows.append(test_qwen_router_moe(M, model_dim, inter_dim, E, k, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "qwen_router_moe summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
