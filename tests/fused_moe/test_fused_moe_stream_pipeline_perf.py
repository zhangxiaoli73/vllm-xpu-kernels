# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe as BaselineXpuFusedMoe
from vllm_xpu_kernels.fused_moe_pipeline_interface import (
    XpuFusedMoe as PipelineXpuFusedMoe,
)

DEVICE = "xpu"

MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(128, 5120, 8192)],
    },
}


def _build_inputs(m, n, k, e, topk, dtype):
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 16
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 16
    scores = torch.randn((m, e), device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    return (
        hidden_states,
        w13.transpose(-1, -2).contiguous(),
        w2.transpose(-1, -2).contiguous(),
        topk_weights,
        topk_ids,
    )


def _measure_ms(moe_impl, hidden_states, topk_weights, topk_ids, warmup, iters):
    out = torch.empty_like(hidden_states)
    for _ in range(warmup):
        moe_impl.apply(
            output=out,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        moe_impl.apply(
            output=out,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    end_event.record()
    torch.xpu.synchronize()
    return start_event.elapsed_time(end_event) / iters


@pytest.mark.parametrize("m,n,k", [(1024, 5120, 8192)])
@pytest.mark.parametrize("ep_size", [1, 4])
def test_fused_moe_stream_pipeline_perf_regression(m, n, k, ep_size):
    if os.environ.get("XPU_KERNEL_RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set XPU_KERNEL_RUN_PERF_TESTS=1 to run performance UT.")

    seed_everything(13)

    e = 16
    topk = 1
    dtype = torch.bfloat16
    warmup = 5
    iters = 20
    allowed_slowdown = float(
        os.environ.get("XPU_KERNEL_PIPELINE_PERF_ALLOW_SLOWDOWN", "1.10")
    )

    hidden_states, w13, w2, topk_weights, topk_ids = _build_inputs(
        m=m,
        n=n,
        k=k,
        e=e,
        topk=topk,
        dtype=dtype,
    )

    baseline_impl = BaselineXpuFusedMoe(
        w13=w13,
        w13_scales=None,
        w13_bias=None,
        w2=w2,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
        ep_rank=0,
        ep_size=ep_size,
    )
    pipeline_impl = PipelineXpuFusedMoe(
        w13=w13,
        w13_scales=None,
        w13_bias=None,
        w2=w2,
        w2_scales=None,
        w2_bias=None,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
        ep_rank=0,
        ep_size=ep_size,
        pipeline_depth=4,
        pipeline_streams=2,
    )

    baseline_ms = _measure_ms(
        moe_impl=baseline_impl,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        warmup=warmup,
        iters=iters,
    )
    pipeline_ms = _measure_ms(
        moe_impl=pipeline_impl,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        warmup=warmup,
        iters=iters,
    )

    assert pipeline_ms <= baseline_ms * allowed_slowdown, (
        "Pipeline regression detected: "
        f"pipeline={pipeline_ms:.4f}ms, baseline={baseline_ms:.4f}ms, "
        f"allowed_slowdown={allowed_slowdown:.3f}"
    )