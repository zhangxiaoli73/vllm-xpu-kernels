# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.utils import format_tc, seed_everything
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe as BaselineXpuFusedMoe
from vllm_xpu_kernels.fused_moe_pipeline_interface import (
    XpuFusedMoe as PipelineXpuFusedMoe,
)

DEVICE = "xpu"

FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (128, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]
EP_SIZES = [1, 4]

MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
    },
}


def _build_inputs(m, n, k, e, topk, dtype):
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 16
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 16

    scores = torch.randn((m, e), device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    w13 = w13.transpose(-1, -2).contiguous()
    w2 = w2.transpose(-1, -2).contiguous()

    return hidden_states, w13, w2, topk_weights, topk_ids


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=format_tc)
def test_fused_moe_stream_pipeline_accuracy(m, n, k, e, topk, ep_size, dtype):
    seed_everything(11)
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

    baseline_output = torch.empty_like(hidden_states)
    pipeline_output = torch.empty_like(hidden_states)

    baseline_impl.apply(
        output=baseline_output,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    pipeline_impl.apply(
        output=pipeline_output,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(pipeline_output, baseline_output, rtol=rtol,
                               atol=atol)