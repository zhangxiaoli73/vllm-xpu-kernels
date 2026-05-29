# SPDX-License-Identifier: Apache-2.0
import os
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_pipeline_interface import XpuFusedMoe

DEVICE = "xpu"


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dist_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    backend = os.environ.get("XPU_KERNEL_DIST_BACKEND", "gloo")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    try:
        seed_everything(17 + rank)
        m = 32
        k = 256
        n = 512
        local_experts = 8
        topk = 1

        hidden_states = torch.randn((m, k), device=DEVICE, dtype=torch.bfloat16) / 16
        w13 = torch.randn(
            (local_experts, 2 * n, k),
            device=DEVICE,
            dtype=torch.bfloat16,
        ) / 16
        w2 = torch.randn(
            (local_experts, k, n),
            device=DEVICE,
            dtype=torch.bfloat16,
        ) / 16

        total_experts = local_experts * world_size
        scores = torch.randn((m, total_experts), device=DEVICE, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

        w13 = w13.transpose(-1, -2).contiguous()
        w2 = w2.transpose(-1, -2).contiguous()

        moe_impl = XpuFusedMoe(
            w13=w13,
            w13_scales=None,
            w13_bias=None,
            w2=w2,
            w2_scales=None,
            w2_bias=None,
            n_experts_per_token=topk,
            activation="silu",
            num_experts=local_experts,
            ep_rank=rank,
            ep_size=world_size,
            process_group=dist.group.WORLD,
            pipeline_depth=4,
            pipeline_streams=2,
        )

        out = torch.empty_like(hidden_states)
        moe_impl.apply(
            output=out,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all().item()
    finally:
        dist.destroy_process_group()


def test_fused_moe_stream_pipeline_dist_ring_smoke():
    if os.environ.get("XPU_KERNEL_RUN_DIST_TESTS", "0") != "1":
        pytest.skip("Set XPU_KERNEL_RUN_DIST_TESTS=1 to run dist ring smoke test.")

    world_size = int(os.environ.get("XPU_KERNEL_DIST_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.skip("Distributed ring smoke test requires world_size >= 2.")

    if not hasattr(torch, "xpu"):
        pytest.skip("XPU is required for this distributed smoke test.")

    port = _find_free_port()
    mp.spawn(
        _dist_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )
