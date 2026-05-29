# SPDX-License-Identifier: Apache-2.0
import os
import sys

# Ensure project root is in sys.path for direct execution (e.g. mpirun)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

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


def _build_inputs(m, n, k, e, topk, dtype, total_experts=None):
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 16
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 16
    score_experts = total_experts if total_experts is not None else e
    scores = torch.randn((m, score_experts), device=DEVICE, dtype=torch.float32)
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


if __name__ == "__main__":
    import torch.distributed as dist

    rank = int(os.environ.get("PMI_RANK", "0"))
    world_size = int(os.environ.get("PMI_SIZE", "1"))

    # Set per-rank device affinity
    torch.xpu.set_device(rank)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    backend = os.environ.get("XPU_KERNEL_DIST_BACKEND", "xccl")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        seed_everything(13)

        n, k = 6144, 2048
        e = 32              # 32 local experts per device
        topk = 8
        dtype = torch.bfloat16
        warmup = 3
        iters = 5
        ep_size = world_size
        total_experts = e * ep_size  # 128 total experts
        threshold = 256
        token_counts = [2048, 1024, 512]

        # Shared weights across all m values
        w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 16
        w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 16
        w13 = w13.transpose(-1, -2).contiguous()
        w2 = w2.transpose(-1, -2).contiguous()

        baseline_impl = BaselineXpuFusedMoe(
            w13=w13, w13_scales=None, w13_bias=None,
            w2=w2, w2_scales=None, w2_bias=None,
            n_experts_per_token=topk, activation="silu",
            num_experts=e, ep_rank=rank, ep_size=ep_size,
        )

        pipeline_impl = PipelineXpuFusedMoe(
            w13=w13, w13_scales=None, w13_bias=None,
            w2=w2, w2_scales=None, w2_bias=None,
            n_experts_per_token=topk, activation="silu",
            num_experts=e, ep_rank=rank, ep_size=ep_size,
            process_group=dist.group.WORLD,
            pipeline_depth=4, pipeline_streams=2,
            expert_token_threshold=threshold,
        )

        # --- Fair baseline: allgather + local compute + reduce_scatter ---
        def _baseline_with_comms(hs, tw, ti, output):
            hidden_size = hs.shape[1]
            gathered_h = [torch.empty_like(hs) for _ in range(world_size)]
            gathered_w = [torch.empty_like(tw) for _ in range(world_size)]
            gathered_i = [torch.empty_like(ti) for _ in range(world_size)]
            dist.all_gather(gathered_h, hs)
            dist.all_gather(gathered_w, tw)
            dist.all_gather(gathered_i, ti)

            all_h = torch.cat(gathered_h, dim=0)
            all_w = torch.cat(gathered_w, dim=0)
            all_i = torch.cat(gathered_i, dim=0)

            full_partial = torch.empty(
                all_h.shape[0], hidden_size, dtype=output.dtype, device=output.device,
            )
            baseline_impl.apply(
                output=full_partial,
                hidden_states=all_h,
                topk_weights=all_w,
                topk_ids=all_i,
            )
            partial_chunks = list(full_partial.chunk(world_size, dim=0))
            dist.reduce_scatter(output, partial_chunks, op=dist.ReduceOp.SUM)

        def _measure_baseline_ms(hs, tw, ti):
            out = torch.empty_like(hs)
            for _ in range(warmup):
                _baseline_with_comms(hs, tw, ti, out)
            torch.xpu.synchronize()

            start_ev = torch.xpu.Event(enable_timing=True)
            end_ev = torch.xpu.Event(enable_timing=True)
            start_ev.record()
            for _ in range(iters):
                _baseline_with_comms(hs, tw, ti, out)
            end_ev.record()
            torch.xpu.synchronize()
            return start_ev.elapsed_time(end_ev) / iters

        # Header
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Config: e_local={e}, e_total={total_experts}, topk={topk}, "
                  f"ws={world_size}, threshold={threshold}")
            print(f"  n(intermediate)={n}, k(hidden)={k}, dtype={dtype}")
            print(f"{'='*70}")
            print(f"{'m':>6} {'rpe':>5} {'Baseline':>12} {'Pipeline':>12} "
                  f"{'Speedup':>10} {'Note':>15}")
            print(f"{'-'*6} {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*15}")

        for m in token_counts:
            hidden_states = torch.randn((m, k), device=DEVICE, dtype=dtype) / 16
            scores = torch.randn(
                (m, total_experts), device=DEVICE, dtype=torch.float32,
            )
            topk_weights, topk_ids = torch.topk(
                scores, k=topk, dim=-1, sorted=False,
            )

            rpe = m * topk // total_experts

            # Measure baseline (allgather + compute + reduce_scatter)
            base_ms = _measure_baseline_ms(hidden_states, topk_weights, topk_ids)

            # Measure pipeline (ring pipeline with threshold)
            pipe_ms = _measure_ms(
                pipeline_impl, hidden_states, topk_weights, topk_ids,
                warmup, iters,
            )

            speedup = base_ms / pipe_ms
            note = "FASTER" if speedup > 1.05 else (
                "SLOWER" if speedup < 0.95 else "~same")

            if rank == 0:
                print(f"{m:>6} {rpe:>5} {base_ms:>11.3f}ms {pipe_ms:>11.3f}ms "
                      f"{speedup:>9.2f}x {note:>15}")

        if rank == 0:
            print(f"{'='*70}")
    finally:
        dist.destroy_process_group()
