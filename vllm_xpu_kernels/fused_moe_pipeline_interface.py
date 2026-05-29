# SPDX-License-Identifier: Apache-2.0
import math
import os
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist

from .fused_moe_interface import XpuFusedMoe as _BaseXpuFusedMoe

PIPELINE_DEPTH_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_DEPTH"
PIPELINE_STREAMS_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_STREAMS"
PIPELINE_MIN_ROWS_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_MIN_ROWS"


def _read_int_env(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class XpuFusedMoe(_BaseXpuFusedMoe):
    """Fused-MoE with host-side stream pipeline over token micro-batches."""

    def __init__(
        self,
        w13,
        w13_scales,
        w13_bias,
        w2,
        w2_scales,
        w2_bias,
        n_experts_per_token,
        activation,
        num_experts,
        ep_rank=0,
        ep_size=1,
        expert_map=None,
        is_fp8=False,
        is_int4=False,
        is_mxfp4=False,
        is_mxfp8=False,
        is_block_fp8=False,
        process_group=None,
        pipeline_depth=None,
        pipeline_streams=None,
        min_compute_rows=None,
    ):
        super().__init__(
            w13=w13,
            w13_scales=w13_scales,
            w13_bias=w13_bias,
            w2=w2,
            w2_scales=w2_scales,
            w2_bias=w2_bias,
            n_experts_per_token=n_experts_per_token,
            activation=activation,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expert_map=expert_map,
            is_fp8=is_fp8,
            is_int4=is_int4,
            is_mxfp4=is_mxfp4,
            is_mxfp8=is_mxfp8,
            is_block_fp8=is_block_fp8,
        )
        self.pipeline_depth = max(
            1,
            pipeline_depth
            if pipeline_depth is not None
            else _read_int_env(PIPELINE_DEPTH_ENV, 4),
        )
        self.pipeline_streams = max(
            1,
            pipeline_streams
            if pipeline_streams is not None
            else _read_int_env(PIPELINE_STREAMS_ENV, 2),
        )
        self.min_compute_rows = max(
            1,
            min_compute_rows
            if min_compute_rows is not None
            else _read_int_env(PIPELINE_MIN_ROWS_ENV, 1),
        )
        self.process_group = process_group

    def _build_streams(self) -> list:
        xpu_mod = getattr(torch, "xpu", None)
        if xpu_mod is None or not hasattr(xpu_mod, "Stream"):
            return []
        return [xpu_mod.Stream() for _ in range(self.pipeline_streams)]

    def _stream_context(self, stream):
        xpu_mod = getattr(torch, "xpu", None)
        if xpu_mod is None or stream is None or not hasattr(xpu_mod, "stream"):
            return nullcontext()
        return xpu_mod.stream(stream)

    def _sync_xpu(self):
        xpu_mod = getattr(torch, "xpu", None)
        if xpu_mod is not None and hasattr(xpu_mod, "synchronize"):
            xpu_mod.synchronize()

    def _is_ring_enabled(self) -> bool:
        if self.ep_size <= 1:
            return False
        if not _is_dist_ready():
            return False
        world_size = dist.get_world_size(self.process_group)
        if world_size != self.ep_size:
            warnings.warn(
                (
                    "Distributed world size does not match ep_size "
                    f"(world_size={world_size}, ep_size={self.ep_size}). "
                    "Fallback to local non-ring execution."
                ),
                stacklevel=2,
            )
            return False
        return True

    def _run_local_chunk(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map,
    ):
        if hidden_states.shape[0] < self.min_compute_rows:
            self._apply_ref(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
        else:
            super()._apply_kernel(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )

    def _ring_send_recv_package(self, package, send_rank, recv_rank):
        recv_package = {
            "owner": torch.empty_like(package["owner"]),
            "valid_rows": torch.empty_like(package["valid_rows"]),
            "hidden": torch.empty_like(package["hidden"]),
            "topk_weights": torch.empty_like(package["topk_weights"]),
            "topk_ids": torch.empty_like(package["topk_ids"]),
            "accum": torch.empty_like(package["accum"]),
        }

        ops = [
            dist.P2POp(dist.isend, package["owner"], send_rank, self.process_group),
            dist.P2POp(dist.irecv, recv_package["owner"], recv_rank, self.process_group),
            dist.P2POp(dist.isend, package["valid_rows"], send_rank, self.process_group),
            dist.P2POp(dist.irecv, recv_package["valid_rows"], recv_rank, self.process_group),
            dist.P2POp(dist.isend, package["hidden"], send_rank, self.process_group),
            dist.P2POp(dist.irecv, recv_package["hidden"], recv_rank, self.process_group),
            dist.P2POp(
                dist.isend,
                package["topk_weights"],
                send_rank,
                self.process_group,
            ),
            dist.P2POp(
                dist.irecv,
                recv_package["topk_weights"],
                recv_rank,
                self.process_group,
            ),
            dist.P2POp(dist.isend, package["topk_ids"], send_rank, self.process_group),
            dist.P2POp(dist.irecv, recv_package["topk_ids"], recv_rank, self.process_group),
            dist.P2POp(dist.isend, package["accum"], send_rank, self.process_group),
            dist.P2POp(dist.irecv, recv_package["accum"], recv_rank, self.process_group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        return recv_package

    def _clone_package(self, package):
        return {
            "owner": package["owner"].clone(),
            "valid_rows": package["valid_rows"].clone(),
            "hidden": package["hidden"].clone(),
            "topk_weights": package["topk_weights"].clone(),
            "topk_ids": package["topk_ids"].clone(),
            "accum": package["accum"].clone(),
        }

    def _make_dispatch_package(
        self,
        rank,
        hidden_chunk,
        weights_chunk,
        ids_chunk,
        output_dtype,
        padded_rows,
    ):
        valid_rows = hidden_chunk.shape[0]
        hidden_size = hidden_chunk.shape[1]
        topk = ids_chunk.shape[1]

        package = {
            "owner": torch.tensor([rank], dtype=torch.int32, device=hidden_chunk.device),
            "valid_rows": torch.tensor([valid_rows], dtype=torch.int32, device=hidden_chunk.device),
            "hidden": torch.zeros(
                (padded_rows, hidden_size),
                dtype=hidden_chunk.dtype,
                device=hidden_chunk.device,
            ),
            "topk_weights": torch.zeros(
                (padded_rows, topk),
                dtype=weights_chunk.dtype,
                device=weights_chunk.device,
            ),
            "topk_ids": torch.zeros(
                (padded_rows, topk),
                dtype=ids_chunk.dtype,
                device=ids_chunk.device,
            ),
            "accum": torch.zeros(
                (padded_rows, hidden_size),
                dtype=output_dtype,
                device=hidden_chunk.device,
            ),
        }
        if valid_rows > 0:
            package["hidden"][:valid_rows].copy_(hidden_chunk)
            package["topk_weights"][:valid_rows].copy_(weights_chunk)
            package["topk_ids"][:valid_rows].copy_(ids_chunk)
        return package

    def _dispatch_ring_allgather(
        self,
        hidden_chunk,
        weights_chunk,
        ids_chunk,
        output_dtype,
        padded_rows,
    ):
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        prev_rank = (rank - 1 + world_size) % world_size
        next_rank = (rank + 1) % world_size

        current = self._make_dispatch_package(
            rank=rank,
            hidden_chunk=hidden_chunk,
            weights_chunk=weights_chunk,
            ids_chunk=ids_chunk,
            output_dtype=output_dtype,
            padded_rows=padded_rows,
        )
        gathered_packages = [None] * world_size
        gathered_packages[rank] = self._clone_package(current)

        for _ in range(world_size - 1):
            current = self._ring_send_recv_package(
                package=current,
                send_rank=next_rank,
                recv_rank=prev_rank,
            )
            owner = int(current["owner"].item())
            gathered_packages[owner] = self._clone_package(current)

        assert all(pkg is not None for pkg in gathered_packages), (
            "Dispatch ring allgather did not collect all owner packages"
        )
        return gathered_packages

    def _combine_ring_reducescatter(
        self,
        output_chunk,
        gathered_packages,
        local_partial_outputs,
    ):
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        prev_rank = (rank - 1 + world_size) % world_size
        next_rank = (rank + 1) % world_size

        local_pkg = gathered_packages[rank]
        current = {
            "owner": local_pkg["owner"].clone(),
            "valid_rows": local_pkg["valid_rows"].clone(),
            "hidden": local_pkg["hidden"].clone(),
            "topk_weights": local_pkg["topk_weights"].clone(),
            "topk_ids": local_pkg["topk_ids"].clone(),
            "accum": torch.zeros_like(local_partial_outputs[rank]),
        }

        # Reducescatter-style ring: each owner package circles world_size hops,
        # and every rank adds its local contribution for that owner once.
        for _ in range(world_size):
            current = self._ring_send_recv_package(
                package=current,
                send_rank=next_rank,
                recv_rank=prev_rank,
            )
            owner = int(current["owner"].item())
            valid_rows = int(current["valid_rows"].item())
            if valid_rows > 0:
                current["accum"][:valid_rows].add_(
                    local_partial_outputs[owner][:valid_rows]
                )

        owner = int(current["owner"].item())
        assert owner == rank, "Combine ring reducescatter did not return owner package"

        local_valid = int(local_pkg["valid_rows"].item())
        if local_valid > 0:
            output_chunk[:local_valid].copy_(current["accum"][:local_valid])

    def _run_ring_chunk(
        self,
        output_chunk,
        hidden_chunk,
        weights_chunk,
        ids_chunk,
        expert_map,
        padded_rows,
    ):
        world_size = dist.get_world_size(self.process_group)
        hidden_size = hidden_chunk.shape[1]

        gathered_packages = self._dispatch_ring_allgather(
            hidden_chunk=hidden_chunk,
            weights_chunk=weights_chunk,
            ids_chunk=ids_chunk,
            output_dtype=output_chunk.dtype,
            padded_rows=padded_rows,
        )

        local_partial_outputs = [
            torch.zeros(
                (padded_rows, hidden_size),
                dtype=output_chunk.dtype,
                device=output_chunk.device,
            )
            for _ in range(world_size)
        ]

        for owner in range(world_size):
            package = gathered_packages[owner]
            cur_valid = int(package["valid_rows"].item())
            if cur_valid > 0:
                local_out = torch.zeros(
                    (cur_valid, hidden_size),
                    dtype=output_chunk.dtype,
                    device=output_chunk.device,
                )
                self._run_local_chunk(
                    output=local_out,
                    hidden_states=package["hidden"][:cur_valid],
                    topk_weights=package["topk_weights"][:cur_valid],
                    topk_ids=package["topk_ids"][:cur_valid],
                    expert_map=expert_map,
                )
                local_partial_outputs[owner][:cur_valid].copy_(local_out)

        self._combine_ring_reducescatter(
            output_chunk=output_chunk,
            gathered_packages=gathered_packages,
            local_partial_outputs=local_partial_outputs,
        )

    def _apply_kernel(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        num_rows = hidden_states.shape[0]
        if num_rows == 0:
            return

        if self.pipeline_depth <= 1 or num_rows == 1:
            self._run_local_chunk(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
            return

        micro_batch_rows = max(1, math.ceil(num_rows / self.pipeline_depth))
        streams = self._build_streams()
        if not streams:
            self._run_local_chunk(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
            return

        ring_enabled = self._is_ring_enabled()
        if ring_enabled:
            padded_rows = micro_batch_rows

        chunk_id = 0
        for start in range(0, num_rows, micro_batch_rows):
            end = min(start + micro_batch_rows, num_rows)
            output_chunk = output[start:end]
            hidden_chunk = hidden_states[start:end]
            weights_chunk = topk_weights[start:end]
            ids_chunk = topk_ids[start:end]

            stream = streams[chunk_id % len(streams)]
            with self._stream_context(stream):
                if ring_enabled:
                    self._run_ring_chunk(
                        output_chunk=output_chunk,
                        hidden_chunk=hidden_chunk,
                        weights_chunk=weights_chunk,
                        ids_chunk=ids_chunk,
                        expert_map=expert_map,
                        padded_rows=padded_rows,
                    )
                else:
                    self._run_local_chunk(
                        output=output_chunk,
                        hidden_states=hidden_chunk,
                        topk_weights=weights_chunk,
                        topk_ids=ids_chunk,
                        expert_map=expert_map,
                    )
            chunk_id += 1

        self._sync_xpu()


def xpu_fused_moe_pipeline(
    hidden_states,
    w13,
    w13_scales,
    w13_bias,
    w2,
    w2_scales,
    w2_bias,
    topk_weights,
    topk_ids,
    n_experts_per_token,
    activation,
    num_experts,
    ep_rank=0,
    ep_size=1,
    expert_map=None,
    output=None,
    is_fp8=False,
    is_int4=False,
    is_mxfp4=False,
    is_mxfp8=False,
    is_block_fp8=False,
    process_group=None,
    pipeline_depth=None,
    pipeline_streams=None,
    min_compute_rows=None,
):
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, (
            "output shape must be the same as hidden_states shape"
        )

    fused_moe = XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=w13_bias,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=w2_bias,
        n_experts_per_token=n_experts_per_token,
        activation=activation,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        expert_map=expert_map,
        is_fp8=is_fp8,
        is_int4=is_int4,
        is_mxfp4=is_mxfp4,
        is_mxfp8=is_mxfp8,
        is_block_fp8=is_block_fp8,
        process_group=process_group,
        pipeline_depth=pipeline_depth,
        pipeline_streams=pipeline_streams,
        min_compute_rows=min_compute_rows,
    )
    fused_moe.apply(
        output=output,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    return output