# SPDX-License-Identifier: Apache-2.0
import math
import os
import warnings

import torch
import torch.distributed as dist

from .fused_moe_interface import (
    _get_recipe,
    _should_use_ref_fused_moe,
    implement_zp,
    ref_fused_moe,
)

try:
    import torch.distributed._symmetric_memory as symm_mem
    _SYMM_MEM_AVAILABLE = True
except ImportError:
    _SYMM_MEM_AVAILABLE = False

PIPELINE_DEPTH_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_DEPTH"
PIPELINE_STREAMS_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_STREAMS"
PIPELINE_MIN_ROWS_ENV = "VLLM_XPU_FUSED_MOE_PIPELINE_MIN_ROWS"
EXPERT_THRESHOLD_ENV = "VLLM_XPU_FUSED_MOE_EXPERT_THRESHOLD"


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


class XpuFusedMoe:
    """Fused-MoE with stream pipeline over token micro-batches.

    Pipeline design (from megaMOE):
      p0(stream0): dispatch + (gemm + act + gemm) + combine
      p1(stream1):            dispatch + (gemm + act + gemm) + combine
      ...
    Two streams alternate over token chunks. Each chunk executes
    dispatch + (gemm + act + gemm) + combine as one pipeline stage.

    Ring communication uses symmetric memory with pull-based reads
    instead of P2P send/recv operations.
    """

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
        expert_token_threshold=None,
    ):
        # ---- weight / quantisation setup (from base XpuFusedMoe) ----
        if not is_int4 and not is_mxfp4:
            self.inter_size = w13.shape[-1] // 2
        else:
            self.inter_size = w13.shape[-2] // 2

        assert w13.is_contiguous() and w2.is_contiguous()

        if is_int4 and not hasattr(w13, "xpu_fused_moe"):
            w13_tmp = torch.empty_like(w13)
            w2_tmp = torch.empty_like(w2)
            for i in range(num_experts):
                w13_tmp[i] = implement_zp(w13[i])
                w2_tmp[i] = implement_zp(w2[i])
            w13_tmp = w13_tmp.contiguous()
            w2_tmp = w2_tmp.contiguous()
            w13.data = w13_tmp
            w2.data = w2_tmp
            w13.xpu_fused_moe = True

        self.w13 = w13
        self.w2 = w2

        if not is_fp8 and not is_int4 and not is_mxfp4 and not is_block_fp8:
            self.gemm1_scales = None
            self.gemm2_scales = None
        else:
            self.gemm1_scales = w13_scales
            self.gemm2_scales = w2_scales

        self.w13_bias = w13_bias
        self.w2_bias = w2_bias

        self.n_experts_per_token = n_experts_per_token
        self.activation = activation
        self.inter_size_scale = 2 if self.activation == "relu2_no_mul" else 1
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.is_fp8 = is_fp8
        self.is_int4 = is_int4
        self.is_mxfp4 = is_mxfp4
        self.is_mxfp8 = is_mxfp8
        self.is_block_fp8 = is_block_fp8
        self.recipe = _get_recipe(is_fp8, is_mxfp8, is_mxfp4, is_int4,
                                   is_block_fp8)
        self._use_ref = _should_use_ref_fused_moe(is_mxfp8)

        if self.activation == "silu":
            self.act_func = torch.ops._C.silu_and_mul
        elif self.activation == "gelu":
            self.act_func = torch.ops._C.gelu_and_mul
        elif self.activation == "gelu_tanh":
            self.act_func = torch.ops._C.gelu_tanh_and_mul
        elif self.activation == "swigluoai" \
                or ("SWIGLUOAI" in str(self.activation)):
            self.act_func = torch.ops._C.swigluoai_and_mul
        elif self.activation == "relu2_no_mul":
            self.act_func = torch.ops._C.relu2_no_mul
        elif self.activation == "swiglustep":
            self.act_func = torch.ops._C.swiglustep_and_mul
        else:
            raise ValueError(
                f"Unsupported FusedMoe activation: {self.activation}.")

        self.expert_map = expert_map
        if self.expert_map is None and self.ep_size > 1:
            self.expert_map = torch.empty(
                (self.num_experts * self.ep_size),
                dtype=torch.int32,
                device=w13.device,
            )
            torch.ops._moe_C.init_expert_map(
                self.expert_map,
                self.num_experts,
                self.ep_rank,
                self.ep_size,
            )

        if self.expert_map is not None:
            self.total_experts_num = self.expert_map.shape[0]
        else:
            self.total_experts_num = self.num_experts * self.ep_size
        self.local_experts_num = self.num_experts

        # ---- pipeline-specific setup ----
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
        self.expert_token_threshold = max(
            0,
            expert_token_threshold
            if expert_token_threshold is not None
            else _read_int_env(EXPERT_THRESHOLD_ENV, 128),
        )

        self._streams = []
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and hasattr(xpu, "Stream"):
            self._streams = [xpu.Stream() for _ in range(self.pipeline_streams)]

        # ---- symmetric memory buffer cache ----
        self._symm_cache_key = None
        self._symm_handle = None

        # ---- allgather + reduce-scatter handle cache ----
        self._ag_cache_key = None
        self._ag_handles = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        if self._use_ref:
            self._apply_ref(output, hidden_states,
                            topk_weights, topk_ids, expert_map)
        else:
            self._apply_kernel(output, hidden_states,
                               topk_weights, topk_ids, expert_map)

    # ------------------------------------------------------------------
    # Reference path
    # ------------------------------------------------------------------

    def _apply_ref(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        out = ref_fused_moe(
            recipe=self.recipe,
            x=hidden_states,
            w13=self.w13,
            w13_scales=self.gemm1_scales,
            w13_bias=self.w13_bias,
            w2=self.w2,
            w2_scales=self.gemm2_scales,
            w2_bias=self.w2_bias,
            expert_weights=topk_weights,
            expert_indices=topk_ids,
            num_per_tok=self.n_experts_per_token,
            activation=self.activation,
            num_experts=self.num_experts,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
        )
        output.copy_(out)

    # ------------------------------------------------------------------
    # Kernel path (grouped GEMM)
    # ------------------------------------------------------------------

    def _apply_kernel_impl(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map=None,
    ):
        """Core fused-MoE kernel: remap → gemm1 → act → gemm2 → gather."""
        num_rows, hidden_size = hidden_states.shape
        num_moe_inputs = self.n_experts_per_token * num_rows

        if expert_map is None and self.ep_size > 1:
            expert_map = self.expert_map

        remapped_hidden_states = torch.empty(
            (num_rows * self.n_experts_per_token, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows_per_expert = torch.zeros(
            (self.num_experts,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        unpermuted_row_to_permuted_row = torch.empty(
            (num_rows, self.n_experts_per_token),
            dtype=torch.int32,
            device=hidden_states.device,
        )

        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=self.total_experts_num,
            local_experts_num=self.local_experts_num,
        )

        # Valid rows after EP filtering (only local experts have data)
        valid_rows = rows_per_expert.sum().item()

        # gemm1
        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * self.inter_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_B=self.w13,
            ptr_scales=self.gemm1_scales,
            ptr_bias=self.w13_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * self.inter_size,
            K=hidden_size,
            num_experts=self.num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4,
        )

        # activation — only on valid rows (GEMM only wrote to [0, valid_rows))
        act_output = torch.empty(
            (num_moe_inputs, self.inter_size * self.inter_size_scale),
            dtype=gemm1_output.dtype,
            device=gemm1_output.device,
        )
        self.act_func(act_output[:valid_rows], gemm1_output[:valid_rows])

        # gemm2
        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_B=self.w2,
            ptr_scales=self.gemm2_scales,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=self.inter_size * self.inter_size_scale,
            num_experts=self.num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4,
        )

        torch.ops._moe_C.moe_gather(
            output, gemm2_output, topk_weights,
            unpermuted_row_to_permuted_row, self.num_experts,
        )

    # ------------------------------------------------------------------
    # Pipeline dispatch
    # ------------------------------------------------------------------

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

        if self._is_ring_enabled():
            self._run_ring_pipeline(
                output, hidden_states, topk_weights, topk_ids, expert_map,
            )
            return

        self._run_local_pipeline(
            output, hidden_states, topk_weights, topk_ids, expert_map,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_ring_enabled(self) -> bool:
        if self.ep_size <= 1:
            return False
        if not _is_dist_ready():
            return False
        world_size = dist.get_world_size(self.process_group)
        if world_size != self.ep_size:
            warnings.warn(
                f"Distributed world size does not match ep_size "
                f"(world_size={world_size}, ep_size={self.ep_size}). "
                "Fallback to local non-ring execution.",
                stacklevel=2,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Local token-microbatch pipeline
    # ------------------------------------------------------------------

    def _run_local_pipeline(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map,
    ):
        num_rows = hidden_states.shape[0]
        if num_rows == 0:
            return

        if not self._streams:
            self._apply_kernel_impl(
                output, hidden_states, topk_weights, topk_ids, expert_map,
            )
            return

        micro_batch_rows = max(1, math.ceil(num_rows / self.pipeline_depth))

        chunk_id = 0
        for start in range(0, num_rows, micro_batch_rows):
            end = min(start + micro_batch_rows, num_rows)
            stream = self._streams[chunk_id % len(self._streams)]

            with torch.xpu.stream(stream):
                self._run_local_chunk(
                    output=output[start:end],
                    hidden_states=hidden_states[start:end],
                    topk_weights=topk_weights[start:end],
                    topk_ids=topk_ids[start:end],
                    expert_map=expert_map,
                )
            chunk_id += 1

        torch.xpu.synchronize()

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
            self._apply_kernel_impl(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
            )

    # ------------------------------------------------------------------
    # Allgather + reduce-scatter handle management
    # ------------------------------------------------------------------

    def _get_allgather_handles(self, num_rows, hidden_size, topk,
                               hidden_dtype, weights_dtype, ids_dtype,
                               output_dtype, device, world_size):
        """Allocate symmetric memory handles for allgather + reduce-scatter.

        Each rank gets:
          - hidden:      [num_rows, hidden_size]  for allgather input
          - topk_weights:[num_rows, topk]         for allgather input
          - topk_ids:    [num_rows, topk]          for allgather input
          - partial_out: [world_size * num_rows, hidden_size] for accumulation
                         (world_size slots, one per contributing rank)
        """
        cache_key = (num_rows, hidden_size, topk,
                     hidden_dtype, weights_dtype, ids_dtype,
                     output_dtype, device, world_size)
        if (self._ag_cache_key == cache_key
                and self._ag_handles is not None):
            return self._ag_handles

        group = self.process_group or dist.group.WORLD

        def _alloc(shape, dtype):
            t = symm_mem.empty(*shape, dtype=dtype, device=device)
            return symm_mem.rendezvous(t, group)

        handles = {
            "hidden": _alloc((num_rows, hidden_size), hidden_dtype),
            "topk_weights": _alloc((num_rows, topk), weights_dtype),
            "topk_ids": _alloc((num_rows, topk), ids_dtype),
            "partial_out": _alloc(
                (world_size * num_rows, hidden_size), output_dtype,
            ),
        }
        shapes = {
            "hidden": ((num_rows, hidden_size), hidden_dtype),
            "topk_weights": ((num_rows, topk), weights_dtype),
            "topk_ids": ((num_rows, topk), ids_dtype),
            "partial_out": (
                (world_size * num_rows, hidden_size), output_dtype,
            ),
        }

        self._ag_cache_key = cache_key
        self._ag_handles = (handles, shapes)
        return handles, shapes

    # ------------------------------------------------------------------
    # Ring pipeline – symmetric-memory pull-based exchange
    # ------------------------------------------------------------------

    def _get_symm_handles(self, padded_rows, hidden_size, topk,
                          hidden_dtype, weights_dtype, ids_dtype,
                          output_dtype, device):
        """Lazily allocate / cache separate symmetric-memory buffers.

        Each package field gets its own symmetric buffer and rendezvous
        handle.  The ``meta`` handle is used for barrier synchronisation.
        """
        cache_key = (padded_rows, hidden_size, topk,
                     hidden_dtype, weights_dtype, ids_dtype,
                     output_dtype, device)

        if self._symm_cache_key == cache_key and self._symm_handle is not None:
            return self._symm_handle

        group = self.process_group or dist.group.WORLD

        def _alloc(shape, dtype):
            t = symm_mem.empty(*shape, dtype=dtype, device=device)
            return symm_mem.rendezvous(t, group)

        handles = {
            "meta": _alloc((3,), torch.int32),
            "hidden": _alloc((padded_rows, hidden_size), hidden_dtype),
            "topk_weights": _alloc((padded_rows, topk), weights_dtype),
            "topk_ids": _alloc((padded_rows, topk), ids_dtype),
            "accum": _alloc((padded_rows, hidden_size), output_dtype),
        }
        shapes = {
            "meta": ((3,), torch.int32),
            "hidden": ((padded_rows, hidden_size), hidden_dtype),
            "topk_weights": ((padded_rows, topk), weights_dtype),
            "topk_ids": ((padded_rows, topk), ids_dtype),
            "accum": ((padded_rows, hidden_size), output_dtype),
        }

        self._symm_cache_key = cache_key
        self._symm_handle = (handles, shapes)
        return self._symm_handle

    def _write_package_to_symm(self, handles, shapes, rank, package):
        """Write *package* dict into the local symmetric-memory slots."""
        meta = torch.tensor(
            [package["owner"].item(),
             package["hop"].item(),
             package["valid_rows"].item()],
            dtype=torch.int32,
            device=package["owner"].device,
        )
        handles["meta"].get_buffer(rank, *shapes["meta"]).copy_(meta)
        handles["hidden"].get_buffer(
            rank, *shapes["hidden"]).copy_(package["hidden"])
        handles["topk_weights"].get_buffer(
            rank, *shapes["topk_weights"]).copy_(package["topk_weights"])
        handles["topk_ids"].get_buffer(
            rank, *shapes["topk_ids"]).copy_(package["topk_ids"])
        handles["accum"].get_buffer(
            rank, *shapes["accum"]).copy_(package["accum"])

    def _pull_package_from_rank(self, handles, shapes, src_rank):
        """Pull (remote-read) a package from *src_rank*'s symmetric slots."""
        meta = handles["meta"].get_buffer(
            src_rank, *shapes["meta"]).clone()
        return {
            "owner": meta[0:1].clone(),
            "hop": meta[1:2].clone(),
            "valid_rows": meta[2:3].clone(),
            "hidden": handles["hidden"].get_buffer(
                src_rank, *shapes["hidden"]).clone(),
            "topk_weights": handles["topk_weights"].get_buffer(
                src_rank, *shapes["topk_weights"]).clone(),
            "topk_ids": handles["topk_ids"].get_buffer(
                src_rank, *shapes["topk_ids"]).clone(),
            "accum": handles["accum"].get_buffer(
                src_rank, *shapes["accum"]).clone(),
        }

    def _ring_exchange_package(self, handles, shapes, rank, recv_rank,
                               package):
        """Write local package, barrier, pull from *recv_rank*, barrier."""
        self._write_package_to_symm(handles, shapes, rank, package)
        # Fence: ensure all ranks have finished writing before any reads.
        handles["meta"].barrier(channel=0)
        recv_package = self._pull_package_from_rank(
            handles, shapes, recv_rank,
        )
        # Fence: ensure all ranks have finished reading before next write.
        handles["meta"].barrier(channel=1)
        return recv_package

    def _make_dispatch_package(
        self, rank, hidden_chunk, weights_chunk, ids_chunk, output_dtype,
        padded_rows,
    ):
        valid_rows = hidden_chunk.shape[0]
        hidden_size = hidden_chunk.shape[1]
        topk = ids_chunk.shape[1]

        package = {
            "owner": torch.tensor(
                [rank], dtype=torch.int32, device=hidden_chunk.device,
            ),
            "hop": torch.zeros(
                [1], dtype=torch.int32, device=hidden_chunk.device,
            ),
            "valid_rows": torch.tensor(
                [valid_rows], dtype=torch.int32, device=hidden_chunk.device,
            ),
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

    def _run_ring_pipeline(
        self, output, hidden_states, topk_weights, topk_ids, expert_map,
    ):
        """Allgather + local compute + reduce-scatter pipeline.

        Only 2 barriers for the entire operation (pull-based design):
          Barrier 1: after all ranks write input → before any rank pulls
          Barrier 2: after all ranks push partial outputs → before sum

        Pull operations use the copy engine (DMA), overlapping with
        compute on a separate stream.

        Expert threshold filtering:
          After barrier 1, read all ranks' topk_ids to count per-expert
          tokens globally.  Experts with count >= threshold are "big"
          (computed per source rank), those below are "small" (deferred
          to a single combined batch at the end).  Both partials are
          accumulated locally per source rank, then pushed once before
          barrier 2.
        """
        num_rows = hidden_states.shape[0]
        if not self._streams:
            self._apply_kernel_impl(
                output, hidden_states, topk_weights, topk_ids, expert_map,
            )
            return

        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)
        hidden_size = hidden_states.shape[1]
        topk = topk_ids.shape[1]
        device = hidden_states.device

        handles, shapes = self._get_allgather_handles(
            num_rows, hidden_size, topk,
            hidden_states.dtype, topk_weights.dtype, topk_ids.dtype,
            output.dtype, device, world_size,
        )

        nr = num_rows  # shorthand for slicing partial_out

        # === Phase 1: Write local input to symmetric memory ===
        handles["hidden"].get_buffer(
            rank, *shapes["hidden"],
        ).copy_(hidden_states)
        handles["topk_ids"].get_buffer(
            rank, *shapes["topk_ids"],
        ).copy_(topk_ids)
        handles["topk_weights"].get_buffer(
            rank, *shapes["topk_weights"],
        ).copy_(topk_weights)

        # Barrier 1: all ranks' input data is visible for pulling
        handles["hidden"].barrier(channel=0)

        # === Phase 2: Pull + compute with per-expert accumulation ===
        if expert_map is None and self.ep_size > 1:
            expert_map = self.expert_map

        threshold = self.expert_token_threshold
        copy_stream = self._streams[0]
        remote_ranks = [
            (rank + i) % world_size for i in range(1, world_size)
        ]

        # Local accumulators per source rank
        local_accum = {}
        for r in range(world_size):
            local_accum[r] = torch.zeros(
                nr, hidden_size, dtype=output.dtype, device=device,
            )

        if threshold > 0 and expert_map is not None:
            # --- Batch steps to reach threshold tokens per expert ---
            # Compute how many ring steps to accumulate so each local
            # expert reaches ~threshold tokens before computing.
            tokens_per_expert_per_step = max(
                1,
                (num_rows * topk) // self.total_experts_num,
            )
            steps_per_batch = max(
                1,
                (threshold + tokens_per_expert_per_step - 1)
                // tokens_per_expert_per_step,
            )

            # Gather all step data: own + remote ranks
            all_steps = [(rank, hidden_states, topk_weights, topk_ids)]
            for src_rank in remote_ranks:
                src_h = handles["hidden"].get_buffer(
                    src_rank, *shapes["hidden"],
                ).clone()
                src_w = handles["topk_weights"].get_buffer(
                    src_rank, *shapes["topk_weights"],
                ).clone()
                src_i = handles["topk_ids"].get_buffer(
                    src_rank, *shapes["topk_ids"],
                ).clone()
                all_steps.append((src_rank, src_h, src_w, src_i))

            # Process in batches
            for batch_start in range(0, len(all_steps), steps_per_batch):
                batch = all_steps[batch_start:batch_start + steps_per_batch]

                combined_h = torch.cat([d[1] for d in batch], dim=0)
                combined_w = torch.cat([d[2] for d in batch], dim=0)
                combined_i = torch.cat([d[3] for d in batch], dim=0)

                combined_partial = torch.zeros(
                    combined_h.shape[0], hidden_size,
                    dtype=output.dtype, device=device,
                )
                self._apply_kernel_impl(
                    combined_partial, combined_h, combined_w,
                    combined_i, expert_map,
                )

                # Split back per source rank
                offset = 0
                for d_rank, _, _, _ in batch:
                    local_accum[d_rank].add_(
                        combined_partial[offset:offset + nr],
                    )
                    offset += nr

        else:
            # --- No threshold: compute per-step (original behavior) ---
            # Prefetch first remote rank's data
            prefetch = None
            if remote_ranks:
                with torch.xpu.stream(copy_stream):
                    r0 = remote_ranks[0]
                    prefetch = (
                        handles["hidden"].get_buffer(
                            r0, *shapes["hidden"],
                        ).clone(),
                        handles["topk_weights"].get_buffer(
                            r0, *shapes["topk_weights"],
                        ).clone(),
                        handles["topk_ids"].get_buffer(
                            r0, *shapes["topk_ids"],
                        ).clone(),
                    )

            # Own contribution (overlaps with first prefetch)
            own_partial = torch.zeros(
                nr, hidden_size, dtype=output.dtype, device=device,
            )
            self._apply_kernel_impl(
                own_partial, hidden_states, topk_weights, topk_ids,
                expert_map,
            )
            local_accum[rank].add_(own_partial)

            # Remote contributions with pipeline overlap
            for i, src_rank in enumerate(remote_ranks):
                torch.xpu.current_stream().wait_stream(copy_stream)
                src_h, src_w, src_i = prefetch

                if i + 1 < len(remote_ranks):
                    next_r = remote_ranks[i + 1]
                    with torch.xpu.stream(copy_stream):
                        prefetch = (
                            handles["hidden"].get_buffer(
                                next_r, *shapes["hidden"],
                            ).clone(),
                            handles["topk_weights"].get_buffer(
                                next_r, *shapes["topk_weights"],
                            ).clone(),
                            handles["topk_ids"].get_buffer(
                                next_r, *shapes["topk_ids"],
                            ).clone(),
                        )

                partial = torch.zeros(
                    nr, hidden_size, dtype=output.dtype, device=device,
                )
                self._apply_kernel_impl(
                    partial, src_h, src_w, src_i, expert_map,
                )
                local_accum[src_rank].add_(partial)

        # === Push all accumulators to remote ===
        for src_rank in range(world_size):
            handles["partial_out"].get_buffer(
                src_rank, *shapes["partial_out"],
            )[rank * nr:(rank + 1) * nr].copy_(local_accum[src_rank])

        # Barrier 2: all partial outputs are visible for reading
        handles["hidden"].barrier(channel=1)

        # === Phase 3: Sum partial outputs for own tokens ===
        own_accum = handles["partial_out"].get_buffer(
            rank, *shapes["partial_out"],
        )
        # Sum over world_size slots → [num_rows, hidden_size]
        output.zero_()
        for r in range(world_size):
            output.add_(own_accum[r * nr:(r + 1) * nr])

        torch.xpu.synchronize()


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
    expert_token_threshold=None,
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
        expert_token_threshold=expert_token_threshold,
    )
    fused_moe.apply(
        output=output,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    return output