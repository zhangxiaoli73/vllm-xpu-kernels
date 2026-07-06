# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# isort: off
import argparse
import gc
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
# isort: on


def _parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower().strip()
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}. Use bf16 or fp16.")


def _clear_xpu_cache() -> None:
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def _build_inputs(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    has_bias: bool,
    device: str,
):
    hidden_states = torch.randn((m, k), device=device, dtype=dtype) / 16

    # In test_fused_moe.py, w13/w2 are generated as [E, 2N, K] / [E, K, N],
    # then transposed to kernel-friendly layout [E, K, 2N] / [E, N, K].
    w13 = torch.randn((num_experts, 2 * n, k), device=device, dtype=dtype) / 16
    w2 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 16

    if has_bias:
        w13_bias = torch.randn((num_experts, 2 * n), device=device,
                               dtype=dtype) / 16
        w2_bias = torch.randn((num_experts, k), device=device, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None

    scores = torch.randn((m, num_experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    w13 = w13.transpose(-1, -2).contiguous()
    w2 = w2.transpose(-1, -2).contiguous()

    fused_moe = XpuFusedMoe(
        w13=w13,
        w13_scales=None,
        w13_bias=w13_bias,
        w2=w2,
        w2_scales=None,
        w2_bias=w2_bias,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=num_experts,
        is_fp8=False,
        is_int4=False,
        is_mxfp4=False,
        is_mxfp8=False,
        is_block_fp8=False,
    )

    return fused_moe, hidden_states, topk_weights, topk_ids


def _print_latency(tag: str, latencies_ms: list[float]) -> None:
    avg_ms = sum(latencies_ms) / len(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    print(
        f"[{tag}] avg={avg_ms:.4f} ms, median={median_ms:.4f} ms, "
        f"min={min_ms:.4f} ms, max={max_ms:.4f} ms"
    )
    print(f"[{tag}] detail={latencies_ms}")


def _print_repeat_summary(tag: str, repeat_avgs: list[float]) -> None:
    if not repeat_avgs:
        return
    avg_ms = sum(repeat_avgs) / len(repeat_avgs)
    median_ms = statistics.median(repeat_avgs)
    min_ms = min(repeat_avgs)
    max_ms = max(repeat_avgs)
    print(
        f"[{tag}_repeat_summary] avg={avg_ms:.4f} ms, "
        f"median={median_ms:.4f} ms, min={min_ms:.4f} ms, max={max_ms:.4f} ms"
    )
    print(f"[{tag}_repeat_summary] per_repeat_avg={repeat_avgs}")


def _get_trace_path(base_path: str, repeat_idx: int, repeats: int) -> str:
    path = Path(base_path)
    if repeats <= 1:
        return str(path)
    stem = path.stem
    suffix = path.suffix if path.suffix else ".json"
    return str(path.with_name(f"{stem}_repeat{repeat_idx + 1}{suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark XpuFusedMoe._apply_kernel (16-bit only): "
            "all kernels, pre-gemm2, and gemm2 + moe_gather"
        ))
    parser.add_argument("--m", type=int, default=4096, help="num_rows")
    parser.add_argument("--n", type=int, default=5120, help="intermediate_size")
    parser.add_argument("--k", type=int, default=8192, help="hidden_size")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--dtype",
                        type=str,
                        default="bf16",
                        choices=["bf16", "fp16"],
                        help="Only 16-bit dtypes are supported")
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--loop", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat",
                        type=int,
                        default=3,
                        help="Repeat the same benchmark and summarize")
    parser.add_argument(
        "--enable-profile",
        action="store_true",
        help="Enable torch profiler during benchmark run",
    )
    parser.add_argument(
        "--profile-trace-path",
        type=str,
        default="./profile_kineto_trace_fused_moe.json",
        help="Path to export profiler chrome trace",
    )
    parser.add_argument(
        "--print-profile-table",
        action="store_true",
        help="Print profiler key averages table when profiling is enabled",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.loop <= args.warmup:
        raise ValueError("loop must be greater than warmup")
    if args.repeat <= 0:
        raise ValueError("repeat must be greater than 0")

    dtype = _parse_dtype(args.dtype)
    seed_everything(args.seed)
    torch.xpu.synchronize()

    fused_moe, hidden_states, topk_weights, topk_ids = _build_inputs(
        m=args.m,
        n=args.n,
        k=args.k,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=dtype,
        has_bias=args.has_bias,
        device="xpu",
    )

    print(
        "Running benchmark with "
        f"(m={args.m}, n={args.n}, k={args.k}, "
        f"num_experts={args.num_experts}, topk={args.topk}, "
        f"dtype={dtype}, has_bias={args.has_bias}, "
        f"warmup={args.warmup}, loop={args.loop})"
    )

    all_repeat_avgs = []
    pre_gemm2_repeat_avgs = []
    gemm2_gather_repeat_avgs = []
    output = torch.empty_like(hidden_states)

    for r in range(args.repeat):
        print(f"\n[repeat {r + 1}/{args.repeat}]", flush=True)

        timed = args.loop - args.warmup

        begin_events_all = [
            torch.xpu.Event(enable_timing=True) for _ in range(timed)
        ]
        end_events_all = [
            torch.xpu.Event(enable_timing=True) for _ in range(timed)
        ]

        begin_events_g2g = [
            torch.xpu.Event(enable_timing=True) for _ in range(timed)
        ]
        end_events_g2g = [
            torch.xpu.Event(enable_timing=True) for _ in range(timed)
        ]

        if args.enable_profile:
            prof_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ])
        else:
            prof_ctx = nullcontext()

        with prof_ctx as prof:
            # warmup
            for _ in range(args.warmup):
                fused_moe.apply(
                    output=output,
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
            torch.xpu.synchronize()

            # Single pass: one apply call records all three segments.
            for i in range(args.loop):
                fused_moe.apply(
                    output=output,
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    start_event_all=begin_events_all[i - args.warmup]
                    if i >= args.warmup else None,
                    end_event_all=end_events_all[i - args.warmup]
                    if i >= args.warmup else None,
                    start_event_gemm2_gather=begin_events_g2g[i - args.warmup]
                    if i >= args.warmup else None,
                    end_event_gemm2_gather=end_events_g2g[i - args.warmup]
                    if i >= args.warmup else None,
                )
            torch.xpu.synchronize()

        if args.enable_profile:
            trace_path = _get_trace_path(args.profile_trace_path, r, args.repeat)
            prof.export_chrome_trace(trace_path)
            print(f"[repeat {r + 1}] profiler trace exported: {trace_path}")
            if args.print_profile_table:
                print(
                    prof.key_averages().table(sort_by="self_xpu_time_total")
                )

        lat_all = [
            b.elapsed_time(e) for b, e in zip(begin_events_all, end_events_all)
        ]
        lat_gemm2_gather = [
            b.elapsed_time(e) for b, e in zip(begin_events_g2g, end_events_g2g)
        ]
        lat_pre_gemm2 = [
            max(0.0, all_t - g2g_t)
            for all_t, g2g_t in zip(lat_all, lat_gemm2_gather)
        ]

        _print_latency("apply_kernel_all_kernels", lat_all)
        _print_latency("apply_kernel_pre_gemm2", lat_pre_gemm2)
        _print_latency("apply_kernel_gemm2_plus_gather", lat_gemm2_gather)

        all_repeat_avgs.append(sum(lat_all) / len(lat_all))
        pre_gemm2_repeat_avgs.append(sum(lat_pre_gemm2) / len(lat_pre_gemm2))
        gemm2_gather_repeat_avgs.append(
            sum(lat_gemm2_gather) / len(lat_gemm2_gather))

    print("\n===== Repeat Summary =====")
    _print_repeat_summary("apply_kernel_all_kernels", all_repeat_avgs)
    _print_repeat_summary("apply_kernel_pre_gemm2", pre_gemm2_repeat_avgs)
    _print_repeat_summary("apply_kernel_gemm2_plus_gather",
                          gemm2_gather_repeat_avgs)

    _clear_xpu_cache()


if __name__ == "__main__":
    main()
