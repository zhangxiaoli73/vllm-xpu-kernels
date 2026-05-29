#!/bin/bash
cd /root/cherry/megamoe/vllm-xpu-kernels
PYTHONPATH=. python -m pytest tests/fused_moe/test_fused_moe_stream_pipeline_accuracy.py -v -s
PYTHONPATH=. XPU_KERNEL_RUN_PERF_TESTS=1 python -m pytest tests/fused_moe/test_fused_moe_stream_pipeline_perf.py -v -s
