#!/bin/bash
source /root/miniforge3/etc/profile.d/conda.sh && conda activate cherry-symm
cd /root/cherry/ccl-graph/megamoe/vllm-xpu-kernels
FI_PROVIDER=shm PYTHONPATH=. mpirun -n 4 /root/miniforge3/envs/hanchao/bin/python tests/fused_moe/test_fused_moe_stream_pipeline_perf.py
