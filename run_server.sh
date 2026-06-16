#!/bin/bash
# Launch HRM-Text-1B on mini-sglang with the best-known serving config.
#
# Why these flags (see README "Speeding up inference"):
#   --attention-backend fi    flashinfer; the only backend that does the non-causal
#                             PrefixLM prefill HRM needs (and JIT-compiles on sm_89).
#   --cache-type naive        REQUIRED: radix prefix-sharing is incorrect under the
#                             PrefixLM bidirectional prefill.
#   --cuda-graph-max-bs 64    CUDA graphs ON — the biggest decode speedup (TPOT ~39 -> ~14 ms).
#   (no --enable-torch-compile) measured ~40-60% SLOWER for this GEMM/flashinfer-bound
#                             model; the win comes from the fused RMSNorm + CUDA graphs.
#   --memory-ratio 0.85       auto-size the KV cache to fill the GPU for max throughput.
#
# Env overrides: HOST, PORT, GRAPH_BS, MEM_RATIO, NUM_PAGES (fixed KV size for a
# shared/small GPU), EXTRA_ARGS (e.g. "--enable-torch-compile" to A/B it).
set -euo pipefail
cd /home/ljj/Desktop/dingli/HRM-Text-Inference

export PYTHONPATH="$PWD/mini-sglang/python"
export CUDA_HOME=/usr/local/cuda-12.4          # flashinfer JIT needs a CUDA 12 toolkit
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9                 # RTX 4090 (sm_89)

ARGS=(
  --model-path checkpoints/HRM-Text-1B
  --attention-backend fi
  --cache-type naive
  --dtype bfloat16
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-1919}"
  --max-prefill-length 4096
  --cuda-graph-max-bs "${GRAPH_BS:-64}"
  --memory-ratio "${MEM_RATIO:-0.85}"
)
# Fixed KV size (overrides --memory-ratio auto-sizing) when set, e.g. NUM_PAGES=4096.
[ -n "${NUM_PAGES:-}" ] && ARGS+=(--num-pages "$NUM_PAGES")

exec .venv/bin/python -m minisgl "${ARGS[@]}" ${EXTRA_ARGS:-}
