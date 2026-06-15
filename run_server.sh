#!/bin/bash
cd /home/ljj/Desktop/dingli/HRM-Text-Inference
export PYTHONPATH=$PWD/mini-sglang/python
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9
exec .venv/bin/python -m minisgl \
  --model-path checkpoints/HRM-Text-1B \
  --attention-backend fi --cache-type naive --dtype bfloat16 \
  --host 127.0.0.1 --port 1919 --max-prefill-length 4096 \
  --cuda-graph-max-bs "${GRAPH_BS:-0}" --num-pages "${NUM_PAGES:-2048}"
