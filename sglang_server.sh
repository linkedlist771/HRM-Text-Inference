#!/bin/bash

python -m sglang.launch_server \
  --model-path checkpoints/HRM-Text-1B \
  --model-impl transformers \
  --host 0.0.0.0 --port 30000