# HRM-Text on mini-sglang

Serves the [**HRM-Text-1B**](checkpoints/HRM-Text-1B) (Hierarchical Reasoning Model) checkpoint
through [mini-sglang](mini-sglang)'s OpenAI-compatible HTTP API, with E2E output parity against
the HuggingFace reference (`hf_inference.py`) and a throughput/stress benchmark.

HRM is a dual-timescale **recurrent** transformer: high-level (`H`) and low-level (`L`) stacks
iterate over the same embeddings for `H_cycles × (L_cycles + 1)` steps. The port handles the
parts that differ from a vanilla decoder:

- **Recurrent KV cache** — `num_hidden_layers` inflates to `num_layers_per_stack × H × (L+1) = 128`
  cache slots (32 physical layers reused), addressed by `cycle_offset + layer_idx`.
- **Fused projections** — `attn.gqkv_proj` → `[gate, q, k, v]`, `mlp.gate_up_proj` → `[gate, up]`.
- **PrefixLM mask** — the whole prompt is one bidirectional block (non-causal prefill, causal decode).
- Parameterless float32 RMSNorm, embedding scaling, sigmoid-gated attention.

Implementation lives in `mini-sglang/python/minisgl/models/hrm_text.py` (+ small changes to the
config, weight loader, attention backends, and config loader). See
[`instructions.md`](instructions.md) for the task spec.

## Correctness (parity vs. `hf_inference.py`)

- `parity_check.py` — pure-torch reimplementation vs. HuggingFace: **100% teacher-forced
  next-token argmax agreement** (bf16), across both the bidirectional-prefill and mixed
  prefix/causal paths.
- Live server vs. HF bf16 reference (greedy): identical for the first **33 tokens**, diverging
  only at a genuine near-tie (`":\n\n"` vs `":\n"`) — expected bf16 numerical difference between
  flashinfer and HF attention kernels — then producing a correct answer
  (`… 9.8 is bigger than 9.11. Final answer: $\boxed{9.8}$`).

```bash
# pure-torch vs HF (no server needed)
python parity_check.py
# server vs saved HF bf16 reference (server must be running)
python validate_parity.py
```

## Benchmark (压测)

Streaming benchmark reporting the metrics that matter for serving — **TTFT** (time to first
token / prefill latency), **TPOT** (time per output token / inter-token latency), end-to-end
latency, and throughput — with p99 percentiles.

```bash
python benchmark.py --concurrencies 1 8 32 64 --requests-per-conc 32 --max-tokens 64 --markdown
```

**Setup:** 1× NVIDIA RTX 4090 (24 GB, sm_89) · HRM-Text-1B · bfloat16 · flashinfer attention
backend · CUDA graphs (bs ≤ 64) · 12 288-token KV cache · 64 output tokens/request · greedy.

| Concurrency | Requests | Failures | TTFT avg | TTFT p99 | TPOT avg | TPOT p99 | E2E avg | Output tok/s | Req/s |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 32  | 0 | 32.7 ms | 34.0 ms  | 13.6 ms | 13.7 ms | 0.88 s | 72   | 1.14  |
| 8  | 32  | 0 | 37.0 ms | 37.8 ms  | 16.0 ms | 16.1 ms | 1.03 s | 488  | 7.75  |
| 32 | 128 | 0 | 58.1 ms | 61.2 ms  | 17.5 ms | 17.5 ms | 1.14 s | 1761 | 27.95 |
| 64 | 256 | 0 | 99.8 ms | 111.5 ms | 19.9 ms | 20.0 ms | 1.34 s | 3007 | 47.72 |

Output throughput scales to **~3 000 tok/s** at concurrency 64 while TPOT stays ~14–20 ms and
TTFT degrades gracefully — **0 failures** across all scenarios. (CUDA graphs cut TPOT from
~39 ms to ~14 ms vs. eager decode.)

A correctness/robustness stress test is also provided:

```bash
python stress_test.py --num-requests 300 --concurrency 64 --max-tokens 128
# 300/300 succeeded, no errors/crashes/hangs
```

## Running the server

```bash
# env: flashinfer needs a CUDA 12 toolkit; mini-sglang's torch-fallback kernels need no nvcc
export PYTHONPATH=$PWD/mini-sglang/python
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9

python -m minisgl \
  --model-path checkpoints/HRM-Text-1B \
  --attention-backend fi --cache-type naive --dtype bfloat16 \
  --host 127.0.0.1 --port 1919 --cuda-graph-max-bs 64

# query (raw prompt; the HRM checkpoint has no chat template)
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"hrm",
  "prompt":"<|im_start|><|quad_end|><|object_ref_end|>9.8 and 9.11, which is bigger?<|im_end|>",
  "max_tokens":128, "temperature":0}'
```

`run_server.sh` wraps the launch above (env vars `NUM_PAGES`, `GRAPH_BS` override KV size and
CUDA-graph batch size). `--cache-type naive` is required: radix prefix-sharing is incorrect under
the PrefixLM bidirectional prefill.

### Notes

- `hubert/` in this repo is the unrelated audio-Hubert source (a red herring from the task setup);
  the real model files are in `checkpoints/HRM-Text-1B/`.
- Numbers above are from a single RTX 4090; absolute throughput depends on hardware and KV budget.
