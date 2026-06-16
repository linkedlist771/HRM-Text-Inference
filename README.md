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
| 1  | 48  | 0 | 27.5 ms | 64.5 ms | 11.6 ms | 11.7 ms | 0.75 s | 84   | 1.34  |
| 8  | 48  | 0 | 29.4 ms | 30.1 ms | 13.3 ms | 13.4 ms | 0.86 s | 588  | 9.33  |
| 32 | 128 | 0 | 48.9 ms | 51.6 ms | 14.5 ms | 14.6 ms | 0.95 s | 2117 | 33.61 |
| 64 | 256 | 0 | 80.0 ms | 94.0 ms | 17.1 ms | 17.3 ms | 1.14 s | 3517 | 55.82 |

Output throughput scales to **~3 500 tok/s** at concurrency 64 while TPOT stays ~12–17 ms and
TTFT degrades gracefully — **0 failures** across all scenarios.

### Speeding up inference (CUDA graphs · profiling · torch.compile)

**CUDA graphs** are the biggest win: the recurrent HRM forward issues ~128 attention + 128 MLP
calls *per token*, so eager decode is dominated by kernel-launch overhead. Capturing decode into
CUDA graphs (`--cuda-graph-max-bs N`, on by default) cut TPOT from ~39 ms to ~14 ms.

**Profiler-driven optimization.** Profiling one decode run with graphs *and* compile disabled
(`profile_decode.py`, so nothing hides the real per-op cost) showed the eager path was
**launch-bound** (CPU 1.94 s vs CUDA 1.07 s) and that the parameterless RMSNorm alone was ~6
eager ops (`pow`/`mean`/`rsqrt`/cast/`mul`) × 16.9 k calls, plus my attention `.contiguous()`
copies cost another ~40 ms / 24.6 k clones. Two fixes — a fused flashinfer `rmsnorm` (all-ones
weight) and dropping the redundant `.contiguous()` — gave:

| | Self CPU | Self CUDA | `copy_` calls |
|---|---|---|---|
| before | 1.94 s | 1.07 s | 67.5 k |
| after  | **0.875 s** (−55 %) | **0.863 s** (−19 %) | **9.2 k** |

This also lifted the CUDA-graph decode path: TPOT 13.6→11.6 ms (c=1), throughput 3007→**3517**
tok/s (c=64) — and parity is unchanged (still 33 matching tokens vs. HF).

**torch.compile** (`--enable-torch-compile`) is wired in and composes with CUDA graphs (the
compiled, fused kernels are captured into each graph). For *this* model it is **slower** and left
off by default — a measured A/B (graph-bs 32) shows why: the forward is dominated by tiny cuBLAS
GEMMs (Inductor can't beat them) and flashinfer custom ops (128 graph breaks), so there is little
left to fuse and the breaks add overhead.

| graph-bs 32 | c=1 TPOT | c=8 TPOT | c=32 TPOT | c=32 tok/s |
|---|---|---|---|---|
| CUDA graphs only        | **11.6 ms** | **13.3 ms** | **14.6 ms** | **2102** |
| + torch.compile         | 19.7 ms | 22.9 ms | 23.3 ms | 1339 |

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
