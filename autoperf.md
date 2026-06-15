# autoperf

Autonomous, correctness-gated optimization of the **HRM-Text-1B** model as served by
**mini-sglang**. Same spirit as `autoresearch`, but the metric you maximize is **serving
performance** (throughput / latency), and every change must first clear a **correctness gate**
against the HuggingFace reference before its performance counts.

> The golden rule: **correctness first, performance second.** A faster server that produces the
> wrong tokens is a failed experiment, full stop — you don't even look at its benchmark.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a date-based tag (e.g. `jun15`). The branch `autoperf/<tag>`
   must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoperf/<tag>` from current master.
3. **Read the in-scope files** (the repo is small — read these for full context):
   - `instructions.md` / `README.md` — task spec and the serving setup.
   - `hf_inference.py` — the HuggingFace reference. **Ground truth. Do NOT modify.**
   - `mini-sglang/python/minisgl/models/hrm_text.py` — **the only file you edit.**
   - `parity_check.py` — pure-torch vs HF teacher-forced argmax check (no server).
   - `validate_parity.py` — live server vs saved HF bf16 reference (server must be up).
   - `benchmark.py` — streaming TTFT / TPOT / throughput benchmark (压测).
   - `stress_test.py` — correctness/robustness stress test.
   - `run_server.sh` — launch wrapper.
4. **Verify environment & assets**:
   - Use the env in `.venv` (`.venv/bin/python ...`).
   - `checkpoints/HRM-Text-1B/` exists (the real weights — ignore `hubert/`, it's a red herring).
   - CUDA 12.4 toolkit + flashinfer importable; GPU is free (`nvidia-smi`).
5. **Initialize `results.tsv`**: create it with just the header row (below). Leave it
   **untracked** by git.
6. **Establish the baseline (first run)**: do NOT edit anything yet. Start the server on the
   unmodified model, pass the correctness gate, run the benchmark, and record the baseline row.
   This is the reference point for every later "improved / regressed" decision.
7. **Confirm and go.** Then start the loop and don't stop.

## The two gates

Every experiment passes through two gates, **in order**:

### Gate 1 — Correctness (hard pass/fail; evaluated first)

The edited `hrm_text.py` *is* the served model, so the check that matters is the one that
exercises the **real server path**:

- `validate_parity.py` (server vs HF bf16 reference, greedy): must reproduce baseline behavior —
  **same correct final answer**, and **no earlier divergence** than the baseline's benign
  near-tie. Diverging earlier, a wrong answer, or NaNs → **correctness regression**.
- `parity_check.py` (teacher-forced argmax vs HF): the deterministic, numerically-robust check.
  Target stays **100% next-token argmax agreement**. Use it as the strong supplement whenever the
  change touches logic the pure-torch path mirrors.

**bf16 tolerance**: tiny numerical noise that flips a *genuine* near-tie (logits within ~1e-2,
like the baseline's `":\n\n"` vs `":\n"` at token 33) is acceptable. Anything that drops
teacher-forced argmax below 100%, diverges *earlier* than baseline, or changes the final answer
is **not** — discard it no matter how fast it is.

If Gate 1 fails: **kill the server, log `discard` (reason: correctness), `git reset --hard`, move
on.** Do not benchmark a wrong model.

> Cautionary example: turning on radix prefix-sharing (`--cache-type radix`) speeds up prefill but
> is **incorrect** under HRM's PrefixLM bidirectional prefill — exactly the kind of "optimization"
> Gate 1 exists to catch. `--cache-type naive` is mandatory.

### Gate 2 — Performance (the thing you maximize; only if Gate 1 passes)

Run `benchmark.py`. **Headline metric = output tok/s at concurrency 64** (the 压测 peak — serving
capacity). Secondary metrics that should not meaningfully regress: **TPOT** (per-token latency,
esp. at concurrency 1) and **TTFT** (prefill latency). VRAM is a soft constraint.

- **Improved** = headline tok/s up, no bad TPOT/TTFT regression, correctness intact → **keep / advance**.
- **Equal or worse** = `git reset --hard` back to where you started.

## What you CAN / CANNOT do

**CAN** (all inside `hrm_text.py`):
- Fuse kernels with **Triton** — parameterless float32 RMSNorm, SwiGLU gate-up, RoPE,
  sigmoid-gated attention output, etc.
- **`torch.compile`** the forward or hot submodules.
- Hand-written / **custom ops**, better memory layout, fewer copies, contiguous tensors.
- Cut **Python dispatch overhead** in the recurrent H/L loop — the stacks iterate
  `H_cycles × (L_cycles + 1)` times reusing 32 physical layers (128 effective layer-steps), so
  per-step host overhead is a prime suspect.

**CANNOT**:
- Modify `hf_inference.py` (reference truth) or the harness (`parity_check.py`,
  `validate_parity.py`, `benchmark.py`, `stress_test.py`).
- Edit anything outside `hrm_text.py` — per the task, it's the only file to change.
- Install new packages / add dependencies — use what's in `.venv`.
- Change dtype, model, or eval methodology to "win" correctness or the benchmark.
- Enable radix caching or otherwise trade correctness for speed.

**Simplicity criterion** (same as autoresearch): all else equal, simpler `hrm_text.py` wins. A
tiny throughput gain that adds a gnarly hand-rolled kernel may not beat a clean `torch.compile`
that captures most of it. Deleting code for equal-or-better perf is a great outcome. Weigh
complexity cost against the speedup magnitude.

## Profile to steer (directed search)

Don't guess blindly — **profile, then optimize the actual bottleneck.** Between ideas, capture a
trace and read it:

- `torch.profiler` with a Chrome/Perfetto trace export.
- Look for: GPU bubbles, kernel-launch overhead stacking up across the 128 recurrent layer-steps,
  memory-bound elementwise ops (norm / activation / gating) that want fusing, and whether decode
  is host-bound (→ CUDA graphs / compile) or kernel-bound (→ fused Triton).
- Let the trace pick the next hypothesis. This is what makes the loop a search and not a lottery.

## Output / metrics

`benchmark.py --markdown` prints a per-concurrency table with **TTFT (avg/p99)**, **TPOT
(avg/p99)**, **E2E**, **output tok/s**, **req/s**, and **failures**. Pull the headline:

- output tok/s @ concurrency 64 → primary
- TPOT avg @ concurrency 1 → single-stream latency
- TTFT avg @ concurrency 1 → prefill latency
- peak VRAM from `nvidia-smi` / the server log → memory_gb

Correctness comes from the Gate-1 scripts (argmax % / pass-fail).

## Logging results (`results.tsv`)

Tab-separated (NOT commas — commas break the description). Header + 8 columns:

```
commit	parity	tok_s	tpot_ms	ttft_ms	memory_gb	status	description
```

1. git commit (short, 7 chars)
2. parity: `100` (teacher-forced argmax %), or `fail` if Gate 1 broke, `0` for a crash
3. tok_s: headline output throughput @ conc 64 (`0.0` for crash/fail)
4. tpot_ms: TPOT avg @ conc 1, `.1f`
5. ttft_ms: TTFT avg @ conc 1, `.1f`
6. memory_gb: peak VRAM, `.1f`
7. status: `keep`, `discard`, or `crash`
8. short description of what this experiment tried

Example:

```
commit	parity	tok_s	tpot_ms	ttft_ms	memory_gb	status	description
a1b2c3d	100	3007.0	13.6	32.7	21.4	keep	baseline fi backend + cuda graphs bs<=64
b2c3d4e	100	3180.5	12.9	32.5	21.4	keep	fused parameterless rmsnorm in triton
c3d4e5f	100	3120.0	13.1	33.0	21.6	discard	torch.compile decode block slower than triton rmsnorm
d4e5f6g	fail	0.0	0.0	0.0	0.0	discard	enabled radix cache breaks prefixlm prefill parity
e5f6g7h	0	0.0	0.0	0.0	0.0	crash	fused gated-attn triton kernel OOM during graph capture
```

(Do **not** commit `results.tsv`.)

## The experiment loop

The loop runs on the dedicated branch (e.g. `autoperf/jun15`). Server lifecycle is part of the
loop: each change rebuilds CUDA graphs, so **restart the server fresh every iteration** — there is
no hot reload.

**LOOP FOREVER:**

1. Check git state (current branch/commit).
2. Edit `hrm_text.py` with one experimental idea — ideally the one the last profile pointed at.
3. `git commit`.
4. **Start the server** (background), wait until ready (poll the port / `curl` the endpoint until
   it answers; the first request triggers CUDA-graph capture).
5. **GATE 1 — correctness.** Run `validate_parity.py` (+ `parity_check.py`). If it fails (wrong
   answer / earlier divergence / argmax < 100% beyond a benign near-tie / NaN):
   → kill server, log `discard` (correctness), `git reset --hard`, go to 1. **Do not benchmark.**
6. **GATE 2 — performance.** Run `benchmark.py --concurrencies 1 8 32 64 ... --markdown`.
   Optionally run `stress_test.py` for a robustness pass (must finish with 0 failures).
7. **Profile** (when useful) to inform the next idea.
8. Read out metrics (headline tok/s, TPOT, TTFT, VRAM).
9. Log the row to `results.tsv` (leave it untracked).
10. **Decide:** if performance improved AND correctness held → keep the commit (advance the
    branch). Else → `git reset --hard` back to where this iteration started.
11. **Kill the server** before the next iteration (free the GPU; avoid port collisions).

You're an autonomous performance engineer: profile → hypothesize → implement → gate on
correctness → benchmark → keep or revert → repeat.

### Server lifecycle (sketch)

```bash
export PYTHONPATH=$PWD/mini-sglang/python
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9

# start (background); run_server.sh wraps this. NUM_PAGES / GRAPH_BS override KV size + graph bs.
.venv/bin/python -m minisgl \
  --model-path checkpoints/HRM-Text-1B \
  --attention-backend fi --cache-type naive --dtype bfloat16 \
  --host 127.0.0.1 --port 1919 --cuda-graph-max-bs 64 > server.log 2>&1 &
SERVER_PID=$!

# wait until ready, then run Gate 1 + Gate 2, then:
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
```

`--cache-type naive` is **required** (radix is incorrect for PrefixLM prefill). Keep
`--attention-backend fi` and `--dtype bfloat16` fixed so benchmarks stay comparable across runs.

### Timeouts

A full iteration (server start + 2 gates) should take on the order of a few minutes. If the server
hangs on startup/capture, or a benchmark stalls well past its normal time, **kill it, treat the
iteration as a failure (discard + revert), and move on.** Never let one stuck run block the loop.

### Crashes

If a run crashes (OOM during graph capture, a kernel bug, an import error):
- Dumb + easy (typo, missing import, slightly-too-big graph bs) → fix and re-run.
- Fundamentally broken idea (kernel can't be made correct; OOM that needs the whole approach
  rethought) → log `crash`, `git reset --hard`, move on. Read `server.log` / the stack trace
  before deciding.

### NEVER STOP

Once the loop has begun, **do not pause to ask whether to continue.** Don't ask "should I keep
going?" or "is this a good stopping point?" — the human may be asleep and expects you to keep
working **indefinitely** until manually stopped. If you run out of ideas, think harder: re-read the
profile, re-read `hrm_text.py` and the HRM math (recurrent H/L cycles, fused gqkv / gate-up
projections, sigmoid-gated attention, embedding scaling, float32 RMSNorm), revisit near-misses and
combine them, or try a more radical fusion. The loop runs until the human interrupts you, period.

## Seed ideas backlog

Concrete, HRM-specific hypotheses to pull from (profile-permitting). Always: implement in
`hrm_text.py` → Gate 1 → Gate 2 → keep or revert.

- Fused **parameterless float32 RMSNorm** Triton kernel (cast + mean-square + rsqrt + scale in one pass).
- Fused **SwiGLU** over the `gate_up_proj` output (`silu(gate) * up`) in a single kernel.
- Fused **RoPE** application on q/k.
- Fused **sigmoid-gated attention output** (`sigmoid(g) * attn`) to kill an elementwise pass.
- `torch.compile` (`mode="max-autotune"` / `"reduce-overhead"`) on the per-layer-step block;
  compare against hand-fused Triton and against CUDA-graph-only.
- Trim **Python/host overhead** in the recurrent loop — precompute `cycle_offset + layer_idx`
  indexing, avoid per-step allocations, reuse buffers across the 128 layer-steps.
- Ensure **contiguous / fused** `gqkv_proj` and `gate_up_proj` slicing avoids hidden copies.
- Revisit **CUDA-graph coverage** for batch sizes the benchmark hits but capture currently misses.