# Task: Adapt the HRM Text Model into mini-sglang

## Objective

Integrate the HRM (Hierarchical Reasoning Model) text model into the `mini-sglang`
project so it can be served through mini-sglang's OpenAI-compatible HTTP API.

## Context & Resources

- **`mini-sglang/`** (repo root): the target inference engine. You may modify any
  part of it freely.
- **`hf_inference.py`**: a working HuggingFace reference implementation of the HRM
  text model. Treat its outputs as the ground truth for correctness.
- **`hubert/`**: the model's required files (weights, config, tokenizer, etc.),
  already copied locally from the HuggingFace hub.
- **`.venv`**: holds all environment, use the env from here.

## Acceptance Criteria (both must pass, end-to-end)

1. **Correctness — E2E OpenAI request**
   A standard OpenAI-compatible HTTP request (e.g. `POST /v1/chat/completions`)
   to the running server returns a correct response that matches the reference
   output from `hf_inference.py`.
2. **Performance — load/stress test (压测)**
   A throughput/stress benchmark runs cleanly against the server with no errors,
   crashes, or hangs.

## Notes

- Use `hf_inference.py` as the parity reference when validating correctness.