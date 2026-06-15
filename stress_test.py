"""Throughput / stress test for the mini-sglang HRM server (压测).

Fires many concurrent OpenAI-compatible requests at the running server and
checks that every one completes with no errors/crashes/hangs, then reports
throughput. Uses the raw `prompt` field (no chat template needed).
"""

import argparse
import json
import random
import threading
import time
import urllib.request

URL = "http://127.0.0.1:1919/v1/chat/completions"

_BASES = [
    "<|im_start|><|quad_end|><|object_ref_end|>{q}<|im_end|>",
]
_QUESTIONS = [
    "9.8 and 9.11, which is bigger?",
    "Explain why the sky is blue.",
    "What is the capital of France?",
    "Compute 17 * 23 step by step.",
    "Write a short poem about the ocean.",
    "Summarize the theory of relativity in one sentence.",
    "List three prime numbers greater than 50.",
    "Translate 'good morning' into three languages.",
]


def make_prompt(i: int) -> str:
    q = _QUESTIONS[i % len(_QUESTIONS)]
    # add some length variety
    pad = " ".join(["please be detailed."] * (i % 5))
    return _BASES[0].format(q=(q + " " + pad).strip())


def one_request(idx: int, max_tokens: int, results: list, lock: threading.Lock):
    prompt = make_prompt(idx)
    body = json.dumps(
        {
            "model": "hrm",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        content = resp["choices"][0]["message"]["content"]
        ok, err, n = True, None, len(content)
    except Exception as e:  # noqa: BLE001
        ok, err, n = False, f"{type(e).__name__}: {e}", 0
    dt = time.perf_counter() - t0
    with lock:
        results.append({"idx": idx, "ok": ok, "err": err, "chars": n, "dt": dt, "max_tokens": max_tokens})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    results: list = []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)
    threads = []

    def worker(i):
        with sem:
            one_request(i, args.max_tokens, results, lock)

    print(
        f"Stress: {args.num_requests} requests, concurrency {args.concurrency}, "
        f"{args.max_tokens} tokens each (ignore_eos)"
    )
    t0 = time.perf_counter()
    for i in range(args.num_requests):
        th = threading.Thread(target=worker, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    total_tokens = len(ok) * args.max_tokens
    print(f"\n=== RESULTS ===")
    print(f"Succeeded: {len(ok)}/{len(results)}")
    if bad:
        print(f"FAILED: {len(bad)}")
        for r in bad[:10]:
            print(f"  req {r['idx']}: {r['err']}")
    print(f"Wall time: {wall:.2f}s")
    print(f"Output throughput: {total_tokens / wall:.1f} tok/s")
    print(f"Request throughput: {len(ok) / wall:.2f} req/s")
    lats = sorted(r["dt"] for r in ok)
    if lats:
        print(f"Latency p50/p90/p99/max: "
              f"{lats[len(lats)//2]:.2f}/{lats[int(len(lats)*0.9)]:.2f}/"
              f"{lats[int(len(lats)*0.99)]:.2f}/{lats[-1]:.2f} s")
    print("\nSTRESS TEST:", "PASS (no errors)" if not bad else "FAIL")
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
