"""Serving benchmark for the mini-sglang HRM server.

Measures the metrics people usually care about for an LLM serving engine:
  - TTFT  : Time To First Token (prefill latency)
  - TPOT  : Time Per Output Token (inter-token latency during decode)
  - E2E   : end-to-end request latency
  - Output throughput (tokens/s) and request throughput (req/s)
with avg / p50 / p90 / p99 percentiles.

Uses streaming so TTFT/TPOT are measured directly, and the raw `prompt` field
(the HRM checkpoint has no chat template). One scenario per `--concurrency`.
"""

import argparse
import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:1919/v1/chat/completions"

_QUESTIONS = [
    "9.8 and 9.11, which is bigger?",
    "Explain why the sky is blue.",
    "Compute 17 * 23 step by step.",
    "Summarize the theory of relativity.",
    "List three prime numbers greater than 50.",
    "Describe the water cycle in detail.",
]


def make_prompt(i: int) -> str:
    q = _QUESTIONS[i % len(_QUESTIONS)]
    return f"<|im_start|><|quad_end|><|object_ref_end|>{q}<|im_end|>"


def stream_one(idx: int, max_tokens: int, out: list, lock: threading.Lock):
    prompt = make_prompt(idx)
    body = json.dumps(
        {
            "model": "hrm",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunk_times = []
    ok, err = True, None
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            delta = obj["choices"][0].get("delta", {})
            if delta.get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                chunk_times.append(now)
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()
    n_out = len(chunk_times)
    tpot = (chunk_times[-1] - chunk_times[0]) / (n_out - 1) if n_out > 1 else 0.0
    with lock:
        out.append(
            {"ok": ok, "err": err, "ttft": ttft, "tpot": tpot,
             "e2e": t_end - t0, "n_out": n_out}
        )


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run_scenario(num_requests, concurrency, max_tokens):
    out, lock = [], threading.Lock()
    sem = threading.Semaphore(concurrency)
    threads = []

    def worker(i):
        with sem:
            stream_one(i, max_tokens, out, lock)

    # warm a couple requests so JIT/graph state is hot before timing
    t0 = time.perf_counter()
    for i in range(num_requests):
        th = threading.Thread(target=worker, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    ok = [r for r in out if r["ok"]]
    fails = len(out) - len(ok)
    ttfts = [r["ttft"] * 1000 for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] * 1000 for r in ok if r["tpot"] > 0]
    e2es = [r["e2e"] for r in ok]
    total_out = sum(r["n_out"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "fails": fails,
        "ttft_avg": sum(ttfts) / len(ttfts) if ttfts else 0,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p90": pct(ttfts, 0.9), "ttft_p99": pct(ttfts, 0.99),
        "tpot_avg": sum(tpots) / len(tpots) if tpots else 0,
        "tpot_p99": pct(tpots, 0.99),
        "e2e_avg": sum(e2es) / len(e2es) if e2es else 0,
        "e2e_p99": pct(e2es, 0.99),
        "out_tok_s": total_out / wall if wall else 0,
        "req_s": len(ok) / wall if wall else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--requests-per-conc", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--markdown", action="store_true", help="print a markdown table row block")
    args = ap.parse_args()

    rows = []
    for c in args.concurrencies:
        n = max(c * 4, args.requests_per_conc)
        print(f"[scenario] concurrency={c}, requests={n}, max_tokens={args.max_tokens} ...", flush=True)
        rows.append(run_scenario(n, c, args.max_tokens))

    hdr = ("conc", "reqs", "fail", "TTFT avg", "TTFT p99", "TPOT avg", "TPOT p99",
           "E2E avg", "out tok/s", "req/s")
    print("\n" + " | ".join(f"{h:>9}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{v:>9}" for v in (
            r["concurrency"], r["requests"], r["fails"],
            f'{r["ttft_avg"]:.1f}ms', f'{r["ttft_p99"]:.1f}ms',
            f'{r["tpot_avg"]:.1f}ms', f'{r["tpot_p99"]:.1f}ms',
            f'{r["e2e_avg"]:.2f}s', f'{r["out_tok_s"]:.0f}', f'{r["req_s"]:.2f}')))

    if args.markdown:
        print("\n--- MARKDOWN ---")
        print("| Concurrency | Requests | Failures | TTFT avg | TTFT p99 | TPOT avg | TPOT p99 | E2E avg | Output tok/s | Req/s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['concurrency']} | {r['requests']} | {r['fails']} | "
                  f"{r['ttft_avg']:.1f} ms | {r['ttft_p99']:.1f} ms | "
                  f"{r['tpot_avg']:.1f} ms | {r['tpot_p99']:.1f} ms | "
                  f"{r['e2e_avg']:.2f} s | {r['out_tok_s']:.0f} | {r['req_s']:.2f} |")


if __name__ == "__main__":
    main()
