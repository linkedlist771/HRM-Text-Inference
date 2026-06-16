"""Head-to-head decode-throughput benchmark: HuggingFace transformers vs. the
mini-sglang HRM engine, on identical workloads. Produces a table and a
vLLM/SGLang-blog-style grouped bar chart.

Run each engine in its own process (they each own the GPU), then plot:
    python bench_compare.py --engine hf
    python bench_compare.py --engine minisgl
    python bench_compare.py --plot
"""

import argparse
import json
import os
import time

from configs import MODEL_PATH

RESULTS = "compare_results.json"
# decode-dominated: short prompt, fixed-length greedy continuation
PROMPT = (
    "<|im_start|><|quad_end|><|object_ref_end|>"
    "Explain step by step why the sky is blue and how rainbows form.<|im_end|>"
)
GEN = 128
BATCH_SIZES = [1, 8, 16, 32]


def _save(engine: str, data: dict) -> None:
    allr = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    allr[engine] = data
    json.dump(allr, open(RESULTS, "w"), indent=2)


def bench_hf() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
    res = {}
    for bs in BATCH_SIZES:
        enc = tok([PROMPT] * bs, return_tensors="pt").to("cuda")
        enc["token_type_ids"] = torch.ones_like(enc["input_ids"])  # PrefixLM: whole prompt
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False)  # warmup
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=GEN, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        tps = bs * GEN / dt
        res[str(bs)] = tps
        print(f"HF          bs={bs:3d}: {tps:8.1f} tok/s  ({dt:5.2f}s)", flush=True)
    _save("hf", res)


def bench_minisgl() -> None:
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        str(MODEL_PATH),
        attention_backend="fi",
        cache_type="naive",
        cuda_graph_max_bs=max(BATCH_SIZES),
        num_page_override=8192,
        max_seq_len_override=2048,
    )

    def sp():
        return SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=GEN)

    llm.generate([PROMPT], sp())  # warmup
    res = {}
    for bs in BATCH_SIZES:
        t = time.perf_counter()
        llm.generate([PROMPT] * bs, sp())
        dt = time.perf_counter() - t
        tps = bs * GEN / dt
        res[str(bs)] = tps
        print(f"mini-sglang bs={bs:3d}: {tps:8.1f} tok/s  ({dt:5.2f}s)", flush=True)
    _save("minisgl", res)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    allr = json.load(open(RESULTS))
    hf, ms = allr["hf"], allr["minisgl"]
    bss = sorted(set(int(k) for k in hf) & set(int(k) for k in ms))
    hf_v = [hf[str(b)] for b in bss]
    ms_v = [ms[str(b)] for b in bss]

    print("\n| Batch size | HF transformers (tok/s) | mini-sglang (tok/s) | Speedup |")
    print("|---|---|---|---|")
    for b, h, m in zip(bss, hf_v, ms_v):
        print(f"| {b} | {h:.0f} | {m:.0f} | **{m / h:.1f}×** |")

    x = np.arange(len(bss))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - w / 2, hf_v, w, label="HuggingFace transformers", color="#9aa0a6")
    b2 = ax.bar(x + w / 2, ms_v, w, label="mini-sglang (ours)", color="#4285f4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={b}" for b in bss])
    ax.set_ylabel("Decode throughput (tokens / s)")
    ax.set_title("HRM-Text-1B throughput: HuggingFace vs mini-sglang\n(RTX 4090, bf16, 128 output tokens)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(
                f"{r.get_height():.0f}",
                (r.get_x() + r.get_width() / 2, r.get_height()),
                ha="center", va="bottom", fontsize=8,
            )
    for i, (h, m) in enumerate(zip(hf_v, ms_v)):
        ax.annotate(f"{m / h:.1f}×", (x[i] + w / 2, m), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#1a73e8", xytext=(0, 12),
                    textcoords="offset points")
    fig.tight_layout()
    os.makedirs("assets", exist_ok=True)
    fig.savefig("assets/throughput_comparison.png", dpi=150)
    print("\nsaved assets/throughput_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "minisgl"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.engine == "hf":
        bench_hf()
    elif args.engine == "minisgl":
        bench_minisgl()
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
