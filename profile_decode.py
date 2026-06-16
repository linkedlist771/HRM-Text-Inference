"""Profile one decode-heavy run of the HRM model with CUDA graphs AND torch.compile
DISABLED, so the real per-op cost is visible (graphs/compile hide kernel launches
and fuse ops). Prints the top ops by CUDA time and writes a chrome trace.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from configs import MODEL_PATH

PROMPT = "<|im_start|><|quad_end|><|object_ref_end|>Explain why the sky is blue.<|im_end|>"


def main():
    llm = LLM(
        str(MODEL_PATH),
        attention_backend="fi",
        cache_type="naive",
        cuda_graph_max_bs=0,        # <-- CUDA graphs OFF
        enable_torch_compile=False,  # <-- torch.compile OFF
        num_page_override=8192,
        max_seq_len_override=2048,
    )

    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=64)
    # warm up flashinfer JIT / first-call compiles
    llm.generate([PROMPT], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8))

    prompts = [PROMPT] * 8  # decode batch size 8
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate(prompts, sp)
    torch.cuda.synchronize()

    print("\n================ TOP OPS BY CUDA TIME ================")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace("trace_decode.json")
    print("\nchrome trace -> trace_decode.json")


if __name__ == "__main__":
    main()
