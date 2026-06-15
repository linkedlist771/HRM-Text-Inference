"""Standalone correctness check for the mini-sglang HRM port.

It reimplements the HRM forward EXACTLY as ported into
``minisgl/models/hrm_text.py`` (fused gqkv/gate_up split, parameterless RMSNorm,
embedding scaling, sigmoid-gated attention, H/L recurrence, PrefixLM mask) using
plain PyTorch + the raw checkpoint tensors, then compares logits against the
HuggingFace reference (the ground truth from hf_inference.py).

This validates the algorithm independently of the flashinfer/minisgl runtime.
"""

import torch
import torch.nn.functional as F
import safetensors.torch as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from configs import MODEL_PATH

DEV = "cuda"
DT = torch.bfloat16

cfg = AutoConfig.from_pretrained(MODEL_PATH)
H_CYCLES = cfg.H_cycles
L_CYCLES = cfg.L_cycles
NPS = cfg.num_layers_per_stack
HID = cfg.hidden_size
NH = cfg.num_attention_heads
HD = cfg.head_dim
SCALE = HD**-0.5
EMB_SCALE = cfg.embedding_scale
EPS = cfg.rms_norm_eps
THETA = cfg.rope_parameters["rope_theta"]
QO = NH * HD  # qkv share dim (MHA)

W = st.load_file(f"{MODEL_PATH}/model.safetensors")
W = {k: v.to(DEV, DT) for k, v in W.items()}


def rmsnorm(x):
    d = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return x.to(d)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(seq_len):
    inv_freq = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=DEV).float() / HD))
    pos = torch.arange(seq_len, device=DEV).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(DT), emb.sin().to(DT)  # (S, HD)


def attn(x, pfx, cos, sin, mask):
    S = x.shape[0]
    gqkv = x @ W[f"{pfx}.attn.gqkv_proj.weight"].T  # (S, 4*QO)
    gate, q, k, v = gqkv.split([QO, QO, QO, QO], dim=-1)
    # (S, NH, HD) -> (NH, S, HD)
    q = q.view(S, NH, HD).transpose(0, 1)
    k = k.view(S, NH, HD).transpose(0, 1)
    v = v.view(S, NH, HD).transpose(0, 1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    scores = (q.float() @ k.float().transpose(-1, -2)) * SCALE + mask
    p = scores.softmax(-1).to(DT)
    o = (p @ v).transpose(0, 1).reshape(S, QO)  # (S, QO)
    o = torch.sigmoid(gate) * o
    return o @ W[f"{pfx}.attn.o_proj.weight"].T


def mlp(x, pfx):
    gu = x @ W[f"{pfx}.mlp.gate_up_proj.weight"].T
    gate, up = gu.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ W[f"{pfx}.mlp.down_proj.weight"].T


def stack(x, which, cos, sin, mask):
    for i in range(NPS):
        pfx = f"model.{which}_module.layers.{i}"
        x = x + attn(rmsnorm(x), pfx, cos, sin, mask)
        x = x + mlp(rmsnorm(x), pfx)
    return rmsnorm(x)  # final_norm


def forward(ids, num_prompt):
    S = ids.shape[0]
    cos, sin = rope_cos_sin(S)  # (S, HD), broadcast over heads
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    # PrefixLM mask: prompt block bidirectional, generated tokens causal.
    allow = torch.tril(torch.ones(S, S, device=DEV, dtype=torch.bool))
    allow[:num_prompt, :num_prompt] = True  # prompt is one bidirectional block
    mask = torch.where(allow, 0.0, float("-inf")).to(torch.float32)

    z_H = W["model.embed_tokens.weight"][ids] * EMB_SCALE
    z_L = torch.zeros_like(z_H)
    for _ in range(H_CYCLES):
        for _ in range(L_CYCLES):
            z_L = stack(z_L + z_H, "L", cos, sin, mask)
        z_H = stack(z_H + z_L, "H", cos, sin, mask)
    return z_H @ W["lm_head.weight"].T  # (S, vocab)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    query = "9.8 and 9.11, which is bigger?"
    condition = "<|quad_end|><|object_ref_end|>"
    prompt = f"<|im_start|>{condition}{query}<|im_end|>"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(DEV)
    P = ids.shape[0]

    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DT).to(DEV).eval()

    # ---- Test 1: full prompt, all-bidirectional (matches minisgl prefill) ----
    with torch.no_grad():
        hf_logits = hf(
            input_ids=ids[None], token_type_ids=torch.ones_like(ids)[None]
        ).logits[0]
        my_logits = forward(ids, num_prompt=P)
    _report("prefill (all-prompt, bidirectional)", hf_logits, my_logits)

    # ---- Test 2: prompt + greedy-generated tokens (mixed prefix/causal mask) ----
    with torch.no_grad():
        gen = hf.generate(
            ids[None],
            token_type_ids=torch.ones_like(ids)[None],
            max_new_tokens=32,
            do_sample=False,
        )[0]
    full = gen.to(DEV)
    ttids = torch.zeros_like(full)
    ttids[:P] = 1
    with torch.no_grad():
        hf_logits2 = hf(input_ids=full[None], token_type_ids=ttids[None]).logits[0]
        my_logits2 = forward(full, num_prompt=P)
    _report("prompt+gen (mixed mask)", hf_logits2, my_logits2)

    # greedy next-token agreement over the whole sequence
    hf_next = hf_logits2.argmax(-1)
    my_next = my_logits2.argmax(-1)
    agree = (hf_next == my_next).float().mean().item()
    print(f"  next-token argmax agreement: {agree*100:.1f}%")
    print("\nReference greedy continuation:")
    print(tok.decode(gen[P:], skip_special_tokens=False))


def _report(name, a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = (diff.max() / a.abs().max()).item()
    arg = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"[{name}] max|Δ|={diff.max():.4f} mean|Δ|={diff.mean():.5f} "
          f"rel={rel:.4f} argmax-agree={arg*100:.1f}%")


if __name__ == "__main__":
    main()
