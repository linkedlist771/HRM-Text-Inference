from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    get_rope,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


@triton.jit
def _sigmoid_mul_kernel(
    gate,
    x,
    out,
    n_cols: tl.constexpr,
    gate_stride0: tl.constexpr,
    x_stride0: tl.constexpr,
    out_stride0: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    gate_vals = tl.load(gate + row * gate_stride0 + cols, mask=mask, other=0.0).to(tl.float32)
    x_vals = tl.load(x + row * x_stride0 + cols, mask=mask, other=0.0)
    scale = 1.0 / (1.0 + tl.exp(-gate_vals))
    tl.store(out + row * out_stride0 + cols, x_vals * scale, mask=mask)


def _sigmoid_mul(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _sigmoid_mul_kernel[(gate.shape[0], triton.cdiv(gate.shape[1], 1024))](
        gate,
        x,
        out,
        gate.shape[1],
        gate.stride(0),
        x.stride(0),
        out.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return out


class HrmRMSNorm(BaseOP):
    """Parameterless Pre-RMSNorm (no learnable weight).

    Matches transformers ``HrmTextRMSNorm``. Uses flashinfer's fused rmsnorm kernel
    (one launch instead of ~6 eager pow/mean/rsqrt/cast/mul ops) with an all-ones
    weight — the HRM checkpoint has no norm weights. The ones weight is built lazily
    (underscore attr) so it stays out of the state dict.
    """

    def __init__(self, eps: float) -> None:
        from flashinfer import rmsnorm

        self._eps = eps
        self._rmsnorm = rmsnorm
        self._weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._weight
        if w is None or w.device != x.device or w.dtype != x.dtype:
            w = self._weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return self._rmsnorm(x, w, self._eps)


class HrmAttention(BaseOP):
    """Gated multi-head attention with a fused gate/q/k/v projection.

    The HRM checkpoint packs the projection as ``gqkv_proj`` with output order
    ``[gate, q, k, v]`` (see transformers ``_checkpoint_conversion_mapping``).
    A sigmoid gate is applied to the attention output before ``o_proj``.

    ``cycle_offset`` selects the KV-cache slot for the current recurrent step:
    ``layer_id = cycle_offset + layer_idx``. There are
    ``num_layers_per_stack * H_cycles * (L_cycles + 1)`` such slots in total.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        self._layer_idx = layer_idx
        self._head_dim = config.head_dim
        tp_size = get_tp_info().size
        self._num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self._num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self._qo_dim = self._num_qo_heads * self._head_dim
        self._kv_dim = self._num_kv_heads * self._head_dim

        full_qo = config.num_qo_heads * config.head_dim
        full_kv = config.num_kv_heads * config.head_dim
        # Fused projection, output order: [gate, q, k, v]
        self.gqkv_proj = LinearColParallelMerged(
            config.hidden_size,
            [full_qo, full_qo, full_kv, full_kv],
            has_bias=False,
        )
        self.o_proj = LinearOProj(full_qo, config.hidden_size, has_bias=False)

        rotary = config.rotary_config
        self._rotary = get_rope(
            head_dim=rotary.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )

    @nvtx_annotate("HRM_MHA")
    def forward(self, x: torch.Tensor, cycle_offset: int) -> torch.Tensor:
        ctx = get_global_ctx()
        gqkv = self.gqkv_proj.forward(x)
        gate, q, k, v = gqkv.split(
            [self._qo_dim, self._qo_dim, self._kv_dim, self._kv_dim], dim=-1
        )
        # No .contiguous(): flashinfer rope / KV-store accept the strided split views
        # (same as minisgl's stock RopeAttn), avoiding 3 clone copies per attention.
        q, k = self._rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self._num_qo_heads, self._head_dim)
        o = ctx.attn_backend.forward(q, k, v, cycle_offset + self._layer_idx, ctx.batch)
        o = o.view(-1, self._qo_dim)
        o = _sigmoid_mul(gate, o)
        return self.o_proj.forward(o)


class HrmDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        self.attn = HrmAttention(config, layer_idx)
        self.mlp = GatedMLP(config)
        self.input_layernorm = HrmRMSNorm(config.rms_norm_eps)
        self.post_attention_layernorm = HrmRMSNorm(config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cycle_offset: int) -> torch.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.attn.forward(x, cycle_offset)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        x = residual + x
        return x


class HrmStack(BaseOP):
    """A single transformer stack, used twice (as the H module and the L module)."""

    def __init__(self, config: ModelConfig) -> None:
        self.layers = OPList(
            [HrmDecoderLayer(config, i) for i in range(config.num_layers_per_stack)]
        )
        self.final_norm = HrmRMSNorm(config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cycle_offset: int) -> torch.Tensor:
        for layer in self.layers.op_list:
            x = layer.forward(x, cycle_offset)
        return self.final_norm.forward(x)


class HrmModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.L_module = HrmStack(config)
        self.H_module = HrmStack(config)

        self._embedding_scale = config.embedding_scale
        self._H_cycles = config.H_cycles
        self._L_cycles = config.L_cycles
        self._num_layers_per_stack = config.num_layers_per_stack
        self._L_layers = self.L_module.layers.op_list
        self._H_layers = self.H_module.layers.op_list
        self._L_final_norm = self.L_module.final_norm
        self._H_final_norm = self.H_module.final_norm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # z_H: slow / high-level state. z_L: fast / low-level state (init = 0).
        z_H = self.embed_tokens.forward(input_ids) * self._embedding_scale
        z_L = torch.zeros_like(z_H)

        nps = self._num_layers_per_stack
        for h in range(self._H_cycles):
            for l in range(self._L_cycles):
                offset = (h * (self._L_cycles + 1) + l) * nps
                z_L = z_L + z_H
                for layer in self._L_layers:
                    z_L = layer.forward(z_L, offset)
                z_L = self._L_final_norm.forward(z_L)
            offset = (h * (self._L_cycles + 1) + self._L_cycles) * nps
            z_H = z_H + z_L
            for layer in self._H_layers:
                z_H = layer.forward(z_H, offset)
            z_H = self._H_final_norm.forward(z_H)
        return z_H


class HrmTextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self.model = HrmModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
            tied_embedding=None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["HrmTextForCausalLM"]
