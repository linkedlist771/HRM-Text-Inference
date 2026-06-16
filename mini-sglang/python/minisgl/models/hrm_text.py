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
        num_warps=1,
    )
    return out


@triton.jit
def _state_add_rmsnorm_kernel(
    state,
    update,
    out,
    n_cols: tl.constexpr,
    state_stride0: tl.constexpr,
    update_stride0: tl.constexpr,
    out_stride0: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    state_vals = tl.load(state + row * state_stride0 + cols, mask=mask, other=0.0).to(tl.float32)
    update_vals = tl.load(update + row * update_stride0 + cols, mask=mask, other=0.0).to(tl.float32)
    raw = (state_vals + update_vals).to(tl.bfloat16)
    raw_f32 = raw.to(tl.float32)
    mean_square = tl.sum(raw_f32 * raw_f32, axis=0) / n_cols
    scale = tl.rsqrt(mean_square + eps)
    tl.store(state + row * state_stride0 + cols, raw, mask=mask)
    tl.store(out + row * out_stride0 + cols, raw_f32 * scale, mask=mask)


def _state_add_rmsnorm(state: torch.Tensor, update: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(state)
    _state_add_rmsnorm_kernel[(state.shape[0],)](
        state,
        update,
        out,
        state.shape[1],
        state.stride(0),
        update.stride(0),
        out.stride(0),
        eps,
        BLOCK=2048,
        num_warps=4,
        num_stages=1,
    )
    return out


def _residual_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    _state_add_rmsnorm_kernel[(residual.shape[0],)](
        residual,
        x,
        x,
        residual.shape[1],
        residual.stride(0),
        x.stride(0),
        x.stride(0),
        eps,
        BLOCK=2048,
        num_warps=4,
        num_stages=1,
    )
    return x, residual


@triton.jit
def _rope_store_kv_kernel(
    q,
    k,
    v,
    k_cache,
    v_cache,
    indices,
    positions,
    cos_sin_cache,
    q_stride0: tl.constexpr,
    k_stride0: tl.constexpr,
    v_stride0: tl.constexpr,
    cos_sin_stride0: tl.constexpr,
    n_elems: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, HALF)

    pos = tl.load(positions + token).to(tl.int64)
    cos = tl.load(cos_sin_cache + pos * cos_sin_stride0 + offs).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_stride0 + HALF + offs).to(tl.float32)

    head_offs = head * HEAD_DIM
    q_base = q + token * q_stride0 + head_offs
    k_base = k + token * k_stride0 + head_offs
    v_base = v + token * v_stride0 + head_offs

    q0 = tl.load(q_base + offs).to(tl.float32)
    q1 = tl.load(q_base + HALF + offs).to(tl.float32)
    tl.store(q_base + offs, q0 * cos - q1 * sin)
    tl.store(q_base + HALF + offs, q1 * cos + q0 * sin)

    k0 = tl.load(k_base + offs).to(tl.float32)
    k1 = tl.load(k_base + HALF + offs).to(tl.float32)
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k1 * cos + k0 * sin

    dst = tl.load(indices + token).to(tl.int64) * n_elems + head_offs
    tl.store(k_cache + dst + offs, k_rot0)
    tl.store(k_cache + dst + HALF + offs, k_rot1)
    tl.store(v_cache + dst + offs, tl.load(v_base + offs))
    tl.store(v_cache + dst + HALF + offs, tl.load(v_base + HALF + offs))


def _rope_store_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
) -> None:
    n_elems = k.shape[1]
    _rope_store_kv_kernel[(k.shape[0], triton.cdiv(n_elems, head_dim))](
        q,
        k,
        v,
        k_cache,
        v_cache,
        indices,
        positions,
        cos_sin_cache,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        cos_sin_cache.stride(0),
        n_elems,
        HEAD_DIM=head_dim,
        HALF=head_dim // 2,
        num_warps=1,
    )


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

    def _get_weight(self, x: torch.Tensor) -> torch.Tensor:
        w = self._weight
        if w is None or w.device != x.device or w.dtype != x.dtype:
            w = self._weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x)
        return self._rmsnorm(x, w, self._eps)

    def forward_after_residual_add(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _residual_add_rmsnorm(x, residual, self._eps)

    def forward_after_state_add(
        self, state: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _state_add_rmsnorm(state, update, self._eps), state


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
        backend = ctx.attn_backend
        metadata = ctx.batch.attn_metadata
        backend._initialize_metadata_once(metadata)
        layer_id = cycle_offset + self._layer_idx
        k_cache = backend.kvcache.k_cache(layer_id)
        v_cache = backend.kvcache.v_cache(layer_id)
        _rope_store_kv(
            q,
            k,
            v,
            k_cache,
            v_cache,
            ctx.batch.out_loc,
            ctx.batch.positions,
            self._rotary._cos_sin_cache,
            self._head_dim,
        )
        q = q.view(-1, self._num_qo_heads, self._head_dim)
        kv_cache = (
            k_cache.view(-1, 1, k_cache.shape[2], k_cache.shape[3]),
            v_cache.view(-1, 1, v_cache.shape[2], v_cache.shape[3]),
        )
        o = metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
        o = o.view(-1, self._qo_dim)
        o = _sigmoid_mul(gate, o)
        return self.o_proj.forward(o)


class HrmDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        self.attn = HrmAttention(config, layer_idx)
        self.mlp = GatedMLP(config)
        self.input_layernorm = HrmRMSNorm(config.rms_norm_eps)
        self.post_attention_layernorm = HrmRMSNorm(config.rms_norm_eps)

    def forward_normed(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cycle_offset: int,
        output_norm: HrmRMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.attn.forward(x, cycle_offset)
        x, residual = self.post_attention_layernorm.forward_after_residual_add(x, residual)
        x = self.mlp.forward(x)
        return output_norm.forward_after_residual_add(x, residual)

    def forward(self, x: torch.Tensor, cycle_offset: int) -> torch.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        _, x = self.forward_normed(x, residual, cycle_offset, self.input_layernorm)
        return x


class HrmStack(BaseOP):
    """A single transformer stack, used twice (as the H module and the L module)."""

    def __init__(self, config: ModelConfig) -> None:
        self.layers = OPList(
            [HrmDecoderLayer(config, i) for i in range(config.num_layers_per_stack)]
        )
        self.final_norm = HrmRMSNorm(config.rms_norm_eps)
        layers = self.layers.op_list
        self._layer_norm_pairs = tuple(
            zip(layers, [layer.input_layernorm for layer in layers[1:]] + [self.final_norm])
        )

    def forward(self, x: torch.Tensor, cycle_offset: int) -> torch.Tensor:
        normed = self.layers.op_list[0].input_layernorm.forward(x)
        for layer, next_norm in self._layer_norm_pairs:
            normed, x = layer.forward_normed(normed, x, cycle_offset, next_norm)
        return normed


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
        self._L_layer_norm_pairs = self.L_module._layer_norm_pairs
        self._H_layer_norm_pairs = self.H_module._layer_norm_pairs

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # z_H: slow / high-level state. z_L: fast / low-level state (init = 0).
        z_H = self.embed_tokens.forward(input_ids) * self._embedding_scale
        z_L = torch.zeros_like(z_H)

        nps = self._num_layers_per_stack
        for h in range(self._H_cycles):
            for l in range(self._L_cycles):
                offset = (h * (self._L_cycles + 1) + l) * nps
                normed, z_L = self._L_layers[0].input_layernorm.forward_after_state_add(z_L, z_H)
                for layer, next_norm in self._L_layer_norm_pairs:
                    normed, z_L = layer.forward_normed(normed, z_L, offset, next_norm)
                z_L = normed
            offset = (h * (self._L_cycles + 1) + self._L_cycles) * nps
            normed, z_H = self._H_layers[0].input_layernorm.forward_after_state_add(z_H, z_L)
            for layer, next_norm in self._H_layer_norm_pairs:
                normed, z_H = layer.forward_normed(normed, z_H, offset, next_norm)
            z_H = normed
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
