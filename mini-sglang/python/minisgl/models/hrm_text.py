from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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


class HrmRMSNorm(BaseOP):
    """Parameterless Pre-RMSNorm (no learnable weight), computed in float32.

    Matches transformers ``HrmTextRMSNorm`` exactly. It owns no public tensors, so it
    contributes nothing to the state dict — the HRM checkpoint has no norm weights.
    """

    def __init__(self, eps: float) -> None:
        from flashinfer import rmsnorm

        self._eps = eps
        self._rmsnorm = rmsnorm
        self._weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._weight
        if weight is None or weight.device != x.device or weight.dtype != x.dtype:
            weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
            self._weight = weight
        return self._rmsnorm(x, weight, self._eps)


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
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k = self._rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self._num_qo_heads, self._head_dim)
        o = ctx.attn_backend.forward(q, k, v, cycle_offset + self._layer_idx, ctx.batch)
        o = o.view(-1, self._qo_dim)
        o = torch.sigmoid(gate) * o
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # z_H: slow / high-level state. z_L: fast / low-level state (init = 0).
        z_H = self.embed_tokens.forward(input_ids) * self._embedding_scale
        z_L = torch.zeros_like(z_H)

        nps = self._num_layers_per_stack
        for h in range(self._H_cycles):
            for l in range(self._L_cycles):
                offset = (h * (self._L_cycles + 1) + l) * nps
                z_L = self.L_module.forward(z_L + z_H, offset)
            offset = (h * (self._L_cycles + 1) + self._L_cycles) * nps
            z_H = self.H_module.forward(z_H + z_L, offset)
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
