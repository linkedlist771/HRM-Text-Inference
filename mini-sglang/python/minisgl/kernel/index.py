from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)

# The custom CUDA kernel requires a C++20 host compiler (std::source_location).
# On toolchains that lack one, fall back to a pure-torch gather, decided once.
_USE_JIT: bool | None = None


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def _indexing_torch(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor | None,
    vocab_range: Tuple[int, int] | None,
) -> torch.Tensor:
    idx = indices.long()
    if vocab_range is None:
        gathered = weights[idx]
    else:
        start, length = vocab_range
        local = idx - start
        mask = (local >= 0) & (local < length)
        gathered = weights[local.clamp_(0, length - 1)] * mask.unsqueeze(-1)
    if output is not None:
        output.copy_(gathered)
        return output
    return gathered


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    global _USE_JIT
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1

    if _USE_JIT is None:
        try:
            _jit_index_module(element_size, num_splits=num_splits)
            _USE_JIT = True
        except Exception:
            _USE_JIT = False

    if not _USE_JIT:
        return _indexing_torch(weights, indices, output, vocab_range)

    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
