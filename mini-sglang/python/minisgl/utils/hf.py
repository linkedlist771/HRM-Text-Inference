import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


def _apply_hrm_post_init(raw: dict) -> None:
    """Replicate transformers ``HrmTextConfig.__post_init__`` so the config is
    usable even on a transformers version that predates the ``hrm_text`` class."""
    raw.setdefault("hidden_act", "silu")
    if raw.get("L_bp_cycles") is None:
        raw["L_bp_cycles"] = [2]
    if raw.get("embedding_scale") is None:
        raw["embedding_scale"] = 1.0 / raw["initializer_range"]
    if raw.get("num_layers_per_stack") is None:
        # The serialized ``num_hidden_layers`` is the per-stack count; inflate it to
        # one slot per unique attention invocation under the recurrent forward.
        nps = raw["num_hidden_layers"]
        raw["num_layers_per_stack"] = nps
        raw["num_hidden_layers"] = nps * raw["H_cycles"] * (raw["L_cycles"] + 1)


def _load_config_from_json(model_path: str) -> Any:
    if os.path.isdir(model_path):
        path = os.path.join(model_path, "config.json")
    else:
        path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("model_type") == "hrm_text":
        _apply_hrm_post_init(raw)
    return PretrainedConfig(**raw)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except ValueError:
        # e.g. ``hrm_text`` needs transformers>=5.9; fall back to config.json so
        # mini-sglang can still serve the model with its own implementation.
        return _load_config_from_json(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )