from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def _apply_chat_template(self, messages: list) -> str:
        """Render OpenAI-style chat ``messages`` into a prompt string.

        Falls back to a minimal ``<|im_start|>…<|im_end|>`` envelope when the
        tokenizer ships no chat template (e.g. the pre-alignment HRM checkpoint),
        so the OpenAI-SDK chat-completions route still works.
        """
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            assert isinstance(prompt, str)
            return prompt
        return "".join(f"<|im_start|>{m['content']}<|im_end|>" for m in messages)

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self._apply_chat_template(msg.text)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
