from __future__ import annotations

import os
from textwrap import dedent
from typing import List

from .config import (
    LLM_BACKEND,
    LOCAL_LLM_MODEL_ID,
    LOCAL_MAX_NEW_TOKENS,
    LOCAL_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


SYSTEM_PROMPT = dedent(
    """
    You are a helpful assistant that answers questions using the provided context.
    - Use only the supplied context to answer.
    - If the answer cannot be found in the context, say you don't know.
    - Be concise and clear.
    """
).strip()


class LLM:
    def __init__(self) -> None:
        self.backend = LLM_BACKEND
        self._openai_client = None
        self._hf_model = None
        self._hf_tokenizer = None
        if self.backend == "openai" and OPENAI_API_KEY:
            from openai import OpenAI

            if OPENAI_BASE_URL:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            else:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.backend == "transformers":
            # Lazy import to avoid torch import when not needed
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_ID)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_LLM_MODEL_ID, torch_dtype="auto"
            )

    def _call_openai(self, prompt: str) -> str:
        assert self._openai_client is not None
        resp = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    def _call_extractive(self, prompt: str) -> str:
        # Fallback: just return the context section of the prompt after the marker.
        # This is a simple, dependency-free extractive response.
        marker = "Context:\n"
        if marker in prompt:
            context = prompt.split(marker, 1)[1]
            return context.strip()[:1200]
        return "I don't know."

    def _call_transformers(self, prompt: str) -> str:
        assert self._hf_model is not None and self._hf_tokenizer is not None
        from transformers import TextIteratorStreamer
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Simple chat template: system + user -> assistant
        text = "\n\n".join([m["content"] for m in messages])
        inputs = self._hf_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=LOCAL_TEMPERATURE,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )
        full_text = self._hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Heuristic: return the tail after the prompt
        return full_text[len(text):].strip() or full_text.strip()

    def answer(self, prompt: str) -> str:
        try:
            if self.backend == "transformers":
                return self._call_transformers(prompt)
            if self.backend == "openai" and OPENAI_API_KEY:
                return self._call_openai(prompt)
            return self._call_extractive(prompt)
        except Exception:
            return self._call_extractive(prompt)

