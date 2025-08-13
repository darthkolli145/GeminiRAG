from __future__ import annotations

from textwrap import dedent

from .config import (
    LLM_BACKEND,
    LOCAL_LLM_MODEL_ID,
    LOCAL_MAX_NEW_TOKENS,
    LOCAL_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    LLM_DEVICE_MAP,
    LLM_LOAD_IN_4BIT,
    LLM_LOAD_IN_8BIT,
    LLM_TORCH_DTYPE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


SYSTEM_PROMPT = dedent(
    """
    You are a helpful assistant that answers questions using the provided context.
    - Use only the supplied context to answer.
    - If the answer cannot be found in the context, say you don't know.
    - Prefer direct, concise answers.
    - For factoid questions (who/what/where/when), reply with just the entity name or short phrase.
    """
).strip()


class LLM:
    def __init__(self) -> None:
        self.backend = LLM_BACKEND
        self._openai_client = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._gemini_model = None
        if self.backend == "openai" and OPENAI_API_KEY:
            from openai import OpenAI

            if OPENAI_BASE_URL:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            else:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.backend == "transformers":
            # Lazy import to avoid torch import when not needed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._hf_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_ID)
            # Determine dtype
            if LLM_TORCH_DTYPE == "float16":
                torch_dtype = torch.float16
            elif LLM_TORCH_DTYPE == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = "auto"
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": LLM_DEVICE_MAP,
            }
            # Quantization options
            if LLM_LOAD_IN_4BIT:
                load_kwargs["load_in_4bit"] = True
            elif LLM_LOAD_IN_8BIT:
                load_kwargs["load_in_8bit"] = True
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_LLM_MODEL_ID, **load_kwargs
            )
            if torch.cuda.is_available():
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        elif self.backend == "gemini" and GEMINI_API_KEY:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=GEMINI_API_KEY)
                # Prefer setting system prompt at model level
                self._gemini_model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    system_instruction=SYSTEM_PROMPT,
                )
            except Exception:
                self._gemini_model = None

    def _call_openai(self, prompt: str) -> str:
        assert self._openai_client is not None
        resp = self._openai_client.chat.completions.create(
            model=OPENAI_MODEL,
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
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Prefer chat template if available
        try:
            text = self._hf_tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = "\n\n".join([m["content"] for m in messages])
        inputs = self._hf_tokenizer(text, return_tensors="pt")
        input_length = int(inputs["input_ids"].shape[1])
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=LOCAL_TEMPERATURE,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )
        # Decode only new tokens to avoid prompt echo/offset mismatch
        new_tokens = output_ids[0][input_length:]
        generated = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not generated:
            # Fallback to full decode if slicing failed
            full_text = self._hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated = full_text[len(text):].strip() or full_text.strip()
        return generated

    def _call_gemini(self, prompt: str) -> str:
        assert self._gemini_model is not None
        resp = self._gemini_model.generate_content(prompt)
        try:
            text = getattr(resp, "text", None)
            if text:
                return text
        except Exception:
            pass
        try:
            candidates = getattr(resp, "candidates", [])
            if candidates:
                parts = getattr(candidates[0].content, "parts", [])
                texts = [getattr(p, "text", "") for p in parts]
                return "".join(texts).strip()
        except Exception:
            pass
        return ""

    def answer(self, prompt: str) -> str:
        try:
            if self.backend == "transformers":
                out = self._call_transformers(prompt)
                if not out.strip():
                    return self._call_extractive(prompt)
                return out
            if self.backend == "openai" and OPENAI_API_KEY:
                return self._call_openai(prompt)
            if self.backend == "gemini" and GEMINI_API_KEY and self._gemini_model is not None:
                out = self._call_gemini(prompt)
                if not out.strip():
                    return self._call_extractive(prompt)
                return out
            return self._call_extractive(prompt)
        except Exception:
            return self._call_extractive(prompt)

