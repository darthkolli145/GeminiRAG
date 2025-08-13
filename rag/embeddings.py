from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_SHOW_PROGRESS,
    EMBEDDING_BACKEND,
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
)


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.backend = EMBEDDING_BACKEND
        self._use_gemini = False
        self._genai = None
        self._gemini_model_name = GEMINI_EMBED_MODEL

        # Try Gemini first if requested and API key present
        if self.backend == "gemini" and GEMINI_API_KEY:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=GEMINI_API_KEY)
                self._genai = genai
                self._use_gemini = True
            except Exception:
                self._use_gemini = False

        if not self._use_gemini:
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    try:
                        torch.set_float32_matmul_precision("high")
                    except Exception:
                        pass
                self.model = SentenceTransformer(self.model_name, device=device)
            except Exception:
                # Fallback to default device
                self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self._use_gemini and self._genai is not None:
            vectors = []
            for text in texts:
                try:
                    resp = self._genai.embed_content(model=self._gemini_model_name, content=text)
                    vec = resp["embedding"] if isinstance(resp, dict) else resp.embedding
                    vectors.append(vec)
                except Exception:
                    vectors.append([0.0])
            arr = np.array(vectors, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=EMBEDDING_SHOW_PROGRESS,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

