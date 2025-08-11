from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

