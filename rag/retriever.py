from __future__ import annotations

from typing import List, Tuple

from .config import DEFAULT_TOP_K, INDEX_PATH, METADATA_PATH
from .embeddings import EmbeddingModel
from .vectordb import VectorMetadata, VectorStore


class Retriever:
    def __init__(self) -> None:
        self.embedder = EmbeddingModel()
        self.store = VectorStore.load(INDEX_PATH, METADATA_PATH)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, VectorMetadata]]:
        q = self.embedder.embed_query(query)
        return self.store.search(q, top_k=top_k)

