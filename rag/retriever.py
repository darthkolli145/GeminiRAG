from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .config import (
    DEFAULT_TOP_K,
    INDEX_PATH,
    METADATA_PATH,
    RERANK_STRATEGY,
    RERANK_CANDIDATE_MULTIPLIER,
)
from .embeddings import EmbeddingModel
from .vectordb import VectorMetadata, VectorStore
from .utils import split_into_sentences


class Retriever:
    def __init__(self) -> None:
        # Cache a single embedder across instances to avoid reloading the model
        global _EMBEDDER_SINGLETON
        try:
            _EMBEDDER_SINGLETON
        except NameError:
            _EMBEDDER_SINGLETON = None  # type: ignore
        if _EMBEDDER_SINGLETON is None:
            _EMBEDDER_SINGLETON = EmbeddingModel()  # type: ignore
        self.embedder = _EMBEDDER_SINGLETON  # type: ignore
        self.store = VectorStore.load(INDEX_PATH, METADATA_PATH)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, VectorMetadata]]:
        q = self.embedder.embed_query(query)
        if RERANK_STRATEGY == "maxsim":
            candidate_k = min(self.store.size, max(top_k, top_k * max(1, RERANK_CANDIDATE_MULTIPLIER)))
            initial = self.store.search(q, top_k=candidate_k)
            if not initial:
                return []
            # Build sentence lists per candidate
            sentences_per_idx: List[List[str]] = []
            meta_per_idx: List[VectorMetadata] = []
            for _, meta in initial:
                sentences = split_into_sentences(meta.text, max_sentences=32)
                if not sentences:
                    sentences = [meta.text]
                sentences_per_idx.append(sentences)
                meta_per_idx.append(meta)
            # Flatten for a single batched embedding call
            flat_sentences: List[str] = [s for sentences in sentences_per_idx for s in sentences]
            sent_embeddings = self.embedder.embed_texts(flat_sentences)
            # Compute start offsets for each candidate
            offsets: List[Tuple[int, int]] = []
            cursor = 0
            for sentences in sentences_per_idx:
                start = cursor
                end = cursor + len(sentences)
                offsets.append((start, end))
                cursor = end
            # MaxSim: score by max dot product (embeddings are normalized)
            scores: List[float] = []
            for start, end in offsets:
                if start == end:
                    scores.append(-1.0)
                    continue
                chunk_matrix = sent_embeddings[start:end]
                sims = np.dot(chunk_matrix, q)
                scores.append(float(np.max(sims)))
            ranked = sorted(zip(scores, meta_per_idx), key=lambda x: x[0], reverse=True)[:top_k]
            return ranked
        # Default
        return self.store.search(q, top_k=top_k)

