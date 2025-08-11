from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors  # type: ignore

from .config import INDEX_PATH, METADATA_PATH, VECTORS_PATH
from .utils import read_json, write_json


@dataclass
class VectorMetadata:
    text: str
    source: str
    chunk_id: int
    document_id: str


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) if HAS_FAISS else None
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine") if not HAS_FAISS else None
        self.metadatas: List[VectorMetadata] = []
        self._vectors: np.ndarray | None = None  # used when FAISS is unavailable
        self._is_fitted: bool = False

    @property
    def size(self) -> int:
        if HAS_FAISS:
            assert self.index is not None
            return int(self.index.ntotal)
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray, metadatas: List[VectorMetadata]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("Vectors shape mismatch with index dimension")
        if HAS_FAISS:
            assert self.index is not None
            self.index.add(vectors)
        else:
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            # Delay fitting until finalize or search to avoid repeated refits
            self._is_fitted = False
        self.metadatas.extend(metadatas)

    def finalize(self) -> None:
        if HAS_FAISS:
            return
        if self._vectors is None or len(self.metadatas) == 0:
            self._is_fitted = True
            return
        n_neighbors = min(10, len(self.metadatas))
        self.nn.set_params(n_neighbors=n_neighbors)
        self.nn.fit(self._vectors)
        self._is_fitted = True

    def search(self, vector: np.ndarray, top_k: int = 4) -> List[Tuple[float, VectorMetadata]]:
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.dtype != np.float32:
            vector = vector.astype("float32")
        results: List[Tuple[float, VectorMetadata]] = []
        if HAS_FAISS:
            assert self.index is not None
            scores, indices = self.index.search(vector, top_k)
            for idx, score in zip(indices[0], scores[0]):
                if idx == -1:
                    continue
                meta = self.metadatas[idx]
                results.append((float(score), meta))
        else:
            if self._vectors is None or len(self.metadatas) == 0:
                return []
            if not self._is_fitted:
                # Lazy fit on first search after additions
                self.finalize()
            n = min(top_k, len(self.metadatas))
            self.nn.set_params(n_neighbors=n)
            distances, indices = self.nn.kneighbors(vector, n_neighbors=n)
            for idx, dist in zip(indices[0], distances[0]):
                meta = self.metadatas[idx]
                score = 1.0 - float(dist)
                results.append((score, meta))
        return results

    def save(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH) -> None:
        if HAS_FAISS:
            assert self.index is not None
            faiss.write_index(self.index, str(index_path))
        else:
            # Ensure fitted and vectors are persisted for sklearn backend
            self.finalize()
            if self._vectors is None:
                self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            np.save(str(VECTORS_PATH), self._vectors)
        serializable = [
            {
                "text": m.text,
                "source": m.source,
                "chunk_id": m.chunk_id,
                "document_id": m.document_id,
            }
            for m in self.metadatas
        ]
        write_json(
            metadata_path,
            {"dim": self.dim, "metadatas": serializable, "backend": "faiss" if HAS_FAISS else "sklearn"},
        )

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH) -> "VectorStore":
        if not metadata_path.exists():
            raise FileNotFoundError("Vector store not found. Please run ingestion first.")
        meta = read_json(metadata_path, default=None)
        if not meta or "dim" not in meta:
            raise ValueError("Invalid metadata file")
        dim = int(meta["dim"])  # type: ignore
        store = cls(dim)
        store.metadatas = [
            VectorMetadata(
                text=item["text"],
                source=item["source"],
                chunk_id=int(item["chunk_id"]),
                document_id=str(item["document_id"]),
            )
            for item in meta.get("metadatas", [])
        ]
        backend = meta.get("backend", "faiss")
        if HAS_FAISS and backend == "faiss":
            if not index_path.exists():
                raise FileNotFoundError("FAISS index missing. Re-ingest.")
            store.index = faiss.read_index(str(index_path))
        else:
            # sklearn backend
            if not VECTORS_PATH.exists():
                raise FileNotFoundError("Vectors file missing. Re-ingest.")
            store._vectors = np.load(str(VECTORS_PATH)).astype("float32")
            if store._vectors.size > 0:
                n_neighbors = min(10, store._vectors.shape[0])
                store.nn.set_params(n_neighbors=n_neighbors)
                store.nn.fit(store._vectors)
                store._is_fitted = True
        return store

