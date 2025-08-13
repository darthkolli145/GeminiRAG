from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors  # type: ignore
from .config import INDEX_PATH, METADATA_PATH, VECTORS_PATH, FAISS_USE_GPU

try:
    import torch  # type: ignore

    HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    HAS_TORCH_CUDA = False

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
        self._use_torch_gpu: bool = False
        self._torch_vectors = None  # type: ignore
        # Move FAISS index to GPU if requested and available
        if HAS_FAISS and self.index is not None and FAISS_USE_GPU:
            try:
                import faiss.contrib.torch_utils as _  # type: ignore
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                pass
        # If FAISS is not available, optionally use a Torch GPU fallback for similarity search
        if not HAS_FAISS and FAISS_USE_GPU and HAS_TORCH_CUDA:
            self._use_torch_gpu = True

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
            if self._use_torch_gpu and HAS_TORCH_CUDA:
                # Keep a GPU copy for fast search
                tv = torch.from_numpy(vectors).to("cuda")  # type: ignore[attr-defined]
                if self._torch_vectors is None:
                    self._torch_vectors = tv
                else:
                    self._torch_vectors = torch.vstack([self._torch_vectors, tv])
        self.metadatas.extend(metadatas)

    def finalize(self) -> None:
        if HAS_FAISS:
            return
        if self._use_torch_gpu and HAS_TORCH_CUDA:
            # Nothing to fit for brute-force matmul
            self._is_fitted = True
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
            if self._use_torch_gpu and HAS_TORCH_CUDA:
                if self._torch_vectors is None:
                    # build GPU tensor on demand
                    self._torch_vectors = torch.from_numpy(self._vectors).to("cuda")  # type: ignore[attr-defined]
                q = torch.from_numpy(vector).to("cuda")  # type: ignore[attr-defined]
                sims = torch.matmul(self._torch_vectors, q.T).squeeze(1)
                k = min(top_k, sims.shape[0])
                values, indices = torch.topk(sims, k=k, largest=True, sorted=True)
                for score, idx in zip(values.tolist(), indices.tolist()):
                    meta = self.metadatas[int(idx)]
                    results.append((float(score), meta))
            else:
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
            # Ensure we write a CPU index; FAISS GPU indices can't be saved directly
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)  # type: ignore[attr-defined]
            except Exception:
                cpu_index = self.index
            faiss.write_index(cpu_index, str(index_path))
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
        backend = "faiss" if HAS_FAISS else ("torch" if self._use_torch_gpu and HAS_TORCH_CUDA else "sklearn")
        write_json(
            metadata_path,
            {"dim": self.dim, "metadatas": serializable, "backend": backend},
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
            idx = faiss.read_index(str(index_path))
            if FAISS_USE_GPU:
                try:
                    res = faiss.StandardGpuResources()
                    idx = faiss.index_cpu_to_gpu(res, 0, idx)
                except Exception:
                    pass
            store.index = idx
        elif backend == "torch" and HAS_TORCH_CUDA and FAISS_USE_GPU:
            # Torch GPU backend
            if not VECTORS_PATH.exists():
                raise FileNotFoundError("Vectors file missing. Re-ingest.")
            store._vectors = np.load(str(VECTORS_PATH), mmap_mode="r").astype("float32", copy=False)
            if store._vectors.size > 0:
                store._torch_vectors = torch.from_numpy(store._vectors).to("cuda")  # type: ignore[attr-defined]
                store._use_torch_gpu = True
                store._is_fitted = True
        else:
            # sklearn backend
            if not VECTORS_PATH.exists():
                raise FileNotFoundError("Vectors file missing. Re-ingest.")
            store._vectors = np.load(str(VECTORS_PATH), mmap_mode="r").astype("float32", copy=False)
            if store._vectors.size > 0:
                n_neighbors = min(10, store._vectors.shape[0])
                store.nn.set_params(n_neighbors=n_neighbors)
                store.nn.fit(store._vectors)
                store._is_fitted = True
        return store

