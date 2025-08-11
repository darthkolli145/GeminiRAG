from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import (
    DOCS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    INGEST_MAX_WORKERS,
    INGEST_SHOW_PROGRESS,
)
from .embeddings import EmbeddingModel
from .utils import chunk_text, load_text_from_file
from .vectordb import VectorMetadata, VectorStore


def compute_document_id(file_path: Path, content: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(file_path).encode("utf-8"))
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _prepare_file(file_path: Path) -> Tuple[str, str, List[str]]:
    content = load_text_from_file(file_path)
    document_id = compute_document_id(file_path, content)
    chunks = chunk_text(content)
    return str(file_path), document_id, chunks


def ingest_path(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files: List[Path] = []
    if path.is_file():
        files = [path]
    else:
        for f in path.rglob("*"):
            if f.suffix.lower() in {".txt", ".md", ".markdown", ".pdf"}:
                files.append(f)

    if not files:
        print("No supported documents found to ingest.")
        return 0

    embedder = EmbeddingModel()

    # Try to load an existing store
    try:
        store = VectorStore.load(INDEX_PATH, METADATA_PATH)
        print(f"Loaded existing vector store with {store.size} vectors")
    except Exception:
        test_vec = embedder.embed_texts(["test"])
        dim = int(test_vec.shape[1])
        store = VectorStore(dim=dim)
        print("Created new vector store")

    total_new_chunks = 0
    progress_iter = None
    if INGEST_SHOW_PROGRESS:
        try:
            from tqdm import tqdm  # type: ignore

            progress_iter = tqdm(total=len(files), desc="Ingesting files", unit="file")
        except Exception:
            progress_iter = None

    # Parallel file parsing and chunking
    results: List[Tuple[str, str, List[str]]] = []
    with ProcessPoolExecutor(max_workers=max(1, INGEST_MAX_WORKERS)) as ex:
        future_to_path = {ex.submit(_prepare_file, f): f for f in files}
        for fut in as_completed(future_to_path):
            f = future_to_path[fut]
            try:
                file_path_str, document_id, chunks = fut.result()
                results.append((file_path_str, document_id, chunks))
            except Exception as e:
                print(f"Skipping {f}: {e}")
            finally:
                if progress_iter is not None:
                    progress_iter.update(1)
    if progress_iter is not None:
        progress_iter.close()

    # Embed and add to store with a chunk-level progress bar
    chunk_progress = None
    total_chunks = sum(len(chs) for _, _, chs in results)
    if INGEST_SHOW_PROGRESS and total_chunks > 0:
        try:
            from tqdm import tqdm  # type: ignore

            chunk_progress = tqdm(total=total_chunks, desc="Embedding chunks", unit="chunk")
        except Exception:
            chunk_progress = None

    for file_path_str, document_id, chunks in results:
        if not chunks:
            continue
        vectors = embedder.embed_texts(chunks)
        metadatas = [
            VectorMetadata(
                text=chunk,
                source=file_path_str,
                chunk_id=i,
                document_id=document_id,
            )
            for i, chunk in enumerate(chunks)
        ]
        store.add(vectors, metadatas)
        total_new_chunks += len(chunks)
        print(f"Ingested {len(chunks)} chunks from {file_path_str}")
        if chunk_progress is not None:
            chunk_progress.update(len(chunks))
    if chunk_progress is not None:
        chunk_progress.close()

    # Perform a single fit/save at the end to avoid repeated refits
    try:
        store.finalize()
    except AttributeError:
        # Backward compatibility if finalize doesn't exist
        pass
    store.save(INDEX_PATH, METADATA_PATH)
    print(f"Saved vector store: {store.size} total vectors")
    return total_new_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--path",
        type=str,
        default=str(DOCS_DIR),
        help="Path to a file or directory containing documents (txt, md, pdf)",
    )
    args = parser.parse_args()
    path = Path(args.path)
    added = ingest_path(path)
    print(f"Done. Added {added} chunks.")


if __name__ == "__main__":
    main()

