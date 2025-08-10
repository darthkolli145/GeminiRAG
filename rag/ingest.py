from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import numpy as np

from .config import DOCS_DIR, INDEX_PATH, METADATA_PATH
from .embeddings import EmbeddingModel
from .utils import chunk_text, load_text_from_file
from .vectordb import VectorMetadata, VectorStore


def compute_document_id(file_path: Path, content: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(file_path).encode("utf-8"))
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()[:16]


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
    for file_path in files:
        try:
            content = load_text_from_file(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        document_id = compute_document_id(file_path, content)
        chunks = chunk_text(content)
        if not chunks:
            continue
        vectors = embedder.embed_texts(chunks)
        metadatas = [
            VectorMetadata(
                text=chunk,
                source=str(file_path),
                chunk_id=i,
                document_id=document_id,
            )
            for i, chunk in enumerate(chunks)
        ]
        store.add(vectors, metadatas)
        total_new_chunks += len(chunks)
        print(f"Ingested {len(chunks)} chunks from {file_path}")

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

