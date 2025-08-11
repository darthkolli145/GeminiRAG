from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List
import re

from pypdf import PdfReader

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text_parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    cleaned = "\n".join(line.strip() for line in text.splitlines())
    chunks: List[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + CHUNK_SIZE_CHARS, n)
        chunk = cleaned[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, 0)
    return chunks


_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str, max_sentences: int | None = None) -> List[str]:
    if not text:
        return []
    # Simple regex-based splitter; avoids heavy deps
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if max_sentences is not None and len(sentences) > max_sentences:
        return sentences[: max_sentences]
    return sentences


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

