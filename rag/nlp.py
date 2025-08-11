from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import json
import math
import re
from collections import Counter, defaultdict

from .config import (
    NLP_METADATA_DIR,
    NLP_LOWERCASE,
    NLP_REMOVE_STOPWORDS,
    NLP_SUMMARIZE,
    NLP_SUMMARY_SENTENCES,
    NLP_EXTRACT_KEYWORDS,
    NLP_KEYWORDS_TOP_N,
    NLP_LANGUAGE_DETECT,
)


STOPWORDS = set(
    """
    a an and are as at be by for from has he in is it its of on that the to was were will with this those these you your we our they them their i me my mine
    """.split()
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _split_sentences(text: str) -> List[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s]


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _preprocess(text: str) -> str:
    processed = text
    if NLP_LOWERCASE:
        processed = processed.lower()
    return processed


def _remove_stopwords(tokens: List[str]) -> List[str]:
    if not NLP_REMOVE_STOPWORDS:
        return tokens
    return [t for t in tokens if t not in STOPWORDS]


def _tfidf_keywords(doc_tokens: List[str], corpus_counts: Counter, corpus_docs: int, top_n: int) -> List[Tuple[str, float]]:
    doc_count = Counter(doc_tokens)
    scores: Dict[str, float] = {}
    for term, tf in doc_count.items():
        df = max(1, corpus_counts.get(term, 1))
        idf = math.log((corpus_docs + 1) / df)
        scores[term] = float(tf) * idf
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _lead_n_sentence_summary(text: str, n: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    return " ".join(sentences[: max(1, n)])


def _simple_lang_detect(text: str) -> str:
    # Naive detector: checks presence of non-ASCII as proxy
    try:
        text.encode("ascii")
        return "en"
    except Exception:
        return "unknown"


@dataclass
class NlpArtifacts:
    language: str | None
    summary: str | None
    keywords: List[Tuple[str, float]]


def run_nlp_pipeline(all_texts: List[str]) -> List[NlpArtifacts]:
    # Build naive corpus counts for TF-IDF
    corpus_counts: Counter = Counter()
    documents_tokens: List[List[str]] = []
    for text in all_texts:
        processed = _preprocess(text)
        tokens = _tokenize(processed)
        tokens = _remove_stopwords(tokens)
        documents_tokens.append(tokens)
        corpus_counts.update(set(tokens))

    artifacts: List[NlpArtifacts] = []
    corpus_docs = len(all_texts)
    for idx, text in enumerate(all_texts):
        language = _simple_lang_detect(text) if NLP_LANGUAGE_DETECT else None
        summary = _lead_n_sentence_summary(text, NLP_SUMMARY_SENTENCES) if NLP_SUMMARIZE else None
        keywords: List[Tuple[str, float]] = []
        if NLP_EXTRACT_KEYWORDS:
            keywords = _tfidf_keywords(documents_tokens[idx], corpus_counts, corpus_docs, NLP_KEYWORDS_TOP_N)
        artifacts.append(NlpArtifacts(language=language, summary=summary, keywords=keywords))
    return artifacts


def save_nlp_metadata(document_id: str, chunk_artifacts: List[NlpArtifacts]) -> Path:
    data = {
        "chunks": [
            {
                "language": a.language,
                "summary": a.summary,
                "keywords": a.keywords,
            }
            for a in chunk_artifacts
        ]
    }
    out_path = NLP_METADATA_DIR / f"{document_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


