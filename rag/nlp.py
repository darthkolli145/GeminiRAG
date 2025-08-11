from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import json
import math
import re
from collections import Counter

from .config import (
    NLP_METADATA_DIR,
    NLP_LOWERCASE,
    NLP_REMOVE_STOPWORDS,
    NLP_SUMMARIZE,
    NLP_SUMMARY_SENTENCES,
    NLP_EXTRACT_KEYWORDS,
    NLP_KEYWORDS_TOP_N,
    NLP_LANGUAGE_DETECT,
    NLP_KEYWORDS_BACKEND,
    NLP_KEYWORDS_NGRAM_MAX,
    NLP_KEYWORDS_MAX_FEATURES,
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


def _tfidf_keywords_naive(doc_tokens: List[str], corpus_counts: Counter, corpus_docs: int, top_n: int) -> List[Tuple[str, float]]:
    doc_count = Counter(doc_tokens)
    scores: Dict[str, float] = {}
    for term, tf in doc_count.items():
        df = max(1, corpus_counts.get(term, 1))
        idf = math.log((corpus_docs + 1) / df)
        scores[term] = float(tf) * idf
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _build_sklearn_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    ngram_range = (1, max(1, int(NLP_KEYWORDS_NGRAM_MAX)))
    vectorizer = TfidfVectorizer(
        lowercase=NLP_LOWERCASE,
        max_features=max(1000, int(NLP_KEYWORDS_MAX_FEATURES)),
        ngram_range=ngram_range,
        token_pattern=r"[A-Za-z0-9_]+",
        stop_words=list(STOPWORDS) if NLP_REMOVE_STOPWORDS else None,
    )
    return vectorizer


def _tfidf_keywords_sklearn(texts: List[str], top_n: int) -> List[List[Tuple[str, float]]]:
    if not texts:
        return []
    vectorizer = _build_sklearn_vectorizer()
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    results: List[List[Tuple[str, float]]] = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i)
        coo = row.tocoo()
        pairs = [(feature_names[j], float(v)) for j, v in zip(coo.col, coo.data)]
        topk = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
        results.append(topk)
    return results


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
    # Preprocess once
    preprocessed_texts = [_preprocess(t) for t in all_texts]

    # Summaries and language (fast, streaming)
    summaries = [
        _lead_n_sentence_summary(t, NLP_SUMMARY_SENTENCES) if NLP_SUMMARIZE else None
        for t in all_texts
    ]
    languages = [_simple_lang_detect(t) if NLP_LANGUAGE_DETECT else None for t in all_texts]

    # Keywords: choose backend
    keywords_per_text: List[List[Tuple[str, float]]] = [[] for _ in preprocessed_texts]
    if NLP_EXTRACT_KEYWORDS and preprocessed_texts:
        if NLP_KEYWORDS_BACKEND == "sklearn":
            keywords_per_text = _tfidf_keywords_sklearn(preprocessed_texts, NLP_KEYWORDS_TOP_N)
        else:
            corpus_counts: Counter = Counter()
            documents_tokens: List[List[str]] = []
            for text in preprocessed_texts:
                tokens = _remove_stopwords(_tokenize(text))
                documents_tokens.append(tokens)
                corpus_counts.update(set(tokens))
            corpus_docs = len(preprocessed_texts)
            keywords_per_text = [
                _tfidf_keywords_naive(documents_tokens[i], corpus_counts, corpus_docs, NLP_KEYWORDS_TOP_N)
                for i in range(len(preprocessed_texts))
            ]

    artifacts: List[NlpArtifacts] = []
    for i in range(len(all_texts)):
        artifacts.append(
            NlpArtifacts(language=languages[i], summary=summaries[i], keywords=keywords_per_text[i])
        )
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


