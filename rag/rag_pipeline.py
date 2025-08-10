from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import DEFAULT_TOP_K
from .llm import LLM
from .retriever import Retriever


@dataclass
class SourceChunk:
    text: str
    source: str
    score: float
    chunk_id: int


@dataclass
class RagAnswer:
    answer: str
    sources: List[SourceChunk]


def build_prompt(question: str, sources: List[SourceChunk]) -> str:
    context = "\n\n".join(
        [f"[Source: {s.source} | chunk {s.chunk_id} | score {s.score:.3f}]\n{s.text}" for s in sources]
    )
    prompt = (
        f"Answer the question using the context.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> RagAnswer:
    retriever = Retriever()
    llm = LLM()
    results = retriever.retrieve(question, top_k=top_k)
    sources = [
        SourceChunk(
            text=m.text,
            source=m.source,
            score=score,
            chunk_id=m.chunk_id,
        )
        for score, m in results
    ]
    prompt = build_prompt(question, sources)
    ans = llm.answer(prompt)
    return RagAnswer(answer=ans, sources=sources)

