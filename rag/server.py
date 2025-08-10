from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import DEFAULT_TOP_K
from .ingest import ingest_path
from .rag_pipeline import answer_question


class IngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceResponse(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]


app = FastAPI(title="Minimal RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
def ingest(req: IngestRequest):
    path = Path(req.path)
    try:
        added = ingest_path(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"added": added}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or DEFAULT_TOP_K
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    result = answer_question(req.question, top_k=top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                text=s.text, source=s.source, score=s.score, chunk_id=s.chunk_id
            )
            for s in result.sources
        ],
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG server is running. POST /ingest and /query"}

