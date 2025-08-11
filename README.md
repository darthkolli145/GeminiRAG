## Retrieval-Augmented Generation (RAG) – Minimal Starter

This project is a minimal, production-friendly RAG scaffold you can run locally. It ingests your documents, builds a FAISS/Sklearn vector index using `sentence-transformers` embeddings, retrieves relevant chunks for a query, and answers via:

- Local Hugging Face transformers model (default, no API key required)
- OpenAI Chat API (optional, if `OPENAI_API_KEY` is set)
- Extractive fallback

### Features
- Local embeddings with `all-MiniLM-L6-v2` (small, fast)
- FAISS vector store persisted to disk
- Simple chunking and metadata tracking for citations
- CLI and FastAPI server
- PDF, TXT, and Markdown support
- Optional NLP preprocessing: lowercasing, stopword removal, simple summaries, keyword extraction, and per-document metadata

### Quickstart

1) Create a virtual environment and install dependencies

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/pip.exe install -r requirements.txt
```

2) (Optional) Configure LLM backend

```powershell
# Default uses a small local model (no API):
$env:LLM_BACKEND = "transformers"
$env:LOCAL_LLM_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Or use OpenAI:
# $env:LLM_BACKEND = "openai"
# $env:OPENAI_API_KEY = "sk-..."
# $env:OPENAI_MODEL = "gpt-4o-mini"
```

3) Add documents to `docs/` and ingest them

```powershell
./.venv/Scripts/python.exe -m rag.ingest --path docs
```

Enable optional NLP pipeline during ingest:

```powershell
$env:NLP_ENABLE = "1"
$env:NLP_SUMMARIZE = "1"        # lead-3 sentence heuristic
$env:NLP_EXTRACT_KEYWORDS = "1"  # naive TF-IDF
$env:NLP_KEYWORDS_BACKEND = "sklearn"
$env:NLP_KEYWORDS_NGRAM_MAX = "2"
./.venv/Scripts/python.exe -m rag.ingest --path docs
```

4) Ask questions via CLI

```powershell
./.venv/Scripts/python.exe -m rag.cli --query "What is this project?" --top_k 4
```

5) Or run the API server

```powershell
./.venv/Scripts/python.exe -m uvicorn rag.server:app --reload --port 8000
```

Then query:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -Body (@{question='Explain the project'} | ConvertTo-Json) -ContentType 'application/json'
```

### Project Structure

```
rag/
  __init__.py
  config.py
  utils.py
  embeddings.py
  vectordb.py
  ingest.py
  retriever.py
  llm.py
  rag_pipeline.py
  cli.py
  server.py
docs/
  sample.txt
vectordb/
  ... (created after ingest)
```

### Notes
- If you do not set `OPENAI_API_KEY`, the system returns an extractive answer from the most relevant chunks.
- Supported file types for ingestion: `.txt`, `.md`, `.pdf`.
- You can safely clear the vector store by deleting the `vectordb/` directory.


