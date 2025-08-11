import os
from pathlib import Path


# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_DB_DIR = PROJECT_ROOT / "vectordb"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Vector store paths
INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
METADATA_PATH = VECTOR_DB_DIR / "metadata.json"
VECTORS_PATH = VECTOR_DB_DIR / "vectors.npy"  # used when FAISS is unavailable

# Embeddings
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_SHOW_PROGRESS = os.environ.get("EMBEDDING_SHOW_PROGRESS", "0") == "1"

# Chunking
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "120"))

# LLM
LLM_BACKEND = os.environ.get("LLM_BACKEND", "transformers")  # transformers|openai|extractive

# Local transformers model (no API)
LOCAL_LLM_MODEL_ID = os.environ.get(
    "LOCAL_LLM_MODEL_ID",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)
LOCAL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", "256"))
LOCAL_TEMPERATURE = float(os.environ.get("LOCAL_TEMPERATURE", "0.2"))

# OpenAI (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # Optional for Azure/Proxies

# Retrieval
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "4"))

# Reranking
RERANK_STRATEGY = os.environ.get("RERANK_STRATEGY", "maxsim")  # maxsim|none
RERANK_CANDIDATE_MULTIPLIER = int(os.environ.get("RERANK_CANDIDATE_MULTIPLIER", "5"))

# Ingestion parallelism/progress
INGEST_MAX_WORKERS = int(os.environ.get("INGEST_MAX_WORKERS", "4"))
INGEST_SHOW_PROGRESS = os.environ.get("INGEST_SHOW_PROGRESS", "1") == "1"

