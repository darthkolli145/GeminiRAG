from __future__ import annotations

import argparse
import json

from .config import DEFAULT_TOP_K
from .rag_pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against your ingested docs")
    parser.add_argument("--query", required=True, type=str, help="The question to ask")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    result = answer_question(args.query, top_k=args.top_k)
    print("\n=== Answer ===\n")
    print(result.answer)
    print("\n=== Sources ===\n")
    for s in result.sources:
        print(f"- {s.source} (chunk {s.chunk_id}, score={s.score:.3f})")


if __name__ == "__main__":
    main()

