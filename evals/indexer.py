"""Index a frozen, reviewed corpus into the knowledge collection."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.retrieval import QdrantRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpora" / "development-v1.json")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    documents = corpus.get("documents", [])
    required = {"source_id", "text", "metadata"}
    if not all(required <= document.keys() for document in documents):
        parser.error("every document must have source_id, text, and metadata")
    count = asyncio.run(QdrantRetriever("knowledge_chunks").upsert(
        (document["source_id"], document["text"], {**document["metadata"], "corpus_id": corpus["corpus_id"]})
        for document in documents
    ))
    print(f"indexed {count} documents from {corpus['corpus_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
