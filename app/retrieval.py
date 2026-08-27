"""Small, provider-neutral Qdrant retrieval adapters.

The service deliberately stores source text and IDs alongside vectors so every
answer and every eval assertion can be traced to a frozen corpus item.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    source_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class EmbeddingClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("AI_EMBEDDING_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("AI_EMBEDDING_API_KEY") or os.environ.get("AI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("AI_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    async def embed(self, text: str, *, query: bool = False) -> list[float]:
        if not self.base_url:
            raise RuntimeError("AI_EMBEDDING_BASE_URL is required for Qdrant retrieval")
        prefix = "query: " if query else "passage: "
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": prefix + text}, headers=headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]


class QdrantRetriever:
    def __init__(self, collection: str, embeddings: EmbeddingClient | None = None) -> None:
        self.collection = collection
        self.embeddings = embeddings or EmbeddingClient()
        self.url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")

    async def search(self, query: str, *, top_k: int = 3, filter_: dict[str, Any] | None = None) -> list[RetrievedItem]:
        vector = await self.embeddings.embed(query, query=True)
        payload: dict[str, Any] = {"vector": vector, "limit": min(max(top_k, 1), 10), "with_payload": True}
        if filter_:
            payload["filter"] = {"must": [{"key": key, "match": {"value": value}} for key, value in filter_.items()]}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.url}/collections/{self.collection}/points/search", json=payload)
            response.raise_for_status()
        return [
            RetrievedItem(str(item["payload"].get("source_id", item["id"])), str(item["payload"].get("text", "")), float(item["score"]), dict(item["payload"]))
            for item in response.json().get("result", [])
        ]

    async def upsert(self, items: Iterable[tuple[str, str, dict[str, Any]]]) -> int:
        points = []
        for source_id, text, metadata in items:
            point_id = hashlib.sha256(f"{self.collection}:{source_id}".encode()).hexdigest()[:32]
            points.append({"id": point_id, "vector": await self.embeddings.embed(text), "payload": {**metadata, "source_id": source_id, "text": text}})
        if not points:
            return 0
        async with httpx.AsyncClient(timeout=60) as client:
            existing = await client.get(f"{self.url}/collections/{self.collection}")
            if existing.status_code == 404:
                created = await client.put(
                    f"{self.url}/collections/{self.collection}",
                    json={"vectors": {"size": len(points[0]["vector"]), "distance": "Cosine"}},
                )
                created.raise_for_status()
            else:
                existing.raise_for_status()
            response = await client.put(f"{self.url}/collections/{self.collection}/points?wait=true", json={"points": points})
            response.raise_for_status()
        return len(points)


def render_hits(items: list[RetrievedItem]) -> str:
    if not items:
        return "No authorized source matched the query."
    return "\n\n".join(f"[{item.source_id}] {item.text}" for item in items)


def memory_scope(user_id: str | None, conversation_id: str) -> str:
    """Opaque stable user scope; anonymous conversations remain isolated."""
    material = f"user:{user_id}" if user_id else f"anonymous:{conversation_id}"
    return hashlib.sha256(material.encode()).hexdigest()
