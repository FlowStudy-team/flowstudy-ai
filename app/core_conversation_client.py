import logging
import os
from typing import Any

import httpx

from .schemas import AiContext, ChatMessage

logger = logging.getLogger(__name__)


class CoreConversationClient:
    """Small adapter for Core-owned conversation persistence.

    The user's bearer token is forwarded so Core remains the authority for
    conversation ownership and authorization. Persistence failures are logged
    and do not make the model unavailable.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get("CORE_INTERNAL_BASE_URL", "http://localhost:8080").rstrip("/")
        self.timeout = float(os.environ.get("CORE_INTERNAL_TIMEOUT_SECONDS", "5"))

    def _headers(self, authorization: str | None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        return headers

    async def create(self, context: AiContext, authorization: str) -> str | None:
        payload = {
            "title": context.problemTitle or context.blogTitle or context.tutorialTitle or "AI 学习对话",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/ai/conversations",
                    json=payload,
                    headers=self._headers(authorization),
                )
        except httpx.HTTPError as exc:
            logger.warning("Core conversation creation unavailable: %s", exc)
            return None
        if response.is_success:
            data = response.json().get("data") or {}
            return str(data["id"]) if data.get("id") is not None else None
        logger.warning("Core conversation creation failed: status=%s body=%s", response.status_code, response.text[:500])
        return None

    async def messages(self, conversation_id: str, authorization: str) -> list[ChatMessage]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/ai/conversations/{conversation_id}/messages",
                    headers=self._headers(authorization),
                )
        except httpx.HTTPError as exc:
            logger.warning("Core conversation history unavailable: %s", exc)
            return []
        if not response.is_success:
            logger.warning("Core conversation history failed: status=%s", response.status_code)
            return []
        data: list[dict[str, Any]] = response.json().get("data") or []
        return [
            ChatMessage(role=item["role"], content=item["content"])
            for item in data
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

    async def append(
        self,
        conversation_id: str,
        message: ChatMessage,
        authorization: str,
        model_name: str | None = None,
    ) -> None:
        payload = {"role": message.role, "content": message.content, "modelName": model_name}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/ai/conversations/{conversation_id}/messages",
                    json=payload,
                    headers=self._headers(authorization),
                )
        except httpx.HTTPError as exc:
            logger.warning("Core message persistence unavailable: %s", exc)
            return
        if not response.is_success:
            logger.warning("Core message persistence failed: status=%s body=%s", response.status_code, response.text[:500])
