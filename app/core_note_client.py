import logging
import os

import httpx

logger = logging.getLogger(__name__)


class CoreNoteClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("CORE_INTERNAL_BASE_URL", "http://localhost:8080").rstrip("/")
        self.timeout = float(os.environ.get("CORE_INTERNAL_TIMEOUT_SECONDS", "5"))

    async def create(self, title: str, content_md: str, authorization: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/learning/notes",
                    json={"title": title, "contentMd": content_md},
                    headers={"Authorization": authorization, "Content-Type": "application/json"},
                )
            if not response.is_success:
                logger.warning("Core note persistence failed: status=%s body=%s", response.status_code, response.text[:500])
                raise RuntimeError(f"Core note persistence failed with HTTP {response.status_code}")
            return True
        except httpx.HTTPError as exc:
            logger.warning("Core note persistence unavailable: %s", exc)
            raise RuntimeError("Core note persistence unavailable") from exc
