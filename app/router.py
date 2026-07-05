import json
import logging
from functools import lru_cache

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .llm_client import LLMClient
from .prompt import build_system_prompt
from .schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@lru_cache(maxsize=1)
def _get_llm_client() -> LLMClient:
    return LLMClient()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ai/chat")
async def chat(request: ChatRequest):
    system_prompt = build_system_prompt(request.context)

    async def event_stream():
        try:
            llm_client = _get_llm_client()
            async for token in llm_client.stream_chat(
                system_prompt, request.history, request.message
            ):
                yield _sse({"type": "token", "content": token})
            yield _sse({"type": "done"})
        except Exception as exc:
            logger.exception("LLM streaming error")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/ai/health")
async def ai_health():
    try:
        _get_llm_client()
        return {"status": "ready"}
    except Exception as exc:
        return {"status": "not_ready", "reason": str(exc)}
