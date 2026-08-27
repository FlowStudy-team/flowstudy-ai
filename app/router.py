import json
import logging
import os
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from .langchain_service import LangChainAIService
from .schemas import ChatRequest, NoteGenerationRequest, ProfileAnalysisRequest
from .task_service import NoteTaskService, to_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@lru_cache(maxsize=1)
def _get_ai_service() -> LangChainAIService:
    return LangChainAIService()


@lru_cache(maxsize=1)
def _get_note_tasks() -> NoteTaskService:
    return NoteTaskService(_get_ai_service())


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ai/chat")
async def chat(
        request: ChatRequest,
        authorization: str | None = Header(default=None),
        x_eval_case_id: str | None = Header(default=None)):
    async def event_stream():
        try:
            ai_service = _get_ai_service()
            async for event in ai_service.stream_chat(
                request.message,
                request.history,
                request.context,
                request.conversationId,
                authorization,
                x_eval_case_id,
            ):
                payload = {"type": event.type}
                if event.content is not None:
                    payload["content"] = event.content
                if event.name is not None:
                    payload["name"] = event.name
                if event.conversation_id is not None:
                    payload["conversationId"] = event.conversation_id
                yield _sse(payload)
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
        _get_ai_service()
        return {"status": "ready"}
    except Exception as exc:
        return {"status": "not_ready", "reason": str(exc)}


@router.post("/ai/profile/analyze")
async def analyze_profile(
        request: ProfileAnalysisRequest,
        x_internal_token: str | None = Header(default=None)):
    expected_token = os.environ.get("AI_INTERNAL_TOKEN", "")
    if expected_token and x_internal_token != expected_token:
        raise HTTPException(status_code=401, detail="internal authentication required")
    analysis = await _get_ai_service().analyze_learning_profile(request.events)
    return analysis


@router.post("/ai/notes/generate")
async def generate_note(
        request: NoteGenerationRequest,
        authorization: str | None = Header(default=None)):
    task = await _get_note_tasks().submit(request.context, authorization)
    return to_response(task)


@router.get("/ai/notes/tasks/{task_id}")
async def get_note_task(task_id: str, authorization: str | None = Header(default=None)):
    task = _get_note_tasks().get(task_id, authorization)
    if task is None:
        return {"status": "not_found", "taskId": task_id}
    return to_response(task)
