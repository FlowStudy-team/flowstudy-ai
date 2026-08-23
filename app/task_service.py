import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Literal
from typing import Any

try:
    import aio_pika
except ImportError:  # Local mode does not require the optional MQ dependency.
    aio_pika = None

from .core_note_client import CoreNoteClient
from .langchain_service import LangChainAIService
from .schemas import AiContext, NoteTaskResponse
from .task_store import NoteTaskStore, StoredTask

logger = logging.getLogger(__name__)
TaskStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]


@dataclass(slots=True)
class NoteTask:
    task_id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    authorization: str | None = None


class NoteTaskService:
    """Durable note task facade with local and RabbitMQ executors."""

    def __init__(self, ai_service: LangChainAIService) -> None:
        self.ai_service = ai_service
        self.core_notes = CoreNoteClient()
        self.store = NoteTaskStore()
        self.executor = os.environ.get("AI_TASK_EXECUTOR", "local").lower()
        self.queue_name = os.environ.get("AI_NOTE_QUEUE", "fs.ai.note.queue")
        self.rabbit_url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self._connection: Any = None
        self._consumer_tag: str | None = None
        self._queue: Any = None

    async def start(self) -> None:
        if self.executor != "rabbitmq" or self._consumer_tag is not None:
            return
        if aio_pika is None:
            raise RuntimeError("RabbitMQ executor requires aio-pika; install requirements.txt")
        self._connection = await aio_pika.connect_robust(self.rabbit_url)
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=int(os.environ.get("AI_NOTE_PREFETCH", "2")))
        queue = await channel.declare_queue(self.queue_name, durable=True)
        self._queue = queue
        self._consumer_tag = await queue.consume(self._on_message)
        logger.info("RabbitMQ note executor started on queue %s", self.queue_name)

    async def stop(self) -> None:
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
        self._consumer_tag = None
        self._queue = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def submit(self, context: AiContext, authorization: str | None = None) -> NoteTask:
        task_id = f"note-{uuid.uuid4().hex}"
        self.store.create(task_id, context, authorization)
        if self.executor == "rabbitmq":
            await self.start()
            if self._connection is None:
                raise RuntimeError("RabbitMQ note executor is unavailable")
            channel = await self._connection.channel()
            payload = {
                "schemaVersion": "1.0",
                "eventType": "ai.note.generate.requested",
                "taskId": task_id,
                "context": context.model_dump(),
                "authorization": authorization,
            }
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    message_id=task_id,
                ),
                routing_key=self.queue_name,
            )
        else:
            asyncio.create_task(self._run(task_id, context, authorization))
        return self._to_task(self.store.get(task_id, authorization))

    def get(self, task_id: str, authorization: str | None = None) -> NoteTask | None:
        stored = self.store.get(task_id, authorization)
        return self._to_task(stored) if stored else None

    async def _on_message(self, message: Any) -> None:
        async with message.process(requeue=True):
            payload = json.loads(message.body)
            task_id = payload["taskId"]
            context = AiContext.model_validate(payload.get("context") or {})
            await self._run(task_id, context, payload.get("authorization"))

    async def _run(self, task_id: str, context: AiContext, authorization: str | None) -> None:
        self.store.update(task_id, "RUNNING")
        try:
            result = await self.ai_service.generate_learning_note(context)
            if authorization:
                await self.core_notes.create("AI 学习笔记", result, authorization)
            self.store.update(task_id, "SUCCEEDED", result=result)
        except Exception as exc:  # pragma: no cover - integration path
            logger.exception("AI note task failed: %s", task_id)
            self.store.update(task_id, "FAILED", error=str(exc))

    @staticmethod
    def _to_task(stored: StoredTask | None) -> NoteTask:
        if stored is None:
            raise RuntimeError("task not found")
        return NoteTask(
            task_id=stored.task_id,
            status=stored.status,  # type: ignore[arg-type]
            result=stored.result,
            error=stored.error,
            authorization=stored.authorization,
        )


def to_response(task: NoteTask) -> NoteTaskResponse:
    return NoteTaskResponse(taskId=task.task_id, status=task.status, result=task.result, error=task.error)
