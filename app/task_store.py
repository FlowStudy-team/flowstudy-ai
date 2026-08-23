import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .schemas import AiContext


@dataclass(slots=True)
class StoredTask:
    task_id: str
    status: str
    result: str | None
    error: str | None
    context: AiContext
    authorization: str | None


class NoteTaskStore:
    """Small durable task store used by both local and RabbitMQ executors."""

    def __init__(self) -> None:
        configured = os.environ.get("AI_TASK_DB_PATH", ".data/ai_tasks.sqlite3")
        self.path = Path(configured)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_note_task (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    context_json TEXT NOT NULL,
                    authorization_hash TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _auth_hash(authorization: str | None) -> str | None:
        if not authorization:
            return None
        return hashlib.sha256(authorization.encode("utf-8")).hexdigest()

    def create(self, task_id: str, context: AiContext, authorization: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ai_note_task(task_id,status,context_json,authorization_hash) VALUES(?,?,?,?)",
                (task_id, "PENDING", json.dumps(context.model_dump(), ensure_ascii=False), self._auth_hash(authorization)),
            )

    def update(self, task_id: str, status: str, result: str | None = None, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE ai_note_task SET status=?, result=?, error=?, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                (status, result, error, task_id),
            )

    def get(self, task_id: str, authorization: str | None = None) -> StoredTask | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM ai_note_task WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        if row["authorization_hash"] and row["authorization_hash"] != self._auth_hash(authorization):
            return None
        return StoredTask(
            task_id=row["task_id"],
            status=row["status"],
            result=row["result"],
            error=row["error"],
            context=AiContext.model_validate(json.loads(row["context_json"])),
            authorization=authorization,
        )
