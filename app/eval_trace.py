"""Opt-in JSONL trace export for controlled integration evaluation only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def emit(case_id: str | None, trace: dict[str, Any], final_answer: str) -> None:
    path = os.environ.get("AI_EVAL_TRACE_FILE")
    if not path or not case_id:
        return
    record = {"case_id": case_id, "final_answer": final_answer, "execution_kind": "real_integration", **trace}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
