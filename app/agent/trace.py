"""Per-request execution evidence used by the agent and eval runner."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    success: bool
    latency_ms: float
    result_chars: int


@dataclass(slots=True)
class AgentTrace:
    started_at: float = field(default_factory=time.perf_counter)
    model_rounds: int = 0
    tool_calls: list[ToolTrace] = field(default_factory=list)
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None

    def add_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage")
        if not isinstance(usage, dict):
            return
        self.provider_input_tokens = _number(usage, "input_tokens", "prompt_tokens")
        self.provider_output_tokens = _number(usage, "output_tokens", "completion_tokens")

    def finish(self) -> dict[str, Any]:
        return {"model_rounds": self.model_rounds, "tool_calls": [asdict(item) for item in self.tool_calls], "latency_ms": round((time.perf_counter() - self.started_at) * 1000, 3), "provider_input_tokens": self.provider_input_tokens, "provider_output_tokens": self.provider_output_tokens}


def _number(value: dict[str, Any], *names: str) -> int | None:
    for name in names:
        candidate = value.get(name)
        if isinstance(candidate, int):
            return candidate
    return None
