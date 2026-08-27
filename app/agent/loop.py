"""A small async tool-calling loop inspired by CoreCoder.

The loop owns orchestration only. It does not know about HTTP, users, files,
shell commands, or persistence; those concerns stay in FlowStudy adapters.
"""

import uuid
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from .trace import AgentTrace, ToolTrace


@dataclass(slots=True)
class AgentEvent:
    type: str
    content: str | None = None
    name: str | None = None


class AgentLoop:
    def __init__(self, max_rounds: int = 3) -> None:
        self.max_rounds = max(1, max_rounds)

    async def run(
        self,
        messages: list[Any],
        model: Any,
        tools: Sequence[Any],
        trace: AgentTrace | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run model -> tools -> model until a final answer is produced."""
        tool_by_name = {tool.name: tool for tool in tools}
        for _ in range(self.max_rounds):
            response = await model.ainvoke(messages)
            if trace is not None:
                trace.model_rounds += 1
                trace.add_usage(response)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                yield AgentEvent(type="answer", content=_message_text(response))
                return

            for tool_call in tool_calls:
                name = str(tool_call.get("name", "unknown"))
                tool = tool_by_name.get(name)
                if tool is None:
                    result = f"Unknown tool: {name}"
                    success = False
                    elapsed_ms = 0.0
                else:
                    started_at = time.perf_counter()
                    try:
                        result = str(await tool.ainvoke(tool_call.get("args", {})))
                        success = True
                    except Exception as exc:  # tool failures are observations for the model
                        result = f"Tool {name} failed: {exc}"
                        success = False
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                if trace is not None:
                    trace.tool_calls.append(
                        ToolTrace(name, dict(tool_call.get("args", {})), success, elapsed_ms, len(result))
                    )
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call.get("id", str(uuid.uuid4())),
                    )
                )
                yield AgentEvent(type="tool", name=name, content=result)

        yield AgentEvent(
            type="answer",
            content="I could not finish the requested operation within the tool-call limit.",
        )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content or "")
