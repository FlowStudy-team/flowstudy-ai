"""Cheap context protection for request-scoped agent histories."""

from typing import Any


def compact_messages(messages: list[Any], max_messages: int = 20, max_tool_chars: int = 4000) -> list[Any]:
    """Return a bounded history without splitting assistant/tool call pairs.

    This is the first mechanical layer from CoreCoder's context strategy. A
    future summarizer can be added behind this function without changing the
    chat service contract.
    """
    copied = list(messages)
    for message in copied:
        if getattr(message, "type", None) != "tool":
            continue
        content = str(getattr(message, "content", "") or "")
        if len(content) > max_tool_chars:
            message.content = content[:max_tool_chars // 2] + "\n... tool output compacted ...\n" + content[-max_tool_chars // 2:]

    if len(copied) <= max_messages:
        return copied
    start = len(copied) - max_messages
    while start > 0 and getattr(copied[start], "type", None) == "tool":
        start -= 1
    return copied[start:]
