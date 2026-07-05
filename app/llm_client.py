import logging
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .schemas import ChatMessage

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
        )
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    async def stream_chat(
        self,
        system_prompt: str,
        history: list[ChatMessage],
        user_message: str,
    ) -> AsyncGenerator[str, None]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})

        logger.info("Sending %d messages to DeepSeek model=%s", len(messages), self.model)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
