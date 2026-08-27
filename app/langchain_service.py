import logging
import json
import os
import re
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .prompt import build_system_prompt
from .schemas import AiContext, ChatMessage, LearningEvent, ProfileAnalysisResponse
from .core_conversation_client import CoreConversationClient
from .agent import AgentLoop, compact_messages
from .agent.trace import AgentTrace
from .retrieval import QdrantRetriever, memory_scope, render_hits
from .eval_trace import emit as emit_eval_trace

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20
MAX_TOOL_ROUNDS = 3


@dataclass(slots=True)
class StreamEvent:
    type: str
    content: str | None = None
    name: str | None = None
    conversation_id: str | None = None


class ConversationMemoryStore:
    """Short-term memory for one AI process.

    This is an in-memory adapter. A database-backed history belongs behind the
    same interface after the Core conversation contract is available.
    """

    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = defaultdict(
            InMemoryChatMessageHistory
        )

    def get(self, conversation_id: str) -> InMemoryChatMessageHistory:
        history = self._histories[conversation_id]
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
        return history

    def append(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        history = self.get(conversation_id)
        history.add_user_message(user_message)
        history.add_ai_message(assistant_message)


def _context_text(context: AiContext) -> str:
    values = {
        "tutorial": context.tutorialTitle,
        "blog": context.blogTitle,
        "problem": context.problemTitle,
        "language": context.language,
        "submission_status": context.submissionStatus,
        "failed_input": context.failedCaseInput,
        "expected_output": context.expectedOutput,
        "actual_output": context.actualOutput,
        "compile_message": context.compileMessage,
    }
    lines = [f"{key}: {value}" for key, value in values.items() if value]
    if context.problemDescription:
        lines.append(f"problem_description: {context.problemDescription[:2000]}")
    if context.userCode:
        lines.append(f"user_code:\n{context.userCode[:3000]}")
    return "\n".join(lines) or "No learning context was supplied."


def build_learning_tools(
        context: AiContext,
        knowledge: QdrantRetriever | None = None,
        memory: QdrantRetriever | None = None,
        scope: str | None = None) -> list[StructuredTool]:
    """Build read-only, request-scoped tools.

    Until Core internal APIs are available, tools expose only the authorized
    context supplied by the current request; they do not access a database or
    execute user code.
    """

    context_snapshot = _context_text(context)

    def get_current_learning_context() -> str:
        return context_snapshot

    def search_current_learning_context(keyword: str) -> str:
        if not keyword.strip():
            return context_snapshot
        keyword_lower = keyword.lower()
        matches = [
            line for line in context_snapshot.splitlines() if keyword_lower in line.lower()
        ]
        return "\n".join(matches) or f"No context matched keyword: {keyword}"

    async def search_learning_knowledge(query: str, top_k: int = 3) -> str:
        if knowledge is None:
            return "Knowledge retrieval is not configured for this environment."
        return render_hits(await knowledge.search(query, top_k=top_k))

    async def search_learning_memory(query: str, top_k: int = 3) -> str:
        if memory is None or scope is None:
            return "Long-term memory retrieval is not configured for this conversation."
        return render_hits(await memory.search(query, top_k=top_k, filter_={"scope": scope}))

    return [
        StructuredTool.from_function(
            func=get_current_learning_context,
            name="get_current_learning_context",
            description="Read the current authorized tutorial, problem, code, and submission context.",
        ),
        StructuredTool.from_function(
            func=search_current_learning_context,
            name="search_current_learning_context",
            description="Search the current authorized learning context by a keyword.",
        ),
        StructuredTool.from_function(
            coroutine=search_learning_knowledge,
            name="search_learning_knowledge",
            description="Search published FlowStudy learning material and return cited source chunks.",
        ),
        StructuredTool.from_function(
            coroutine=search_learning_memory,
            name="search_learning_memory",
            description="Search only this user's prior learning conversations and return cited memories.",
        ),
    ]


class LangChainAIService:
    def __init__(self, memory: ConversationMemoryStore | None = None) -> None:
        api_key = os.environ.get("AI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("AI_API_KEY or DEEPSEEK_API_KEY environment variable is not set")

        self.model = os.environ.get("AI_MODEL") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.memory = memory or ConversationMemoryStore()
        self.core_conversations = CoreConversationClient()
        self.knowledge_retriever = QdrantRetriever("knowledge_chunks") if os.environ.get("QDRANT_ENABLED", "false").lower() == "true" else None
        self.memory_retriever = QdrantRetriever("memory_messages") if os.environ.get("QDRANT_ENABLED", "false").lower() == "true" else None
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=os.environ.get("AI_BASE_URL") or os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
            model=self.model,
            temperature=float(os.environ.get("AI_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("AI_MAX_TOKENS", "4096")),
            streaming=True,
        )
        self.agent_loop = AgentLoop(
            max_rounds=int(os.environ.get("AI_AGENT_MAX_ROUNDS", str(MAX_TOOL_ROUNDS)))
        )

    async def stream_chat(
        self,
        user_message: str,
        history: list[ChatMessage],
        context: AiContext,
        conversation_id: str | None,
        authorization: str | None = None,
        eval_case_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        if authorization:
            if not conversation_id:
                conversation_id = await self.core_conversations.create(context, authorization)
            persisted_history = (
                await self.core_conversations.messages(conversation_id, authorization)
                if conversation_id
                else []
            )
        else:
            persisted_history = []

        conversation_id = conversation_id or str(uuid.uuid4())
        user_id = await self.core_conversations.identity(authorization) if authorization else None
        scope = memory_scope(user_id, conversation_id)
        tools = build_learning_tools(context, self.knowledge_retriever, self.memory_retriever, scope)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", build_system_prompt(context)),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        stored_history = self.memory.get(conversation_id).messages
        history_source = persisted_history or [
            HumanMessage(content=item.content) if item.role == "user"
            else AIMessage(content=item.content)
            for item in history[-MAX_HISTORY_MESSAGES:]
        ]
        request_history = stored_history or history_source
        messages = prompt.format_messages(
            history=compact_messages(request_history, MAX_HISTORY_MESSAGES),
            question=user_message,
        )
        model_with_tools = self.llm.bind_tools(tools)

        yield StreamEvent(type="meta", conversation_id=conversation_id)

        final_text = ""
        trace = AgentTrace()
        async for event in self.agent_loop.run(messages, model_with_tools, tools, trace):
            if event.type == "tool":
                yield StreamEvent(type="tool", name=event.name)
            elif event.type == "answer":
                final_text = event.content or ""
                if final_text:
                    yield StreamEvent(type="token", content=final_text)

        self.memory.append(conversation_id, user_message, final_text)
        if self.memory_retriever is not None:
            try:
                await self.memory_retriever.upsert([
                    (f"{conversation_id}:user:{uuid.uuid4()}", user_message, {"scope": scope, "conversation_id": conversation_id, "role": "user"}),
                    (f"{conversation_id}:assistant:{uuid.uuid4()}", final_text, {"scope": scope, "conversation_id": conversation_id, "role": "assistant"}),
                ])
            except Exception as exc:
                logger.warning("Long-term memory persistence unavailable: %s", exc)
        trace_data = trace.finish()
        logger.info("agent_trace conversation_id=%s trace=%s", conversation_id, trace_data)
        emit_eval_trace(eval_case_id, trace_data, final_text)
        if authorization and conversation_id.isdigit():
            await self.core_conversations.append(
                conversation_id,
                ChatMessage(role="user", content=user_message),
                authorization,
                self.model,
            )
            await self.core_conversations.append(
                conversation_id,
                ChatMessage(role="assistant", content=final_text),
                authorization,
                self.model,
            )
        yield StreamEvent(type="done", conversation_id=conversation_id)

    async def generate_learning_note(self, context: AiContext) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a learning-note generator for FlowStudy. "
                    "Create concise Markdown notes from the supplied learning context. "
                    "Do not invent facts that are not present in the context.",
                ),
                (
                    "human",
                    "Generate Markdown with these sections: knowledge points, "
                    "user mistakes or risks, recommended practice, and a short summary.\n\n"
                    "Context:\n{context}",
                ),
            ]
        )
        response = await self.llm.ainvoke(prompt.format_messages(context=_context_text(context)))
        return str(response.content)

    async def analyze_learning_profile(
            self, events: list[LearningEvent]) -> ProfileAnalysisResponse:
        event_text = json.dumps(
            [event.model_dump(mode="json") for event in events[-200:]],
            ensure_ascii=False,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are FlowStudy's learning analytics assistant. "
                    "Analyze only the supplied learning events. Do not invent scores, "
                    "topics, or behavior. Return one valid JSON object only, with keys: "
                    "ability (object), weakPoints (array of objects), codingStyle (object), "
                    "summaryMd (Markdown string), learningSummaryMd (Markdown string). "
                    "Each weakPoints item should contain topic, evidence, severity, and advice. "
                    "If evidence is insufficient, say so explicitly and keep arrays empty.",
                ),
                ("human", "Learning events:\n{events}"),
            ]
        )
        response = await self.llm.ainvoke(prompt.format_messages(events=event_text))
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(
                str(item.get("text", "")) for item in raw if isinstance(item, dict)
            )
        parsed = self._parse_json_object(str(raw))
        return ProfileAnalysisResponse.model_validate(parsed)

    @staticmethod
    def _parse_json_object(raw: str) -> dict:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        candidate = fenced.group(1) if fenced else raw.strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.warning("Profile analysis returned invalid JSON: %s", exc)
            return {
                "ability": {},
                "weakPoints": [],
                "codingStyle": {},
                "summaryMd": "暂时无法生成结构化画像，请积累更多学习行为后重试。",
                "learningSummaryMd": raw[:4000],
            }
        return value if isinstance(value, dict) else {}
