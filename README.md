# flowstudy-ai

FlowStudy 的 AI 服务，基于 Python、FastAPI 和 LangChain 提供上下文问答与学习笔记生成能力。

## 当前能力

- `POST /api/v1/ai/chat`：通过 LangChain Prompt、短期会话记忆和请求级只读工具，围绕文章、题目和判题结果进行 SSE 流式问答。
- `POST /api/v1/ai/notes/generate`：提交学习笔记生成任务。
- `GET /api/v1/ai/notes/tasks/{taskId}`：查询笔记任务状态和结果。
- `POST /api/v1/ai/profile/analyze`：接收 Core 已鉴权的学习行为，生成能力画像、薄弱点和学习总结。
- 工具调用只读取当前请求已授权的学习上下文，不直接访问数据库、不执行用户代码。

## 任务执行边界

当前 `AI_TASK_EXECUTOR=local` 使用进程内任务适配器，适用于本地联调和功能验证。服务重启后任务状态不会保留，也不具备生产环境所需的持久化、重试和死信能力。后续接入 Core/RabbitMQ 时，应保持笔记接口不变，仅替换 `app/task_service.py` 的执行器，并沿用 FlowStudy 的 `messageId`、`traceId`、`schemaVersion` 和 `eventType` 消息契约。

## 启动

```bash
python -m venv .venv
pip install -r requirements.txt
copy .env.example .env  # Windows；Linux 使用 cp
python -m uvicorn app.main:app --reload --port 8000
```

## Durable task execution

`AI_TASK_EXECUTOR=local` runs tasks in-process and stores task state in SQLite for local integration. Use `AI_TASK_EXECUTOR=rabbitmq` in a deployed environment. The service publishes durable `ai.note.generate.requested` messages to `AI_NOTE_QUEUE`, consumes them with manual requeue-on-failure semantics, and stores task state in `AI_TASK_DB_PATH`. RabbitMQ and the SQLite path must be shared or mounted durably when running multiple instances.

配置真实的 `DEEPSEEK_API_KEY` 后，访问 `/health` 和 `/api/v1/ai/health` 检查服务状态。不要提交 `.env` 或任何密钥。
