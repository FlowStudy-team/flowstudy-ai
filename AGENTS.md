# flowstudy-ai Agent Guide

## Service Responsibility

`flowstudy-ai` 是 FlowStudy 的 AI 服务，当前负责 FastAPI 健康检查、AI 聊天 SSE、Prompt 构造和 DeepSeek/OpenAI-compatible 模型流式调用。它不负责 Core 业务数据、判题执行、前端 UI 或生产部署。

## Technology Stack

- Python FastAPI：`requirements.txt`
- Uvicorn：`uvicorn[standard]==0.34.0`
- OpenAI Python SDK：`openai==1.58.1`
- Pydantic 2：`pydantic==2.10.3`
- dotenv/httpx：`python-dotenv`、`httpx`

## Important Entry Points

- 应用入口：`app/main.py`
- 路由：`app/router.py`
- Schema：`app/schemas.py`
- Prompt：`app/prompt.py`
- 模型客户端：`app/llm_client.py`
- 依赖：`requirements.txt`
- 环境示例：`.env.example`

## Key Modules

- `main.py`：创建 FastAPI app、CORS、注册 router、`/health`
- `router.py`：`/api/v1/ai/chat` SSE 和 `/api/v1/ai/health`
- `llm_client.py`：读取模型配置并调用流式 chat completion
- `prompt.py`：根据教程、题目、代码和判题上下文构造 system prompt
- `schemas.py`：`ChatRequest`、`AiContext`、`ChatMessage`

## External Dependencies

- DeepSeek/OpenAI-compatible API：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`
- Frontend 通过 Vite `/ai` proxy 调用。
- 当前未发现 Core 内部 API、MySQL、Redis 或 RabbitMQ 客户端。

## Contracts

- AI API：[../flowstudy-infra/docs/11-ai-agent-service-design.md](../flowstudy-infra/docs/11-ai-agent-service-design.md)
- REST/OpenAPI：[../flowstudy-infra/docs/05-restful-api-contract.md](../flowstudy-infra/docs/05-restful-api-contract.md)、[../flowstudy-infra/docs/api/FlowStudy_Apifox_OpenAPI.yaml](../flowstudy-infra/docs/api/FlowStudy_Apifox_OpenAPI.yaml)
- 错误格式：[../flowstudy-infra/docs/06-result-error-code-contract.md](../flowstudy-infra/docs/06-result-error-code-contract.md)

## Environment Variables

入口：`.env.example` 和运行时环境变量。不要读取、输出或提交真实 `.env`。

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

## Validation Commands

```bash
python -m venv .venv
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

当前未发现测试命令或测试目录。可用只读健康检查：

```bash
python -m compileall app
```

## Modification Rules

- 不得写入真实 API Key。
- 模型调用必须处理异常；新增外部调用时要说明超时、重试和降级。
- Prompt 变更需说明对用户回答、安全和上下文使用的影响。
- AI 失败不应影响登录、教程、题目、判题等核心流程，除非设计文档明确改变。
- SSE 格式变化必须同步 frontend `src/api/modules/ai.ts` 和 infra 契约。

## Task Completion Checklist

完成任务时说明：修改了什么、为什么修改、运行了哪些命令、哪些验证通过、哪些未验证、是否影响契约、是否需要更新 infra 文档。
