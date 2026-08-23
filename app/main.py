import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import _get_note_tasks, router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("AI_TASK_EXECUTOR", "local").lower() == "rabbitmq":
        await _get_note_tasks().start()
    try:
        yield
    finally:
        await _get_note_tasks().stop()


app = FastAPI(title="FlowStudy AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
