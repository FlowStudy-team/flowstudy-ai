from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AiContext(BaseModel):
    tutorialTitle: Optional[str] = None
    blogTitle: Optional[str] = None
    problemTitle: Optional[str] = None
    problemDescription: Optional[str] = None
    language: Optional[str] = None
    userCode: Optional[str] = None
    submissionStatus: Optional[str] = None
    failedCaseInput: Optional[str] = None
    expectedOutput: Optional[str] = None
    actualOutput: Optional[str] = None
    compileMessage: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    history: List[ChatMessage] = Field(default_factory=list)
    context: AiContext = Field(default_factory=AiContext)
    conversationId: Optional[str] = Field(default=None, max_length=128)


class NoteGenerationRequest(BaseModel):
    context: AiContext = Field(default_factory=AiContext)
    conversationId: Optional[str] = Field(default=None, max_length=128)


class TaskResponse(BaseModel):
    taskId: str
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]


class NoteTaskResponse(TaskResponse):
    result: Optional[str] = None
    error: Optional[str] = None


class LearningEvent(BaseModel):
    eventType: str = Field(..., max_length=64)
    resourceType: Optional[str] = Field(default=None, max_length=64)
    resourceId: Optional[int] = None
    durationSeconds: Optional[int] = None
    extraJson: Optional[Any] = None
    occurredAt: Optional[datetime] = None


class ProfileAnalysisRequest(BaseModel):
    events: List[LearningEvent] = Field(default_factory=list, max_length=200)


class ProfileAnalysisResponse(BaseModel):
    ability: Dict[str, Any] = Field(default_factory=dict)
    weakPoints: List[Dict[str, Any]] = Field(default_factory=list)
    codingStyle: Dict[str, Any] = Field(default_factory=dict)
    summaryMd: str = ""
    learningSummaryMd: str = ""
