from typing import List, Literal, Optional

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
