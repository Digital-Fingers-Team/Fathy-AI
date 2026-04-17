from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class RetrievedMemory(BaseModel):
    id: int
    question: str
    answer: str
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    used_memory: list[RetrievedMemory] = Field(default_factory=list)
    model: str | None = None
    note: str | None = None
