from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    # The frontend sends recent conversation turns so the backend stays stateless.
    # Oldest-first ordering; the server will trim to MAX_HISTORY_TURNS pairs.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)


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
