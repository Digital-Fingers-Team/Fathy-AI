from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TeachRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    answer: str = Field(min_length=1, max_length=20_000)
    tags: list[str] = Field(default_factory=list, max_length=50)


class MemoryUpdateRequest(BaseModel):
    question: str | None = Field(default=None, min_length=1, max_length=10_000)
    answer: str | None = Field(default=None, min_length=1, max_length=20_000)
    tags: list[str] | None = None


class MemoryItemOut(BaseModel):
    id: int
    question: str
    answer: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    items: list[MemoryItemOut]
    total: int
