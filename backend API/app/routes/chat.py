from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedMemory
from app.services.memory_service import MemoryService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, db: Session = Depends(get_db)):
    repo = MemoryRepository(db)
    memory_service = MemoryService(repo)
    matches = memory_service.search(payload.message, limit=5)

    ai = request.app.state.ai_service
    result = ai.answer(payload.message, matches)

    used = [
        RetrievedMemory(
            id=m.item.id,
            question=m.item.question,
            answer=m.item.answer,
            tags=m.tags,
            score=round(m.score, 4),
        )
        for m in matches
    ]

    return ChatResponse(answer=result.answer, used_memory=used, model=result.model, note=result.note)
