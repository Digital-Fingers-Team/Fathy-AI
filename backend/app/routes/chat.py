from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.routes.dependencies import get_current_user
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedMemory
from app.services.ai_service import HistoryMessage
from app.services.memory_service import MemoryService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    memory_service = MemoryService(repo)
    matches = memory_service.search(payload.message, user_id=current_user.id, limit=5)

    # Convert schema history to service-layer dataclass.
    history = [HistoryMessage(role=h.role, content=h.content) for h in payload.history]

    # Use the AI service from app state, or create a new one with client-provided API key
    if payload.api_key:
        # If client provides an API key, create a new service with it
        from app.services.ai_service import AIService
        ai = AIService(request.app.state.settings, api_key=payload.api_key)
    else:
        # Otherwise use the global AI service
        ai = request.app.state.ai_service

    result = ai.answer(payload.message, matches, history=history)

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

    return ChatResponse(
        answer=result.answer,
        used_memory=used,
        model=result.model,
        note=result.note,
    )
