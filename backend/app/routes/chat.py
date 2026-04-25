from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import ChatMessage, Conversation, User
from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.routes.dependencies import get_current_user
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedMemory
from app.services.ai_service import AIService, HistoryMessage
from app.services.memory_service import MemoryService, ScoredMemory

router = APIRouter()


def _serialize_memories(matches: list[ScoredMemory]) -> list[RetrievedMemory]:
    return [
        RetrievedMemory(
            id=m.item.id,
            question=m.item.question,
            answer=m.item.answer,
            tags=m.tags,
            score=round(m.score, 4),
        )
        for m in matches
    ]


def _ensure_conversation(
    db: Session,
    *,
    conversation_id: int | None,
    current_user: User,
    ai: AIService,
    first_message: str,
) -> Conversation:
    if conversation_id is not None:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        conversation = db.execute(stmt).scalar_one_or_none()
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        return conversation

    title = ai.generate_title(first_message)
    conversation = Conversation(user_id=current_user.id, title=title or "New Chat")
    db.add(conversation)
    db.flush()
    return conversation


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

    history = [HistoryMessage(role=h.role, content=h.content) for h in payload.history]

    ai = AIService(request.app.state.settings, api_key=payload.api_key) if payload.api_key else request.app.state.ai_service

    conversation = _ensure_conversation(
        db,
        conversation_id=payload.conversation_id,
        current_user=current_user,
        ai=ai,
        first_message=payload.message,
    )

    result = ai.answer(payload.message, matches, history=history)
    used = _serialize_memories(matches)

    db.add(ChatMessage(conversation_id=conversation.id, role="user", content=payload.message, used_memory_ids=""))
    used_ids = ",".join(str(m.id) for m in used)
    db.add(ChatMessage(conversation_id=conversation.id, role="assistant", content=result.answer, used_memory_ids=used_ids))
    db.commit()

    return ChatResponse(
        answer=result.answer,
        used_memory=used,
        model=result.model,
        note=result.note,
        conversation_id=conversation.id,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    memory_service = MemoryService(repo)
    matches = memory_service.search(payload.message, user_id=current_user.id, limit=5)
    used = _serialize_memories(matches)

    history = [HistoryMessage(role=h.role, content=h.content) for h in payload.history]
    ai = AIService(request.app.state.settings, api_key=payload.api_key) if payload.api_key else request.app.state.ai_service

    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id, Conversation.id == payload.conversation_id)
        .options(selectinload(Conversation.messages))
    ) if payload.conversation_id is not None else None

    existing_conversation = db.execute(stmt).scalar_one_or_none() if stmt is not None else None
    if payload.conversation_id is not None and existing_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_id = existing_conversation.id if existing_conversation else None
    if conversation_id is None:
        title = ai.generate_title(payload.message)
        new_conversation = Conversation(user_id=current_user.id, title=title or "New Chat")
        db.add(new_conversation)
        db.flush()
        conversation_id = new_conversation.id

    async def event_generator() -> AsyncGenerator[str, None]:
        full_text = ""
        try:
            memory_payload = {
                "type": "memory",
                "data": [m.model_dump() for m in used],
            }
            yield f"data: {json.dumps(memory_payload, ensure_ascii=False)}\n\n"

            async for token in ai.stream_answer(payload.message, matches, history=history):
                full_text += token
                yield f"data: {json.dumps({'type': 'chunk', 'content': token}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            done_payload = {
                "type": "done",
                "model": ai.model_name,
                "note": None,
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

            db.add(ChatMessage(conversation_id=conversation_id, role="user", content=payload.message, used_memory_ids=""))
            used_ids = ",".join(str(m.id) for m in used)
            db.add(ChatMessage(conversation_id=conversation_id, role="assistant", content=full_text, used_memory_ids=used_ids))
            db.commit()
        except Exception as exc:
            db.rollback()
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
