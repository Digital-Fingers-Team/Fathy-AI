from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Conversation, User
from app.db.session import get_db
from app.routes.dependencies import get_current_user

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationItem(BaseModel):
    id: int
    title: str
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationItem]


class ConversationCreateResponse(BaseModel):
    id: int
    title: str


class ConversationMessageItem(BaseModel):
    role: str
    content: str
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    messages: list[ConversationMessageItem]


class ConversationDeleteResponse(BaseModel):
    deleted: bool


class ConversationTitleUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationTitleUpdateResponse(BaseModel):
    id: int
    title: str


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    items = db.execute(stmt).scalars().all()
    return ConversationListResponse(
        items=[ConversationItem(id=c.id, title=c.title, updated_at=c.updated_at) for c in items]
    )


@router.post("", response_model=ConversationCreateResponse)
def create_conversation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = Conversation(user_id=current_user.id, title="New Chat")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return ConversationCreateResponse(id=conversation.id, title=conversation.title)


@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .options(selectinload(Conversation.messages))
    )
    conversation = db.execute(stmt).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    return ConversationMessagesResponse(
        messages=[
            ConversationMessageItem(role=message.role, content=message.content, created_at=message.created_at)
            for message in conversation.messages
        ]
    )


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    )
    conversation = db.execute(stmt).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    return ConversationDeleteResponse(deleted=True)


@router.patch("/{conversation_id}/title", response_model=ConversationTitleUpdateResponse)
def update_title(
    conversation_id: int,
    payload: ConversationTitleUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    )
    conversation = db.execute(stmt).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    conversation.title = payload.title.strip()
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()
    return ConversationTitleUpdateResponse(id=conversation.id, title=conversation.title)
