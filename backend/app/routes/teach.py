from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.routes.dependencies import get_current_user
from app.schemas.memory import MemoryItemOut, TeachRequest

router = APIRouter()


@router.post("/teach", response_model=MemoryItemOut)
def teach(
    payload: TeachRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    item = repo.create(
        user_id=current_user.id,
        question=payload.question,
        answer=payload.answer,
        tags=payload.tags,
    )
    return MemoryItemOut(
        id=item.id,
        question=item.question,
        answer=item.answer,
        tags=repo.to_tags(item),
        created_at=item.created_at,
        updated_at=item.updated_at,
    )
