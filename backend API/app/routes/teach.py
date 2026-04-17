from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.schemas.memory import MemoryItemOut, TeachRequest

router = APIRouter()


@router.post("/teach", response_model=MemoryItemOut)
def teach(payload: TeachRequest, db: Session = Depends(get_db)):
    repo = MemoryRepository(db)
    item = repo.create(question=payload.question, answer=payload.answer, tags=payload.tags)
    return MemoryItemOut(
        id=item.id,
        question=item.question,
        answer=item.answer,
        tags=repo.to_tags(item),
        created_at=item.created_at,
        updated_at=item.updated_at,
    )
