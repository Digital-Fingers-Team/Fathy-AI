from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db
from app.repositories.memory_repo import MemoryRepository
from app.routes.dependencies import get_current_user
from app.schemas.memory import MemoryItemOut, MemoryListResponse, MemoryUpdateRequest

router = APIRouter()


@router.get("/memory", response_model=MemoryListResponse)
def list_memory(
    q: str | None = Query(default=None, max_length=500),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    items, total = repo.list(user_id=current_user.id, q=q, offset=offset, limit=limit)
    out = [
        MemoryItemOut(
            id=i.id,
            question=i.question,
            answer=i.answer,
            tags=repo.to_tags(i),
            created_at=i.created_at,
            updated_at=i.updated_at,
        )
        for i in items
    ]
    return MemoryListResponse(items=out, total=total)


@router.delete("/memory/{item_id}")
def delete_memory(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    ok = repo.delete(item_id, user_id=current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return {"deleted": True, "id": item_id}


@router.put("/memory/{item_id}", response_model=MemoryItemOut)
def update_memory(
    item_id: int,
    payload: MemoryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    item = repo.update(
        item_id,
        user_id=current_user.id,
        question=payload.question,
        answer=payload.answer,
        tags=payload.tags,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return MemoryItemOut(
        id=item.id,
        question=item.question,
        answer=item.answer,
        tags=repo.to_tags(item),
        created_at=item.created_at,
        updated_at=item.updated_at,
    )
