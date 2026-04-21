from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import MemoryItem


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tags_to_csv(tags: list[str]) -> str:
    cleaned: list[str] = []
    for tag in tags:
        t = (tag or "").strip()
        if not t:
            continue
        if "," in t:
            t = t.replace(",", " ")
        cleaned.append(t)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in cleaned:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return ",".join(deduped)


def _csv_to_tags(csv: str) -> list[str]:
    if not csv:
        return []
    return [part.strip() for part in csv.split(",") if part.strip()]


class MemoryRepository:
    def __init__(self, db: Session):
        self._db = db

    def create(self, user_id: int, question: str, answer: str, tags: list[str]) -> MemoryItem:
        item = MemoryItem(user_id=user_id, question=question, answer=answer, tags_csv=_tags_to_csv(tags))
        self._db.add(item)
        self._db.commit()
        self._db.refresh(item)
        return item

    def get(self, item_id: int, *, user_id: int) -> MemoryItem | None:
        stmt = select(MemoryItem).where(MemoryItem.id == item_id, MemoryItem.user_id == user_id)
        return self._db.execute(stmt).scalar_one_or_none()

    def delete(self, item_id: int, *, user_id: int) -> bool:
        item = self.get(item_id, user_id=user_id)
        if item is None:
            return False
        self._db.delete(item)
        self._db.commit()
        return True

    def update(
        self,
        item_id: int,
        user_id: int,
        *,
        question: str | None = None,
        answer: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem | None:
        item = self.get(item_id, user_id=user_id)
        if item is None:
            return None
        if question is not None:
            item.question = question
        if answer is not None:
            item.answer = answer
        if tags is not None:
            item.tags_csv = _tags_to_csv(tags)
        item.updated_at = _now()
        self._db.add(item)
        self._db.commit()
        self._db.refresh(item)
        return item

    def list(
        self, *, user_id: int, q: str | None = None, offset: int = 0, limit: int = 50
    ) -> tuple[list[MemoryItem], int]:
        stmt = select(MemoryItem).where(MemoryItem.user_id == user_id)
        count_stmt = select(func.count(MemoryItem.id)).where(MemoryItem.user_id == user_id)

        if q:
            like = f"%{q}%"
            stmt = stmt.where(
                (MemoryItem.question.like(like))
                | (MemoryItem.answer.like(like))
                | (MemoryItem.tags_csv.like(like))
            )
            count_stmt = count_stmt.where(
                (MemoryItem.question.like(like))
                | (MemoryItem.answer.like(like))
                | (MemoryItem.tags_csv.like(like))
            )

        stmt = stmt.order_by(MemoryItem.updated_at.desc()).offset(offset).limit(limit)
        items = list(self._db.execute(stmt).scalars().all())
        total = int(self._db.execute(count_stmt).scalar_one())
        return items, total

    @staticmethod
    def to_tags(item: MemoryItem) -> list[str]:
        return _csv_to_tags(item.tags_csv)
