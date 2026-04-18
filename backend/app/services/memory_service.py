from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from app.db.models import MemoryItem
from app.repositories.memory_repo import MemoryRepository


_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize(text)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _recency_boost(updated_at: datetime) -> float:
    # Small boost for recently updated items, decays over ~90 days.
    now = datetime.now(timezone.utc)
    dt = updated_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return math.exp(-days / 90.0)  # 1.0 now -> ~0.37 at 90d


@dataclass(frozen=True)
class ScoredMemory:
    item: MemoryItem
    tags: list[str]
    score: float


class MemoryService:
    """
    Production-minded MVP ranking:
    - token overlap (Jaccard) over question+answer+tags
    - substring match boost
    - light recency boost

    Easy upgrade path:
    - add embeddings + vector DB and swap `search()` implementation
    """

    def __init__(self, repo: MemoryRepository):
        self._repo = repo

    def search(self, query: str, *, limit: int = 5) -> list[ScoredMemory]:
        # Pull a bounded candidate set, then rank in Python.
        candidates, _ = self._repo.list(q=None, offset=0, limit=300)

        query_norm = _normalize(query)
        query_tokens = _tokens(query_norm)

        scored: list[ScoredMemory] = []
        for item in candidates:
            haystack = " ".join([item.question, item.answer, item.tags_csv])
            hay_norm = _normalize(haystack)
            hay_tokens = _tokens(hay_norm)

            base = _jaccard(query_tokens, hay_tokens)
            substring = 0.15 if query_norm and (query_norm in hay_norm) else 0.0
            recency = 0.10 * _recency_boost(item.updated_at)

            score = (0.75 * base) + substring + recency
            if score <= 0.0:
                continue

            scored.append(ScoredMemory(item=item, tags=self._repo.to_tags(item), score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]
