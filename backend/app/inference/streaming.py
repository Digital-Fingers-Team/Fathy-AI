from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Generator, Iterable


@dataclass(slots=True)
class TokenChunk:
    index: int
    token_id: int
    text: str
    finish_reason: str | None = None


def iter_token_chunks(
    token_ids: Iterable[int],
    *,
    tokenizer,
    start_index: int = 0,
) -> Generator[TokenChunk, None, None]:
    """Yield decoded token chunks for incremental streaming."""
    for offset, token_id in enumerate(token_ids):
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        yield TokenChunk(index=start_index + offset, token_id=int(token_id), text=text)


def format_sse_chunk(chunk: TokenChunk, event: str = "token") -> str:
    """Serialize one token chunk into SSE text frame."""
    payload = json.dumps(asdict(chunk), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def format_sse_done(event: str = "done") -> str:
    return f"event: {event}\ndata: [DONE]\n\n"
