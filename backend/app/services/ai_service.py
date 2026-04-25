from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import Settings
from app.services.local_fathy_model import get_local_fathy_model
from app.services.memory_service import ScoredMemory
from app.services.prompts import SYSTEM_PROMPT

MAX_HISTORY_TURNS = 10


@dataclass(frozen=True)
class HistoryMessage:
    role: str
    content: str


@dataclass(frozen=True)
class AIResult:
    answer: str
    model: str | None = None
    note: str | None = None


def _build_known_facts(memories: list[ScoredMemory]) -> str:
    if not memories:
        return ""
    lines = ["[Known facts from stored memory]"]
    for m in memories:
        tags = f" (tags: {', '.join(m.tags)})" if m.tags else ""
        lines.append(f"Q: {m.item.question}{tags}")
        lines.append(f"A: {m.item.answer}")
    return "\n".join(lines)


def _trim_history(history: list[HistoryMessage]) -> list[HistoryMessage]:
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        return history[-max_msgs:]
    return history


class AIService:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self._settings = settings
        effective_api_key = api_key or settings.openai_api_key
        self._client = OpenAI(api_key=effective_api_key) if effective_api_key else None
        self._local_model = get_local_fathy_model()

    @property
    def model_name(self) -> str | None:
        if self._local_model is not None:
            return self._local_model.model_name
        if self._client is not None:
            return self._settings.model_name
        return None

    def _build_messages(
        self,
        message: str,
        memories: list[ScoredMemory],
        history: list[HistoryMessage] | None = None,
    ) -> list[dict[str, str]]:
        known = _build_known_facts(memories)
        user_content = f"{message}\n\n{known}" if known else message
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in _trim_history(history or []):
            messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _fallback_text(self, memories: list[ScoredMemory]) -> tuple[str, str]:
        if memories:
            best = memories[0].item.answer
            return (
                f"*(No AI model — answering from stored memory only)*\n\n{best}",
                "OPENAI_API_KEY missing; answered from stored memory only.",
            )
        return (
            "I can't call the AI model (`OPENAI_API_KEY` not set) and I have no relevant stored memory yet.",
            "OPENAI_API_KEY missing; no stored memory matched.",
        )

    def answer(
        self,
        message: str,
        memories: list[ScoredMemory],
        history: list[HistoryMessage] | None = None,
    ) -> AIResult:
        if self._local_model is not None:
            local_result = self._local_model.answer(message, memories, history=history)
            return AIResult(answer=local_result.answer, model=local_result.model, note=local_result.note)

        if not self._client:
            text, note = self._fallback_text(memories)
            return AIResult(answer=text, model=None, note=note)

        messages = self._build_messages(message, memories, history)
        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip() or "I couldn't generate a response."
        return AIResult(answer=text, model=self._settings.model_name)

    async def stream_answer(
        self,
        message: str,
        memories: list[ScoredMemory],
        history: list[HistoryMessage] | None = None,
    ) -> AsyncGenerator[str, None]:
        if self._local_model is not None:
            for chunk in self._local_model.stream_answer(message, memories, history=history):
                yield chunk
            return

        if not self._client:
            text, _ = self._fallback_text(memories)
            yield text
            return

        messages = self._build_messages(message, memories, history)
        stream = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._settings.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
                await asyncio.sleep(0)

    def generate_title(self, first_message: str) -> str:
        """Generate a short conversation title from the first user message."""
        if not self._client:
            return first_message[:40]

        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a short title (max 6 words, no quotes) "
                        f"for a conversation that starts with: {first_message[:200]}"
                    ),
                }
            ],
            max_tokens=20,
            temperature=0.3,
        )
        return (response.choices[0].message.content or first_message[:40]).strip()
