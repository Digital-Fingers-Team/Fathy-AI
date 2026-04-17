from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from app.core.config import Settings
from app.services.memory_service import ScoredMemory


@dataclass(frozen=True)
class AIResult:
    answer: str
    model: str | None = None
    note: str | None = None


def _build_known_facts(memories: list[ScoredMemory]) -> str:
    if not memories:
        return ""
    lines = ["Known facts:"]
    for m in memories:
        tags = f" (tags: {', '.join(m.tags)})" if m.tags else ""
        lines.append(f"* Q: {m.item.question}{tags}")
        lines.append(f"  A: {m.item.answer}")
    lines.append("")
    lines.append("Answer the user based on these facts if relevant. If not relevant, ignore them.")
    return "\n".join(lines)


class AIService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def answer(self, message: str, memories: list[ScoredMemory]) -> AIResult:
        known = _build_known_facts(memories)

        if not self._client:
            # Real behavior: no model call without credentials. Use memory only.
            if memories:
                return AIResult(
                    answer=f"{known}\n\nI can't call the AI model because `OPENAI_API_KEY` is not set. "
                    f"Based on saved memory above, here's what I can say:\n\n{memories[0].item.answer}",
                    model=None,
                    note="OPENAI_API_KEY missing; answered from stored memory only.",
                )
            return AIResult(
                answer="I can't call the AI model because `OPENAI_API_KEY` is not set, and I have no relevant stored memory yet.",
                model=None,
                note="OPENAI_API_KEY missing; no stored memory matched.",
            )

        instructions = (
            "You are Fathy (فتحي), a bilingual (Arabic/English) AI assistant. "
            "Be precise, explainable, and avoid claiming you retrained or self-learned. "
            "If you use 'Known facts', only use them when relevant."
        )

        user_input = f"{known}\n\nUser message:\n{message}" if known else message

        resp = self._client.responses.create(
            model=self._settings.model_name,
            instructions=instructions,
            input=user_input,
        )

        text = (resp.output_text or "").strip()
        if not text:
            text = "I couldn't generate a response."
        return AIResult(answer=text, model=self._settings.model_name)
