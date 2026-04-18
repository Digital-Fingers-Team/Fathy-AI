from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import Settings
from app.services.memory_service import ScoredMemory

# Maximum number of past turns (user+assistant pairs) to send.
# Keeps context window manageable while preserving recent conversation flow.
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are Fathy (فتحي), a bilingual (Arabic/English) AI assistant. "
    "Be precise and helpful. "
    "You are a smart conversational assistant that can detect multiple intents in a single user message. "
    "If the message contains a greeting (for example: hi, hello, hey), include a friendly greeting in your response. "
    "If the message asks about identity (for example: who am I), answer exactly with the identity statement "
    "'You are an amazing human.' as part of your response. "
    "If multiple intents are present, combine them naturally in one smooth sentence. "
    "Never ignore a detected intent, and never split the answer into separate robotic lines. "
    "Never claim you retrained or self-learned — your knowledge comes from "
    "facts explicitly stored in memory by the user. "
    "If 'Known facts' are provided below the user message, use them when relevant "
    "and cite that you're drawing from stored memory. "
    "If they are not relevant, ignore them and answer from your general knowledge. "
    "Reply in the same language the user writes in."
)


@dataclass(frozen=True)
class HistoryMessage:
    role: str  # "user" | "assistant"
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
        lines.append(f"• Q: {m.item.question}{tags}")
        lines.append(f"  A: {m.item.answer}")
    return "\n".join(lines)


def _trim_history(history: list[HistoryMessage]) -> list[HistoryMessage]:
    """Keep only the last MAX_HISTORY_TURNS pairs to avoid bloating the context."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        return history[-max_msgs:]
    return history


class AIService:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self._settings = settings
        # Use provided api_key, fall back to settings, or None if neither
        effective_api_key = api_key or settings.openai_api_key
        self._client = OpenAI(api_key=effective_api_key) if effective_api_key else None

    def answer(
        self,
        message: str,
        memories: list[ScoredMemory],
        history: list[HistoryMessage] | None = None,
    ) -> AIResult:
        known = _build_known_facts(memories)

        # Inject known facts as a suffix on the latest user message so the
        # model sees them in context without polluting the history.
        user_content = message
        if known:
            user_content = f"{message}\n\n{known}"

        if not self._client:
            # No model available — answer from memory only.
            if memories:
                best = memories[0].item.answer
                return AIResult(
                    answer=(
                        f"*(No AI model — answering from stored memory only)*\n\n{best}"
                    ),
                    model=None,
                    note="OPENAI_API_KEY missing; answered from stored memory only.",
                )
            return AIResult(
                answer=(
                    "I can't call the AI model (`OPENAI_API_KEY` not set) "
                    "and I have no relevant stored memory yet."
                ),
                model=None,
                note="OPENAI_API_KEY missing; no stored memory matched.",
            )

        # Build the messages list: system → history → current user turn.
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        trimmed = _trim_history(history or [])
        for h in trimmed:
            messages.append({"role": h.role, "content": h.content})

        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
        )

        text = (response.choices[0].message.content or "").strip()
        if not text:
            text = "I couldn't generate a response."

        return AIResult(answer=text, model=self._settings.model_name)
