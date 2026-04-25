from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
import re
import sys

from app.core.logging import get_logger
from app.services.memory_service import ScoredMemory
from app.services.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 10
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass(frozen=True)
class LocalGenerationResult:
    answer: str
    model: str
    note: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_checkpoint_path() -> Path:
    return _repo_root() / "fathy-llm" / "checkpoints" / "sft" / "latest.pt"


def _default_tokenizer_path() -> Path:
    return _repo_root() / "fathy-llm" / "tokenizer" / "fathy_tokenizer" / "tokenizer.json"


def _ensure_fathy_paths_on_sys_path() -> None:
    fathy_root = _repo_root() / "fathy-llm"
    root_str = str(fathy_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def _contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))


def _build_known_facts(memories: Sequence[ScoredMemory]) -> str:
    if not memories:
        return ""
    lines = ["[Known facts from stored memory]"]
    for m in memories:
        tags = f" (tags: {', '.join(m.tags)})" if m.tags else ""
        lines.append(f"Q: {m.item.question}{tags}")
        lines.append(f"A: {m.item.answer}")
    return "\n".join(lines)


def _trim_history(history: Sequence[object] | None) -> list[object]:
    if not history:
        return []
    max_msgs = MAX_HISTORY_TURNS * 2
    return list(history[-max_msgs:])


def _looks_unusable(text: str) -> bool:
    value = _normalize_text(text)
    if len(value) < 20:
        return True

    chars = [ch for ch in value if not ch.isspace()]
    if not chars:
        return True

    alpha_ratio = sum(ch.isalpha() for ch in chars) / len(chars)
    punct_ratio = sum((not ch.isalnum()) for ch in chars) / len(chars)
    tokens = re.findall(r"\w+", value, flags=re.UNICODE)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)

    if alpha_ratio < 0.35:
        return True
    if punct_ratio > 0.45:
        return True
    if len(tokens) < 4:
        return True
    if unique_ratio < 0.45 and len(tokens) >= 8:
        return True
    return False


class LocalFathyModelService:
    def __init__(self, model, tokenizer, *, model_name: str, checkpoint_path: Path) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> "LocalFathyModelService":
        _ensure_fathy_paths_on_sys_path()

        import torch

        from architecture.config import ModelConfig
        from architecture.model import FathyCausalLM
        from tokenizer.tokenizer import FathyTokenizer

        ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path()
        tok_path = Path(tokenizer_path) if tokenizer_path else _default_tokenizer_path()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Local checkpoint not found: {ckpt_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Local tokenizer not found: {tok_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config_data = checkpoint.get("config")
        if not isinstance(config_data, dict):
            raise ValueError("Local checkpoint is missing config metadata")

        weights = checkpoint.get("state_dict") or checkpoint.get("model")
        if not isinstance(weights, dict):
            raise ValueError("Local checkpoint is missing model weights")

        model = FathyCausalLM(ModelConfig(**config_data))
        model.load_state_dict(weights, strict=True)
        model.eval()

        tokenizer = FathyTokenizer.from_file(tok_path)

        logger.info(
            "Loaded local Fathy checkpoint",
            extra={"checkpoint": str(ckpt_path), "tokenizer": str(tok_path)},
        )
        return cls(model, tokenizer, model_name=f"fathy-local:{ckpt_path.stem}", checkpoint_path=ckpt_path)

    @staticmethod
    def _turn_role(role: str) -> str | None:
        if role == "system":
            return "system"
        if role == "assistant":
            return "assistant"
        if role in {"user", "human"}:
            return "human"
        return None

    def _build_prompt(
        self,
        message: str,
        memories: Sequence[ScoredMemory],
        history: Sequence[object] | None = None,
    ) -> str:
        from tokenizer.tokenizer import ASSISTANT, BEGIN_OF_TEXT

        pieces: list[str] = [BEGIN_OF_TEXT, self._tokenizer.format_turn("system", SYSTEM_PROMPT)]

        for turn in _trim_history(history):
            role = self._turn_role(str(getattr(turn, "role", "")).lower())
            content = _normalize_text(str(getattr(turn, "content", "")))
            if not role or not content:
                continue
            pieces.append(self._tokenizer.format_turn(role, content))

        user_content = _normalize_text(message)
        known_facts = _build_known_facts(memories)
        if known_facts:
            user_content = f"{user_content}\n\n{known_facts}"

        pieces.append(self._tokenizer.format_turn("human", user_content))
        pieces.append(ASSISTANT)
        return "\n".join(pieces)

    def _fallback_response(self, message: str, memories: Sequence[ScoredMemory]) -> tuple[str, str]:
        lower = (message or "").lower()
        english_greeting = bool(re.search(r"\b(hi|hello|hey)\b", lower))
        arabic_greeting = any(term in message for term in ("مرحبا", "اهلا", "أهلا", "سلام"))
        if english_greeting or arabic_greeting:
            if _contains_arabic(message):
                return "أهلاً! كيف أقدر أساعدك اليوم؟", "Local checkpoint output was low quality; returned a greeting fallback."
            return "Hello! How can I help you today?", "Local checkpoint output was low quality; returned a greeting fallback."

        if "who am i" in lower or "من انا" in lower or "مين انا" in lower:
            return "You are an amazing human.", "Local checkpoint output was low quality; returned the identity fallback."

        if memories:
            return memories[0].item.answer, "Local checkpoint output was low quality; returned stored memory instead."

        if _contains_arabic(message):
            return (
                "أنا شغّال على الموديل المحلي، لكن المخرجات منه لسه غير مستقرة. جرّب تسألني بشكل أوضح أو علّمني معلومة أولاً.",
                "Local checkpoint output was low quality; returned a clear fallback in Arabic.",
            )
        return (
            "I loaded the local Fathy checkpoint, but it isn't producing a clean answer yet. Try a more specific question, or teach me the fact first.",
            "Local checkpoint output was low quality; returned a clear fallback.",
        )

    def _generate_ids(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_k: int = 50,
    ):
        import torch

        input_ids = torch.tensor([self._tokenizer.encode(prompt, normalize=False)], dtype=torch.long)
        input_ids = input_ids.to(next(self._model.parameters()).device)

        end_turn_id = self._tokenizer.special_token_ids.end_turn
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=end_turn_id,
            )
        return output_ids

    def answer(
        self,
        message: str,
        memories: Sequence[ScoredMemory],
        history: Sequence[object] | None = None,
    ) -> LocalGenerationResult:
        prompt = self._build_prompt(message, memories, history=history)
        prompt_ids = self._tokenizer.encode(prompt, normalize=False)
        output_ids = self._generate_ids(prompt)

        generated_ids = output_ids[0, len(prompt_ids) :].tolist()
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        note = f"Answered locally from {self.checkpoint_path.name}."
        if not answer or _looks_unusable(answer):
            answer, note = self._fallback_response(message, memories)

        return LocalGenerationResult(
            answer=answer,
            model=self.model_name,
            note=note,
        )

    def stream_answer(
        self,
        message: str,
        memories: Sequence[ScoredMemory],
        history: Sequence[object] | None = None,
    ) -> Iterable[str]:
        yield self.answer(message, memories, history=history).answer


@lru_cache(maxsize=1)
def get_local_fathy_model(
    checkpoint_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> LocalFathyModelService | None:
    try:
        return LocalFathyModelService.load(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
    except Exception as exc:  # pragma: no cover - logged and handled by caller
        logger.warning("Local Fathy model unavailable; falling back to remote or memory-only mode: %s", exc)
        return None
