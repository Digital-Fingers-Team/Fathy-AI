"""Hugging Face tokenizer wrapper for Fathy LLM training/inference.

This module provides:
- training/loading/saving APIs for a 100k-vocab tokenizer,
- deterministic special-token registration and placement,
- robust conversation formatting helpers,
- configurable Arabic normalization and pre-tokenization controls,
- typed encode/decode helpers for single items and batches.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

# Required special tokens in deterministic order.
BEGIN_OF_TEXT = "<|begin_of_text|>"
END_OF_TEXT = "<|end_of_text|>"
HUMAN = "<|human|>"
ASSISTANT = "<|assistant|>"
SYSTEM = "<|system|>"
END_TURN = "<|end_turn|>"
UNK = "<unk>"

SPECIAL_TOKENS: tuple[str, ...] = (
    BEGIN_OF_TEXT,
    END_OF_TEXT,
    HUMAN,
    ASSISTANT,
    SYSTEM,
    END_TURN,
)

Role = Literal["system", "human", "assistant"]

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_TATWEEL_RE = re.compile(r"\u0640")
_ARABIC_PUNCT_RE = re.compile(r"[،؛؟٪٬٫۔]")
_MULTI_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ArabicNormalizationOptions:
    """Arabic-specific text normalization toggles."""

    normalize_unicode_nfc: bool = True
    strip_diacritics: bool = True
    strip_tatweel: bool = True
    space_around_arabic_punctuation: bool = True
    collapse_whitespace: bool = True


@dataclass(frozen=True)
class PreTokenizationOptions:
    """Pre-tokenization controls to reduce harmful over-fragmentation."""

    split_whitespace: bool = True
    split_punctuation: bool = True
    punctuation_behavior: Literal["removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous"] = (
        "isolated"
    )
    individual_digits: bool = False
    extra_split_patterns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConversationTurn:
    role: Role
    content: str


@dataclass(frozen=True)
class SpecialTokenIds:
    unk: int
    begin_of_text: int
    end_of_text: int
    human: int
    assistant: int
    system: int
    end_turn: int


def normalize_arabic_text(text: str, options: ArabicNormalizationOptions) -> str:
    """Normalize Arabic text according to configuration."""
    value = text

    if options.normalize_unicode_nfc:
        value = unicodedata.normalize("NFC", value)

    if options.strip_diacritics:
        value = _ARABIC_DIACRITICS_RE.sub("", value)

    if options.strip_tatweel:
        value = _ARABIC_TATWEEL_RE.sub("", value)

    if options.space_around_arabic_punctuation:
        value = _ARABIC_PUNCT_RE.sub(lambda m: f" {m.group(0)} ", value)

    if options.collapse_whitespace:
        value = _MULTI_WS_RE.sub(" ", value).strip()

    return value


def _validate_conversation(turns: Sequence[ConversationTurn]) -> None:
    if not turns:
        return

    system_count = 0
    for idx, turn in enumerate(turns):
        if turn.role == "system":
            system_count += 1
            if idx != 0:
                raise ValueError("If a system turn is present, it must be the first turn.")
        if system_count > 1:
            raise ValueError("Conversation supports at most one system turn.")


class FathyTokenizer:
    """High-level API over HF tokenizers with conversation-aware helpers."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        normalization_options: ArabicNormalizationOptions | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._normalization_options = normalization_options or ArabicNormalizationOptions()
        self._special_ids = self._resolve_special_token_ids(tokenizer)

    @property
    def hf_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def special_token_ids(self) -> SpecialTokenIds:
        return self._special_ids

    @staticmethod
    def _resolve_special_token_ids(tokenizer: Tokenizer) -> SpecialTokenIds:
        def _id(token: str) -> int:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Special token missing from tokenizer vocabulary: {token}")
            return token_id

        return SpecialTokenIds(
            unk=_id(UNK),
            begin_of_text=_id(BEGIN_OF_TEXT),
            end_of_text=_id(END_OF_TEXT),
            human=_id(HUMAN),
            assistant=_id(ASSISTANT),
            system=_id(SYSTEM),
            end_turn=_id(END_TURN),
        )

    @classmethod
    def train(
        cls,
        files: Sequence[str | Path],
        *,
        vocab_size: int = 100_000,
        min_frequency: int = 2,
        normalization_options: ArabicNormalizationOptions | None = None,
        pretokenization_options: PreTokenizationOptions | None = None,
    ) -> FathyTokenizer:
        """Train a BPE tokenizer with deterministic special-token setup."""
        if vocab_size <= len(SPECIAL_TOKENS) + 1:
            raise ValueError("vocab_size must be greater than the number of reserved special tokens.")

        tokenizer = Tokenizer(models.BPE(unk_token=UNK))

        # Minimal model-level normalization; Arabic normalization remains configurable at wrapper level.
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFC(),
                normalizers.Replace("\u00A0", " "),
            ]
        )

        pre_opts = pretokenization_options or PreTokenizationOptions()
        tokenizer.pre_tokenizer = _build_pre_tokenizer(pre_opts)
        tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[UNK, *SPECIAL_TOKENS],
            show_progress=True,
        )

        tokenizer.train([str(path) for path in files], trainer=trainer)

        begin_id = tokenizer.token_to_id(BEGIN_OF_TEXT)
        end_id = tokenizer.token_to_id(END_OF_TEXT)
        if begin_id is None or end_id is None:
            raise ValueError("Failed to resolve BOS/EOS ids after tokenizer training.")

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BEGIN_OF_TEXT} $A {END_OF_TEXT}",
            pair=f"{BEGIN_OF_TEXT} $A {END_OF_TEXT} $B:1 {END_OF_TEXT}:1",
            special_tokens=[
                (BEGIN_OF_TEXT, begin_id),
                (END_OF_TEXT, end_id),
            ],
        )

        return cls(tokenizer, normalization_options=normalization_options)

    @classmethod
    def from_file(
        cls,
        tokenizer_path: str | Path,
        *,
        normalization_options: ArabicNormalizationOptions | None = None,
    ) -> FathyTokenizer:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer, normalization_options=normalization_options)

    def save(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(output))

    @staticmethod
    def system_turn(content: str) -> ConversationTurn:
        return ConversationTurn(role="system", content=content)

    @staticmethod
    def human_turn(content: str) -> ConversationTurn:
        return ConversationTurn(role="human", content=content)

    @staticmethod
    def assistant_turn(content: str) -> ConversationTurn:
        return ConversationTurn(role="assistant", content=content)

    def normalize_text(self, text: str) -> str:
        return normalize_arabic_text(text, self._normalization_options)

    def format_turn(self, role: Role, content: str) -> str:
        role_token = {
            "system": SYSTEM,
            "human": HUMAN,
            "assistant": ASSISTANT,
        }[role]
        normalized = self.normalize_text(content)
        return f"{role_token}\n{normalized}\n{END_TURN}"

    def format_conversation(
        self,
        turns: Sequence[ConversationTurn],
        *,
        add_begin_of_text: bool = True,
        add_end_of_text: bool = True,
    ) -> str:
        _validate_conversation(turns)

        blocks: list[str] = []
        if add_begin_of_text:
            blocks.append(BEGIN_OF_TEXT)
        blocks.extend(self.format_turn(turn.role, turn.content) for turn in turns)
        if add_end_of_text:
            blocks.append(END_OF_TEXT)
        return "\n".join(blocks)

    def encode(self, text: str, *, normalize: bool = True) -> list[int]:
        source = self.normalize_text(text) if normalize else text
        return self._tokenizer.encode(source).ids

    def encode_batch(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[int]]:
        prepared = [self.normalize_text(text) for text in texts] if normalize else list(texts)
        return [enc.ids for enc in self._tokenizer.encode_batch(prepared)]

    def encode_conversation(
        self,
        turns: Sequence[ConversationTurn],
        *,
        add_begin_of_text: bool = True,
        add_end_of_text: bool = True,
    ) -> list[int]:
        formatted = self.format_conversation(
            turns,
            add_begin_of_text=add_begin_of_text,
            add_end_of_text=add_end_of_text,
        )
        return self._tokenizer.encode(formatted).ids

    def encode_conversation_batch(
        self,
        conversations: Sequence[Sequence[ConversationTurn]],
        *,
        add_begin_of_text: bool = True,
        add_end_of_text: bool = True,
    ) -> list[list[int]]:
        formatted_batch = [
            self.format_conversation(
                turns,
                add_begin_of_text=add_begin_of_text,
                add_end_of_text=add_end_of_text,
            )
            for turns in conversations
        ]
        return [enc.ids for enc in self._tokenizer.encode_batch(formatted_batch)]

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch_token_ids: Sequence[Sequence[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self._tokenizer.decode_batch(
            [list(ids) for ids in batch_token_ids],
            skip_special_tokens=skip_special_tokens,
        )


def _build_pre_tokenizer(options: PreTokenizationOptions) -> pre_tokenizers.PreTokenizer:
    pretokenizers: list[pre_tokenizers.PreTokenizer] = []

    if options.split_whitespace:
        pretokenizers.append(pre_tokenizers.Whitespace())

    if options.split_punctuation:
        pretokenizers.append(pre_tokenizers.Punctuation(behavior=options.punctuation_behavior))

    pretokenizers.append(pre_tokenizers.Digits(individual_digits=options.individual_digits))

    for pattern in options.extra_split_patterns:
        pretokenizers.append(pre_tokenizers.Split(pattern=pattern, behavior="isolated", invert=False))

    if not pretokenizers:
        return pre_tokenizers.ByteLevel(add_prefix_space=False)

    if len(pretokenizers) == 1:
        return pretokenizers[0]

    return pre_tokenizers.Sequence(pretokenizers)


def iter_training_texts(
    conversations: Iterable[Sequence[ConversationTurn]],
    tokenizer: FathyTokenizer,
    *,
    add_begin_of_text: bool = True,
    add_end_of_text: bool = True,
) -> Iterable[str]:
    """Yield deterministic, formatted conversation texts for corpus generation."""
    for turns in conversations:
        yield tokenizer.format_conversation(
            turns,
            add_begin_of_text=add_begin_of_text,
            add_end_of_text=add_end_of_text,
        )
