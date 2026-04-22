"""Dataset collators/loaders for pretrain, SFT, preference, and RL rollouts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, get_worker_info

from tokenizer.tokenizer import ASSISTANT, BEGIN_OF_TEXT, END_TURN, HUMAN, SYSTEM, FathyTokenizer


@dataclass
class RLHFHyperparameters:
    """Shared knobs frequently tuned during RLHF optimization."""

    kl_penalty_weight: float = 0.02
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_bonus: float = 0.01


@dataclass
class PaddingConfig:
    pad_token_id: int = 0
    label_pad_token_id: int = -100


def _to_tensor(value: Any, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _pad_1d(items: list[Any], dtype: torch.dtype, padding_value: int | float) -> Tensor:
    tensors = [_to_tensor(item, dtype=dtype).view(-1) for item in items]
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def _encode_raw(tokenizer: FathyTokenizer, text: str) -> list[int]:
    normalized = tokenizer.normalize_text(text)
    return tokenizer.hf_tokenizer.encode(normalized, add_special_tokens=False).ids


def _build_block_causal_mask(document_ids: Sequence[int]) -> Tensor:
    doc = torch.tensor(document_ids, dtype=torch.long)
    same_doc = doc.unsqueeze(0).eq(doc.unsqueeze(1))
    causal = torch.tril(torch.ones((len(document_ids), len(document_ids)), dtype=torch.bool))
    return (same_doc & causal).to(dtype=torch.long)


class PretrainingDataset(Dataset[dict[str, Tensor]]):
    """JSONL dataset that tokenizes text and packs multiple documents per sequence."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: FathyTokenizer,
        *,
        max_seq_len: int,
        text_key: str = "text",
        seed: int = 0,
        shuffle: bool = True,
        streaming: bool = False,
        drop_last: bool = False,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.seed = seed
        self.shuffle = shuffle
        self.streaming = streaming
        self.drop_last = drop_last

        self._line_offsets = self._index_offsets()
        if not self.streaming:
            self._samples = list(self._build_samples(self._ordered_indices(seed=self.seed)))

    def _index_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                offsets.append(offset)
                offset += len(line.encode("utf-8"))
        return offsets

    def _ordered_indices(self, *, seed: int) -> list[int]:
        indices = list(range(len(self._line_offsets)))
        if self.shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)
        return indices

    def _read_line(self, line_idx: int) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self._line_offsets[line_idx])
            line = fh.readline()
        return json.loads(line)

    def _iter_documents(self, indices: Iterable[int]) -> Iterator[tuple[int, list[int]]]:
        for doc_uid, line_idx in enumerate(indices):
            row = self._read_line(line_idx)
            text = row.get(self.text_key)
            if not isinstance(text, str) or not text:
                continue
            token_ids = _encode_raw(self.tokenizer, text)
            if token_ids:
                yield doc_uid, token_ids

    def _build_samples(self, indices: Iterable[int]) -> Iterator[dict[str, Tensor]]:
        chunk_tokens: list[int] = []
        chunk_doc_ids: list[int] = []

        for doc_uid, token_ids in self._iter_documents(indices):
            cursor = 0
            while cursor < len(token_ids):
                remaining = self.max_seq_len - len(chunk_tokens)
                take = min(remaining, len(token_ids) - cursor)
                chunk_tokens.extend(token_ids[cursor : cursor + take])
                chunk_doc_ids.extend([doc_uid] * take)
                cursor += take

                if len(chunk_tokens) == self.max_seq_len:
                    ids = torch.tensor(chunk_tokens, dtype=torch.long)
                    yield {
                        "input_ids": ids,
                        "labels": ids.clone(),
                        "attention_mask": _build_block_causal_mask(chunk_doc_ids),
                    }
                    chunk_tokens = []
                    chunk_doc_ids = []

        if chunk_tokens and not self.drop_last:
            ids = torch.tensor(chunk_tokens, dtype=torch.long)
            yield {
                "input_ids": ids,
                "labels": ids.clone(),
                "attention_mask": _build_block_causal_mask(chunk_doc_ids),
            }

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming PretrainingDataset does not support __len__.")
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.streaming:
            raise TypeError("Use iteration with streaming=True.")
        return self._samples[idx]

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        if not self.streaming:
            return iter(self._samples)

        worker = get_worker_info()
        if worker is None:
            indices = self._ordered_indices(seed=self.seed)
        else:
            indices = self._ordered_indices(seed=self.seed + worker.id)
            indices = indices[worker.id :: worker.num_workers]

        return self._build_samples(indices)


class InstructionDataset(Dataset[dict[str, Tensor]]):
    """Instruction tuning dataset with assistant-only labels and left truncation."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: FathyTokenizer,
        *,
        max_seq_len: int,
        conversation_key: str = "messages",
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.conversation_key = conversation_key
        self._rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self._rows)

    def _role_token(self, role: str) -> str:
        if role == "human":
            return HUMAN
        if role == "assistant":
            return ASSISTANT
        if role == "system":
            return SYSTEM
        raise ValueError(f"Unsupported role: {role}")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self._rows[idx]
        turns = row.get(self.conversation_key)
        if not isinstance(turns, list) or len(turns) < 1:
            raise ValueError("Instruction sample must include a non-empty conversation list.")

        assistant_idx = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].get("role") == "assistant":
                assistant_idx = i
                break
        if assistant_idx is None:
            raise ValueError("Instruction sample must include at least one assistant turn.")

        prompt_parts = [BEGIN_OF_TEXT]
        for turn in turns[:assistant_idx]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role not in {"system", "human", "assistant"}:
                continue
            prompt_parts.append(self._role_token(role))
            prompt_parts.append(str(content))
            prompt_parts.append(END_TURN)
        prompt_parts.append(ASSISTANT)
        prompt_text = "\n".join(prompt_parts)

        assistant_content = str(turns[assistant_idx].get("content", ""))
        assistant_text = f"{assistant_content}\n{END_TURN}"

        prompt_ids = _encode_raw(self.tokenizer, prompt_text)
        answer_ids = _encode_raw(self.tokenizer, assistant_text)

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]
            labels = labels[-self.max_seq_len :]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def language_modeling_collate_fn(
    batch: list[dict[str, Any]],
    padding: PaddingConfig | None = None,
) -> dict[str, Tensor]:
    """Pad LM batches to the longest sample; supports 1D or 2D attention masks."""

    padding = padding or PaddingConfig()
    collated = {
        "input_ids": _pad_1d([x["input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "labels": _pad_1d([x["labels"] for x in batch], torch.long, padding.label_pad_token_id),
    }

    sample_attention = _to_tensor(batch[0]["attention_mask"], dtype=torch.long)
    if sample_attention.dim() == 1:
        collated["attention_mask"] = _pad_1d([x["attention_mask"] for x in batch], torch.long, 0)
    elif sample_attention.dim() == 2:
        lengths = [_to_tensor(x["input_ids"], dtype=torch.long).numel() for x in batch]
        max_len = max(lengths)
        attn = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
        for row, item in enumerate(batch):
            mask = _to_tensor(item["attention_mask"], dtype=torch.long)
            seq_len = mask.size(0)
            attn[row, :seq_len, :seq_len] = mask
        collated["attention_mask"] = attn
    else:
        raise ValueError("attention_mask must be rank-1 or rank-2 tensors.")

    return collated


def pretrain_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    return {
        "input_ids": _pad_1d([x["input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "attention_mask": _pad_1d([x["attention_mask"] for x in batch], torch.long, 0),
        "labels": _pad_1d([x["labels"] for x in batch], torch.long, padding.label_pad_token_id),
    }


def sft_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    return pretrain_collate_fn(batch, padding=padding)


def preference_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    return {
        "chosen_input_ids": _pad_1d([x["chosen_input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "chosen_attention_mask": _pad_1d([x["chosen_attention_mask"] for x in batch], torch.long, 0),
        "rejected_input_ids": _pad_1d([x["rejected_input_ids"] for x in batch], torch.long, padding.pad_token_id),
        "rejected_attention_mask": _pad_1d([x["rejected_attention_mask"] for x in batch], torch.long, 0),
    }


def rl_rollout_collate_fn(batch: list[dict[str, Any]], padding: PaddingConfig | None = None) -> dict[str, Tensor]:
    padding = padding or PaddingConfig()
    collated = {
        "prompts": _pad_1d([x["prompts"] for x in batch], torch.long, padding.pad_token_id),
        "prompt_attention_mask": _pad_1d([x["prompt_attention_mask"] for x in batch], torch.long, 0),
        "responses": _pad_1d([x["responses"] for x in batch], torch.long, padding.pad_token_id),
        "response_attention_mask": _pad_1d([x["response_attention_mask"] for x in batch], torch.long, 0),
    }

    optional_float_fields = ("advantages", "returns", "old_log_probs", "values")
    for name in optional_float_fields:
        if name in batch[0]:
            collated[name] = _pad_1d([x[name] for x in batch], torch.float32, 0.0)

    return collated


def create_dataloader(
    dataset: Any,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
