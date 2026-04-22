"""Train Fathy tokenizer from preprocessed JSONL corpora.

Builds temporary plain-text shards sampled from Arabic/English/code corpora,
trains ``FathyTokenizer``, exports tokenizer artifacts, and runs post-training
fragmentation checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer import (  # pylint: disable=wrong-import-position
    ASSISTANT,
    BEGIN_OF_TEXT,
    END_OF_TEXT,
    END_TURN,
    HUMAN,
    SYSTEM,
    UNK,
    FathyTokenizer,
)

logger = logging.getLogger(__name__)

MAX_CORPUS_BYTES = 10 * 1024**3
QUICK_SAMPLE_RATIO = 0.01
DEFAULT_MIN_TEXT_LEN = 20

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CODE_HINT_RE = re.compile(r"[{}();<>]|\b(?:def|class|import|return|function|const|let|var|if|else|for|while|SELECT|FROM)\b")


@dataclass(frozen=True)
class CorpusMix:
    arabic: float
    english: float
    code: float


@dataclass
class SamplingStats:
    bytes_written: int = 0
    lines_written: int = 0
    per_bucket_lines: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.per_bucket_lines is None:
            self.per_bucket_lines = {"arabic": 0, "english": 0, "code": 0}


def _normalize_mix(arabic_ratio: float, english_ratio: float, code_ratio: float) -> CorpusMix:
    ratios = [arabic_ratio, english_ratio, code_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("Ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one ratio must be > 0.")
    return CorpusMix(
        arabic=arabic_ratio / total,
        english=english_ratio / total,
        code=code_ratio / total,
    )


def _classify_record(text: str, row: dict[str, object], source_name: str) -> str:
    lang = str(row.get("language", "")).strip().lower()
    source_hint = f"{source_name} {row.get('source', '')}".lower()

    if lang.startswith("ar"):
        return "arabic"
    if lang.startswith("en"):
        if CODE_HINT_RE.search(text) or "code" in source_hint or "github" in source_hint:
            return "code"
        return "english"

    if CODE_HINT_RE.search(text) or any(tag in source_hint for tag in ("code", "program", "github", "stack")):
        return "code"
    if ARABIC_RE.search(text):
        return "arabic"
    return "english"


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON in %s:%s", path, line_no)
                continue
            if isinstance(row, dict):
                yield row


def _iter_bucketed_texts(data_dir: Path, quick: bool) -> dict[str, Iterator[str]]:
    seed = 13

    def _iterator(bucket: str) -> Iterator[str]:
        rng = random.Random(seed + hash(bucket) % 10_000)
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for row in _iter_jsonl_rows(file_path):
                text = str(row.get("text", "")).strip()
                if len(text) < DEFAULT_MIN_TEXT_LEN:
                    continue
                if quick and rng.random() > QUICK_SAMPLE_RATIO:
                    continue
                if _classify_record(text, row, file_path.name) == bucket:
                    yield text

    return {
        "arabic": _iterator("arabic"),
        "english": _iterator("english"),
        "code": _iterator("code"),
    }


def _weighted_bucket_order(mix: CorpusMix, steps: int = 10_000) -> list[str]:
    targets = {
        "arabic": mix.arabic,
        "english": mix.english,
        "code": mix.code,
    }
    counts = {key: 0 for key in targets}
    order: list[str] = []

    for _ in range(steps):
        bucket = max(targets, key=lambda key: targets[key] - (counts[key] / max(len(order), 1)))
        counts[bucket] += 1
        order.append(bucket)

    return order


def _write_training_shards(
    data_dir: Path,
    temp_dir: Path,
    mix: CorpusMix,
    quick: bool,
    max_corpus_bytes: int,
) -> tuple[list[Path], SamplingStats]:
    bucket_iters = _iter_bucketed_texts(data_dir, quick=quick)
    order = _weighted_bucket_order(mix)

    shard_paths: list[Path] = []
    stats = SamplingStats()
    shard_idx = 0
    shard_size_limit = 256 * 1024 * 1024
    current_path = temp_dir / f"train_shard_{shard_idx:04d}.txt"
    current_file = current_path.open("w", encoding="utf-8")
    shard_paths.append(current_path)

    exhausted = set()
    order_idx = 0

    while stats.bytes_written < max_corpus_bytes:
        if len(exhausted) == len(bucket_iters):
            break

        bucket = order[order_idx % len(order)]
        order_idx += 1
        if bucket in exhausted:
            continue

        try:
            text = next(bucket_iters[bucket])
        except StopIteration:
            exhausted.add(bucket)
            continue

        line = f"{text}\n"
        line_bytes = len(line.encode("utf-8"))
        if stats.bytes_written + line_bytes > max_corpus_bytes:
            break

        current_file.write(line)
        stats.bytes_written += line_bytes
        stats.lines_written += 1
        stats.per_bucket_lines[bucket] += 1

        if current_path.stat().st_size >= shard_size_limit:
            current_file.close()
            shard_idx += 1
            current_path = temp_dir / f"train_shard_{shard_idx:04d}.txt"
            current_file = current_path.open("w", encoding="utf-8")
            shard_paths.append(current_path)

    current_file.close()

    if shard_paths and shard_paths[-1].stat().st_size == 0:
        shard_paths[-1].unlink()
        shard_paths.pop()

    if not shard_paths:
        raise RuntimeError("No training shards were produced. Check input data_dir and filters.")

    return shard_paths, stats


def _save_tokenizer_metadata(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": BEGIN_OF_TEXT,
        "eos_token": END_OF_TEXT,
        "model_max_length": 32768,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": UNK,
    }
    special_tokens_map = {
        "unk_token": UNK,
        "bos_token": BEGIN_OF_TEXT,
        "eos_token": END_OF_TEXT,
        "additional_special_tokens": [HUMAN, ASSISTANT, SYSTEM, END_TURN],
    }

    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fragmentation_ratio(tokenizer: FathyTokenizer, text: str) -> float:
    words = max(len(text.split()), 1)
    token_count = len(tokenizer.hf_tokenizer.encode(text, add_special_tokens=False).ids)
    return token_count / words


def _post_training_checks(tokenizer: FathyTokenizer) -> None:
    thresholds = {
        "مرحبا": 2.2,
        "hello": 1.8,
        "ذهب": 2.5,
        "يقول": 2.5,
        "كان": 2.5,
        "يريد": 2.5,
    }

    for probe, max_ratio in thresholds.items():
        ratio = _fragmentation_ratio(tokenizer, probe)
        if ratio > max_ratio:
            logger.warning(
                "Fragmentation warning for '%s': ratio=%.2f exceeds threshold=%.2f",
                probe,
                ratio,
                max_ratio,
            )
        else:
            logger.info("Fragmentation check passed for '%s' (%.2f <= %.2f)", probe, ratio, max_ratio)


def train_tokenizer(
    data_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int = 100_000,
    arabic_ratio: float = 0.4,
    english_ratio: float = 0.4,
    code_ratio: float = 0.2,
    *,
    quick: bool = False,
) -> Path:
    """Train tokenizer from preprocessed JSONL data and save artifacts.

    Args:
        data_dir: Directory of preprocessed JSONL files.
        output_dir: Base output directory to store tokenizer artifacts.
        vocab_size: Target tokenizer vocabulary size.
        arabic_ratio: Target Arabic corpus fraction.
        english_ratio: Target English corpus fraction.
        code_ratio: Target code corpus fraction.
        quick: If True, sample roughly 1% of records for faster iteration.

    Returns:
        Path to final tokenizer output directory.
    """

    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_path}")

    base_output = Path(output_dir)
    final_output = base_output / "fathy_tokenizer"
    mix = _normalize_mix(arabic_ratio, english_ratio, code_ratio)
    max_bytes = int(MAX_CORPUS_BYTES * (QUICK_SAMPLE_RATIO if quick else 1.0))

    with tempfile.TemporaryDirectory(prefix="fathy_tok_", dir=base_output) as temp_raw:
        temp_dir = Path(temp_raw)
        shard_paths, stats = _write_training_shards(
            data_dir=data_path,
            temp_dir=temp_dir,
            mix=mix,
            quick=quick,
            max_corpus_bytes=max_bytes,
        )

        logger.info(
            "Prepared %s shards (%s lines, %.2f GB). Bucket lines=%s",
            len(shard_paths),
            stats.lines_written,
            stats.bytes_written / (1024**3),
            stats.per_bucket_lines,
        )

        tokenizer = FathyTokenizer.train(
            files=shard_paths,
            vocab_size=vocab_size,
        )

        final_output.mkdir(parents=True, exist_ok=True)
        tokenizer.save(final_output / "tokenizer.json")
        _save_tokenizer_metadata(final_output)
        _post_training_checks(tokenizer)

    return final_output


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fathy tokenizer from preprocessed JSONL corpora")
    parser.add_argument("--data-dir", required=True, help="Directory containing preprocessed JSONL files")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "tokenizer"),
        help="Output base directory (tokenizer will be saved to output_dir/fathy_tokenizer)",
    )
    parser.add_argument("--vocab-size", type=int, default=100_000, help="Tokenizer vocabulary size")
    parser.add_argument("--arabic-ratio", type=float, default=0.4, help="Target Arabic sample ratio")
    parser.add_argument("--english-ratio", type=float, default=0.4, help="Target English sample ratio")
    parser.add_argument("--code-ratio", type=float, default=0.2, help="Target code sample ratio")
    parser.add_argument("--quick", action="store_true", help="Use only ~1%% of data and 1%% size cap")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    output_path = train_tokenizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        arabic_ratio=args.arabic_ratio,
        english_ratio=args.english_ratio,
        code_ratio=args.code_ratio,
        quick=args.quick,
    )
    logger.info("Tokenizer artifacts saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
