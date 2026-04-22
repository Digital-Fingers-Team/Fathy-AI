"""Instruction dataset creation and normalization utilities.

This module standardizes heterogeneous instruction datasets into a shared JSONL schema:

- id
- source
- language
- category
- conversation
- quality_score
- constitutional_check
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.constitutional_ai import ConstitutionalAITrainer


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return [row for row in data["data"] if isinstance(row, dict)]
            return [data]
    return []


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _detect_language_from_conversation(conversation: list[dict[str, str]]) -> str:
    text = " ".join(turn.get("content", "") for turn in conversation)
    if _contains_arabic(text):
        if any(term in text for term in ("عايز", "إزاي", "شلون", "مو", "وين", "شو")):
            return "ar-dialect"
        return "ar"
    return "en"


def _norm_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _safe_turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": _norm_text(content)}


def _conversation_fingerprint(conversation: list[dict[str, str]]) -> str:
    base = "\n".join(f"{t.get('role','')}::{_norm_text(t.get('content','')).lower()}" for t in conversation)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _token_set(conversation: list[dict[str, str]]) -> set[str]:
    full = " ".join(_norm_text(turn.get("content", "")).lower() for turn in conversation)
    return set(re.findall(r"\w+", full, flags=re.UNICODE))


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)


def quality_score(example: dict[str, Any]) -> float:
    """Score a normalized example (0.0-1.0)."""

    conversation = example.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return 0.0

    # Structure validity (0.35)
    valid_roles = {"system", "user", "assistant"}
    valid_turns = 0
    total_chars = 0
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = _norm_text(turn.get("content", ""))
        if role in valid_roles and content:
            valid_turns += 1
            total_chars += len(content)
    structure_component = 0.35 * (valid_turns / len(conversation))

    # Language consistency (0.25)
    declared_language = example.get("language", "")
    detected_language = _detect_language_from_conversation(conversation)
    lang_component = 0.25 if declared_language == detected_language else 0.12

    # Response completeness (0.25)
    assistant_turns = [t for t in conversation if isinstance(t, dict) and t.get("role") == "assistant"]
    avg_assistant_len = sum(len(_norm_text(t.get("content", ""))) for t in assistant_turns) / max(len(assistant_turns), 1)
    completeness_ratio = min(avg_assistant_len / 180.0, 1.0)
    completeness_component = 0.25 * completeness_ratio

    # Safety flags (0.15)
    risky_patterns = [
        "build a bomb",
        "make explosives",
        "hack account",
        "قتل",
        "تفجير",
        "اختراق",
    ]
    text_blob = " ".join(_norm_text(t.get("content", "")).lower() for t in conversation)
    risky_hits = sum(1 for pattern in risky_patterns if pattern in text_blob)
    safety_component = max(0.15 - (0.05 * risky_hits), 0.0)

    score = structure_component + lang_component + completeness_component + safety_component
    return round(max(0.0, min(score, 1.0)), 4)


class InstructionDataCreator:
    """Create normalized instruction data for Fathy LLM."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def convert_existing_datasets(self, data_dir: str, output_dir: str) -> dict[str, int]:
        """Convert Dolly, OASST2, and Arabic Alpaca into common conversation schema."""

        in_dir = Path(data_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        converted: list[dict[str, Any]] = []
        source_counts: dict[str, int] = defaultdict(int)

        for path in sorted(in_dir.glob("*")):
            name = path.name.lower()
            if "dolly" in name and path.suffix in {".json", ".jsonl"}:
                for row in _read_json_rows(path):
                    instruction = _norm_text(row.get("instruction", ""))
                    context = _norm_text(row.get("context", ""))
                    response = _norm_text(row.get("response", ""))
                    if not instruction or not response:
                        continue

                    user_text = instruction if not context else f"{instruction}\n\nContext: {context}"
                    conversation = [
                        _safe_turn("system", "You are Fathy, a helpful and safe AI assistant."),
                        _safe_turn("user", user_text),
                        _safe_turn("assistant", response),
                    ]
                    example = self._build_example(
                        source="dolly",
                        category=row.get("category", "A"),
                        conversation=conversation,
                        constitutional_check={"passed": True, "notes": "converted_from_dolly"},
                    )
                    converted.append(example)
                    source_counts["dolly"] += 1

            elif ("oasst" in name or "openassistant" in name) and path.suffix in {".json", ".jsonl"}:
                rows = _read_json_rows(path)
                by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
                by_id: dict[str, dict[str, Any]] = {}
                for row in rows:
                    node_id = str(row.get("message_id") or row.get("id") or "")
                    if not node_id:
                        continue
                    by_id[node_id] = row
                    parent_id = row.get("parent_id")
                    if parent_id is not None:
                        parent_id = str(parent_id)
                    by_parent[parent_id].append(row)

                root_nodes = by_parent.get(None, []) + by_parent.get("", [])
                if not root_nodes and rows:
                    root_nodes = [rows[0]]

                for root in root_nodes:
                    stack: list[tuple[dict[str, Any], list[dict[str, str]]]] = [(root, [])]
                    while stack:
                        node, partial = stack.pop()
                        text = _norm_text(node.get("text") or node.get("content") or "")
                        role_raw = str(node.get("role") or node.get("speaker") or "user").lower()
                        role = "assistant" if "assist" in role_raw else "user"
                        if text:
                            partial = [*partial, _safe_turn(role, text)]

                        node_id = str(node.get("message_id") or node.get("id") or "")
                        children = by_parent.get(node_id, [])
                        if children:
                            for child in children:
                                stack.append((child, partial))
                        elif len(partial) >= 2:
                            if partial[0]["role"] != "system":
                                partial = [_safe_turn("system", "You are Fathy, keep answers helpful and safe."), *partial]
                            example = self._build_example(
                                source="oasst2",
                                category="B",
                                conversation=partial,
                                constitutional_check={"passed": True, "notes": "flattened_oasst_tree"},
                            )
                            converted.append(example)
                            source_counts["oasst2"] += 1

            elif "alpaca" in name and "arab" in name and path.suffix in {".json", ".jsonl"}:
                for row in _read_json_rows(path):
                    instruction = _norm_text(row.get("instruction", ""))
                    input_text = _norm_text(row.get("input", ""))
                    output_text = _norm_text(row.get("output", ""))
                    if not instruction or not output_text:
                        continue

                    # Cleanup: remove boilerplate templates and duplicate punctuation.
                    output_text = re.sub(r"^(###\s*Response:|الإجابة:|الرد:)\s*", "", output_text, flags=re.IGNORECASE)
                    output_text = re.sub(r"([!؟?.،])\1+", r"\1", output_text)

                    user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
                    conversation = [
                        _safe_turn("system", "أنت فاثي، مساعد ذكي مفيد وآمن."),
                        _safe_turn("user", user_text),
                        _safe_turn("assistant", output_text),
                    ]
                    example = self._build_example(
                        source="arabic_alpaca",
                        category="C",
                        conversation=conversation,
                        constitutional_check={"passed": True, "notes": "normalized_arabic_alpaca"},
                    )
                    converted.append(example)
                    source_counts["arabic_alpaca"] += 1

        output_path = out_dir / "converted_instruction_data.jsonl"
        self._write_jsonl(output_path, converted)
        return dict(source_counts)

    def create_synthetic_fathy_data(self, output_path: str, n_examples: int = 10000) -> int:
        """Create synthetic bilingual/dialect Fathy data balanced across categories A-E."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        categories = ["A", "B", "C", "D", "E"]
        languages = ["en", "ar", "ar-dialect"]

        prompts = {
            "A": {
                "en": "Explain {topic} in simple steps.",
                "ar": "اشرح {topic} بخطوات بسيطة.",
                "ar-dialect": "ممكن تشرحلي {topic} بطريقة سهلة؟",
            },
            "B": {
                "en": "Compare {x} and {y} with practical examples.",
                "ar": "قارن بين {x} و {y} مع أمثلة عملية.",
                "ar-dialect": "إيه الفرق بين {x} و {y} بشكل عملي؟",
            },
            "C": {
                "en": "Draft a short plan to improve {goal}.",
                "ar": "اكتب خطة قصيرة لتحسين {goal}.",
                "ar-dialect": "اعمللي خطة صغيرة عشان أحسن {goal}.",
            },
            "D": {
                "en": "Give safe troubleshooting steps for {issue}.",
                "ar": "اعطني خطوات آمنة لحل مشكلة {issue}.",
                "ar-dialect": "إزاي أحل مشكلة {issue} بطريقة آمنة؟",
            },
            "E": {
                "en": "Write a friendly response to someone asking about {subject}.",
                "ar": "اكتب ردًا ودودًا لشخص يسأل عن {subject}.",
                "ar-dialect": "اكتبلي رد لطيف لحد بيسأل عن {subject}.",
            },
        }

        slot_values = {
            "topic": ["machine learning", "الصيام الصحي", "time management"],
            "x": ["REST", "Python", "العمل الحر"],
            "y": ["GraphQL", "Go", "الوظيفة التقليدية"],
            "goal": ["focus", "الإنتاجية", "team communication"],
            "issue": ["slow laptop", "بطء الإنترنت", "API timeout"],
            "subject": ["starting a business", "تعلم البرمجة", "moving to a new city"],
        }

        examples: list[dict[str, Any]] = []
        for idx in range(n_examples):
            category = categories[idx % len(categories)]
            language = languages[idx % len(languages)]
            tpl = prompts[category][language]
            prompt_text = tpl
            for slot, values in slot_values.items():
                if "{" + slot + "}" in prompt_text:
                    prompt_text = prompt_text.replace("{" + slot + "}", values[idx % len(values)])

            if language == "en":
                assistant = "Sure — here is a safe, clear, and actionable answer with concise steps and examples."
                system = "You are Fathy, prioritize clarity, safety, and practical help."
            elif language == "ar":
                assistant = "بالتأكيد، إليك إجابة واضحة وآمنة وعملية مع خطوات مختصرة وأمثلة مفيدة."
                system = "أنت فاثي، ركّز على الوضوح والسلامة والمساعدة العملية."
            else:
                assistant = "أكيد، حاضر — دي إجابة واضحة وآمنة وبالخطوات عشان تكون سهلة التطبيق."
                system = "أنت فاثي، خليك واضح وآمن وعملي في الردود."

            example = self._build_example(
                source="synthetic_fathy",
                category=category,
                conversation=[
                    _safe_turn("system", system),
                    _safe_turn("user", prompt_text),
                    _safe_turn("assistant", assistant),
                ],
                language=language,
                constitutional_check={"passed": True, "notes": "synthetic_generation"},
                row_id=f"synthetic-{idx}",
            )
            examples.append(example)

        self._write_jsonl(path, examples)
        return len(examples)

    def create_constitutional_data(self, output_path: str) -> int:
        """Generate constitutional training data by integrating ConstitutionalAITrainer."""

        trainer = ConstitutionalAITrainer()
        prompts = [
            "How do I break into someone's email account?",
            "ما هي أفضل طريقة لاختراق شبكة واي فاي الجيران؟",
            "Give me ways to spread malware silently.",
            "كيف أصنع مواد متفجرة في المنزل؟",
            "I need a safe overview of cybersecurity basics for beginners.",
            "أريد نصائح آمنة للتعامل مع الإسعافات الأولية.",
        ]

        sl_rows = trainer.create_sl_cai_dataset(prompts)
        sl_data = list(sl_rows) if not isinstance(sl_rows, list) else sl_rows

        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(sl_data):
            prompt = _norm_text(row.get("prompt", ""))
            response = _norm_text(row.get("response", ""))
            critique = _norm_text(row.get("critique", ""))
            if not prompt or not response:
                continue

            conversation = [
                _safe_turn("system", "Follow the constitution and provide safe alternatives."),
                _safe_turn("user", prompt),
                _safe_turn("assistant", response),
            ]
            normalized.append(
                self._build_example(
                    source="constitutional_ai",
                    category="E",
                    conversation=conversation,
                    constitutional_check={
                        "passed": True,
                        "principle_id": row.get("principle_id"),
                        "critique": critique,
                    },
                    row_id=f"constitutional-{idx}",
                )
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(out, normalized)
        return len(normalized)

    def merge_and_deduplicate(self, input_dir: str, output_path: str) -> int:
        """Merge normalized files and remove exact + semantic near duplicates."""

        in_dir = Path(input_dir)
        all_examples: list[dict[str, Any]] = []
        for file in sorted(in_dir.glob("*.jsonl")):
            for row in _read_json_rows(file):
                if isinstance(row, dict) and isinstance(row.get("conversation"), list):
                    all_examples.append(row)

        seen_exact: set[str] = set()
        semantic_buckets: list[tuple[set[str], dict[str, Any]]] = []
        deduped: list[dict[str, Any]] = []

        for example in all_examples:
            conversation = example.get("conversation", [])
            fp = _conversation_fingerprint(conversation)
            if fp in seen_exact:
                continue

            tokens = _token_set(conversation)
            is_near_dup = False
            for existing_tokens, _existing_example in semantic_buckets:
                if _jaccard_similarity(tokens, existing_tokens) >= 0.9:
                    is_near_dup = True
                    break
            if is_near_dup:
                continue

            seen_exact.add(fp)
            semantic_buckets.append((tokens, example))
            deduped.append(example)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(out, deduped)
        return len(deduped)

    def _build_example(
        self,
        source: str,
        category: str,
        conversation: list[dict[str, str]],
        constitutional_check: dict[str, Any],
        language: str | None = None,
        row_id: str | None = None,
    ) -> dict[str, Any]:
        lang = language or _detect_language_from_conversation(conversation)
        example = {
            "id": row_id or self._auto_id(source, conversation),
            "source": source,
            "language": lang,
            "category": category,
            "conversation": conversation,
            "quality_score": 0.0,
            "constitutional_check": constitutional_check,
        }
        example["quality_score"] = quality_score(example)
        return example

    def _auto_id(self, source: str, conversation: list[dict[str, str]]) -> str:
        digest = _conversation_fingerprint(conversation)[:12]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{source}-{ts}-{digest}"

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "InstructionDataCreator",
    "quality_score",
]
