import html
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasketch import MinHash, MinHashLSH
from ftfy import fix_text
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Text cleaning, filtering, and deduplication for JSONL datasets."""

    ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
    TATWEEL = "ـ"
    ALEF_VARIANTS = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
    }

    def clean_arabic(self, text: str, strip_diacritics: bool = True, unicode_form: str = "NFKC") -> str:
        if not isinstance(text, str):
            return ""

        cleaned = unicodedata.normalize(unicode_form, text)
        for src, dest in self.ALEF_VARIANTS.items():
            cleaned = cleaned.replace(src, dest)
        cleaned = cleaned.replace(self.TATWEEL, "")
        if strip_diacritics:
            cleaned = self.ARABIC_DIACRITICS.sub("", cleaned)

        cleaned = re.sub(r"\s*([،؛؟!,:;.])\s*", r" \1 ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def clean_english(self, text: str, unicode_form: str = "NFC") -> str:
        if not isinstance(text, str):
            return ""

        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = html.unescape(cleaned)
        cleaned = fix_text(cleaned)
        cleaned = unicodedata.normalize(unicode_form, cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def quality_filter(
        self,
        text: str,
        min_length: int = 30,
        max_repetition_ratio: float = 0.30,
        min_unique_word_ratio: float = 0.25,
        max_symbol_number_ratio: float = 0.35,
        expected_language: Optional[str] = None,
        lang_confidence_threshold: float = 0.80,
    ) -> bool:
        if not text or len(text.strip()) < min_length:
            return False

        normalized = text.strip()
        char_counts: Dict[str, int] = {}
        for ch in normalized:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        max_char_ratio = max(char_counts.values()) / max(len(normalized), 1)
        if max_char_ratio > max_repetition_ratio:
            return False

        words = re.findall(r"\b\w+\b", normalized, flags=re.UNICODE)
        if not words:
            return False
        unique_word_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_word_ratio < min_unique_word_ratio:
            return False

        symbol_or_number = sum(1 for c in normalized if (not c.isalpha()) and (not c.isspace()))
        symbol_number_ratio = symbol_or_number / len(normalized)
        if symbol_number_ratio > max_symbol_number_ratio:
            return False

        if expected_language:
            try:
                predictions = detect_langs(normalized)
            except LangDetectException:
                return False
            lang_match = next((p for p in predictions if p.lang == expected_language), None)
            if not lang_match or lang_match.prob < lang_confidence_threshold:
                return False

        return True

    def deduplication(
        self,
        records: Iterable[Dict[str, Any]],
        threshold: float = 0.85,
        num_perm: int = 128,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        deduped: List[Dict[str, Any]] = []

        for idx, record in enumerate(records):
            text = record.get(text_key, "")
            if not text:
                continue

            mh = MinHash(num_perm=num_perm)
            for token in set(re.findall(r"\w+", text.lower(), flags=re.UNICODE)):
                mh.update(token.encode("utf-8"))

            key = str(record.get("id", f"record-{idx}"))
            if lsh.query(mh):
                continue

            lsh.insert(key, mh)
            deduped.append(record)

        return deduped

    def _safe_read_jsonl(self, file_path: Path) -> Iterable[Dict[str, Any]]:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at %s:%s", file_path, line_no)
                    continue
                if not isinstance(row, dict):
                    continue
                yield row

    def process_file(
        self,
        input_file: str,
        output_file: str,
        language: Optional[str] = None,
        apply_deduplication: bool = True,
    ) -> Dict[str, int]:
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        read_count = 0
        kept_records: List[Dict[str, Any]] = []

        for row in self._safe_read_jsonl(input_path):
            read_count += 1
            text = row.get("text", "")

            lang = language or row.get("language")
            if lang == "ar":
                text = self.clean_arabic(text)
            elif lang == "en":
                text = self.clean_english(text)
            else:
                text = self.clean_english(text)

            if not self.quality_filter(text, expected_language=lang if lang in {"ar", "en"} else None):
                continue

            row["text"] = text
            kept_records.append(row)

        if apply_deduplication:
            kept_records = self.deduplication(kept_records, threshold=0.85)

        with output_path.open("w", encoding="utf-8") as out:
            for row in kept_records:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "read": read_count,
            "written": len(kept_records),
            "filtered": max(read_count - len(kept_records), 0),
        }

    def process_all(
        self,
        input_dir: str,
        output_dir: str,
        apply_deduplication: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, int]] = {}
        for file_path in sorted(in_path.glob("*.jsonl")):
            target = out_path / file_path.name
            try:
                results[file_path.name] = self.process_file(
                    str(file_path),
                    str(target),
                    apply_deduplication=apply_deduplication,
                )
                logger.info("Processed %s -> %s", file_path.name, target.name)
            except Exception as exc:
                logger.exception("Failed processing %s: %s", file_path, exc)
                results[file_path.name] = {"read": 0, "written": 0, "filtered": 0}

        return results
