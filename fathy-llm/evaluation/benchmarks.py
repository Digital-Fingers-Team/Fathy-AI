"""Benchmark suite for Fathy LLM.

This module provides a modular ``FathyEvaluator`` that can run independent
benchmark tracks and emit both JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import torch

try:
    from datasets import Dataset, DatasetDict, load_dataset
except Exception:  # pragma: no cover - optional runtime dependency
    Dataset = Any  # type: ignore[assignment]
    DatasetDict = Any  # type: ignore[assignment]
    load_dataset = None

try:
    import sacrebleu
except Exception:  # pragma: no cover - optional runtime dependency
    sacrebleu = None


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    metrics: dict[str, float]
    details: dict[str, Any]


class FathyEvaluator:
    """Evaluation orchestrator with modular dataset loaders.

    The evaluator is intentionally framework-agnostic:
    - ``model`` can be a custom torch model returning ``{"logits": ...}`` or HF outputs.
    - ``tokenizer`` can be any object that supports ``__call__`` or ``encode``.
    - ``generate_fn`` can be supplied to fully customize text generation.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str | torch.device | None = None,
        quick: bool = False,
        quick_samples: int = 32,
        full_samples: int = 500,
        max_new_tokens: int = 64,
        generate_fn: Callable[[str, int], str] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.quick = quick
        self.quick_samples = quick_samples
        self.full_samples = full_samples
        self.max_new_tokens = max_new_tokens
        self.generate_fn = generate_fn

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    # -----------------------------
    # Generic helpers
    # -----------------------------
    def _sample_limit(self) -> int:
        return self.quick_samples if self.quick else self.full_samples

    def _require_datasets(self) -> None:
        if load_dataset is None:
            raise RuntimeError("datasets is required. Install with: pip install datasets")

    def _dataset_take(self, ds: Dataset, n: int | None = None) -> list[dict[str, Any]]:
        limit = n if n is not None else self._sample_limit()
        upper = min(limit, len(ds))
        return [ds[i] for i in range(upper)]

    def _load_first_available(
        self,
        candidates: Sequence[tuple[str, str | None]],
        *,
        split: str,
        **kwargs: Any,
    ) -> Dataset:
        self._require_datasets()
        last_error: Exception | None = None
        for name, config in candidates:
            try:
                if config is None:
                    return load_dataset(name, split=split, **kwargs)
                return load_dataset(name, config, split=split, **kwargs)
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
        raise RuntimeError(f"Could not load any dataset candidate: {candidates}") from last_error

    def _tokenize_text(self, text: str) -> torch.LongTensor:
        if hasattr(self.tokenizer, "__call__"):
            encoded = self.tokenizer(text, return_tensors="pt")
            if isinstance(encoded, dict):
                return encoded["input_ids"].to(self.device)
        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text)
            if isinstance(ids, list):
                return torch.tensor([ids], device=self.device, dtype=torch.long)
            if hasattr(ids, "ids"):
                return torch.tensor([ids.ids], device=self.device, dtype=torch.long)
        raise TypeError("Unsupported tokenizer interface for benchmarking")

    @torch.no_grad()
    def _next_token_nll(self, input_ids: torch.LongTensor) -> tuple[float, int]:
        if input_ids.shape[1] < 2:
            return 0.0, 0

        outputs = self.model(input_ids=input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        nll = -target_log_probs.sum().item()
        tokens = int(shift_labels.numel())
        return nll, tokens

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        gen_tokens = max_new_tokens or self.max_new_tokens
        if self.generate_fn is not None:
            return self.generate_fn(prompt, gen_tokens)

        input_ids = self._tokenize_text(prompt)
        if not hasattr(self.model, "generate"):
            raise TypeError("Model does not expose .generate and no generate_fn was provided")

        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=gen_tokens)

        if hasattr(self.tokenizer, "decode"):
            if isinstance(output_ids, torch.Tensor):
                seq = output_ids[0].detach().cpu().tolist()
            else:
                seq = output_ids[0]
            text = self.tokenizer.decode(seq)
            return text[len(prompt) :] if text.startswith(prompt) else text
        return ""

    # -----------------------------
    # Dataset loaders (modular)
    # -----------------------------
    def load_text_data(self, test_data_path: str | Path, split: str = "test") -> list[str]:
        path = Path(test_data_path)
        if path.suffix.lower() == ".jsonl":
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            texts = [row.get("text", "") for row in lines]
            return [t for t in texts if t]

        self._require_datasets()
        ds = load_dataset(str(path), split=split) if path.is_dir() else load_dataset("json", data_files=str(path), split=split)
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        return [row[text_col] for row in self._dataset_take(ds)]

    def load_arabic_hellaswag(self) -> list[dict[str, Any]]:
        candidates = [
            ("arbml/arabic_hellaswag", None),
            ("MBZUAI/ArabicHellaSwag", None),
            ("hellaswag", None),
        ]
        ds = self._load_first_available(candidates, split="validation")
        rows = self._dataset_take(ds)
        normalized = []
        for row in rows:
            endings = row.get("endings") or row.get("choices") or row.get("options") or []
            label = row.get("label")
            if isinstance(label, str) and label.isdigit():
                label = int(label)
            normalized.append(
                {
                    "context": row.get("ctx") or row.get("context") or row.get("prompt") or "",
                    "endings": endings,
                    "label": int(label) if label is not None else None,
                }
            )
        return normalized

    def load_tydiqa_arabic(self, split: str = "validation") -> list[dict[str, Any]]:
        ds = self._load_first_available(
            [("tydiqa", "secondary_task"), ("tydiqa", "primary_task")],
            split=split,
        )
        rows = []
        for row in self._dataset_take(ds):
            lang = row.get("language") or row.get("id", "")
            if isinstance(lang, str) and "arabic" not in lang.lower() and not lang.lower().startswith("ar"):
                continue
            answers = row.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else row.get("answer_text", [])
            rows.append(
                {
                    "context": row.get("context", ""),
                    "question": row.get("question", ""),
                    "answers": answer_texts,
                }
            )
            if len(rows) >= self._sample_limit():
                break
        return rows

    def load_flores_ar_en(self, split: str = "dev") -> list[dict[str, str]]:
        candidates = [
            ("facebook/flores", "eng_Latn-arb_Arab"),
            ("Muennighoff/flores200", "eng_Latn-arb_Arab"),
        ]
        ds = self._load_first_available(candidates, split=split)
        rows = self._dataset_take(ds)
        pairs: list[dict[str, str]] = []
        for row in rows:
            ar = row.get("sentence_arb_Arab") or row.get("arb_Arab") or row.get("target") or ""
            en = row.get("sentence_eng_Latn") or row.get("eng_Latn") or row.get("source") or ""
            if ar and en:
                pairs.append({"ar": ar, "en": en})
        return pairs

    # -----------------------------
    # Benchmarks
    # -----------------------------
    def eval_perplexity(self, test_data_path: str | Path, split: str = "test") -> BenchmarkResult:
        texts = self.load_text_data(test_data_path, split=split)
        total_nll = 0.0
        total_tokens = 0
        for text in texts[: self._sample_limit()]:
            input_ids = self._tokenize_text(text)
            nll, tokens = self._next_token_nll(input_ids)
            total_nll += nll
            total_tokens += tokens

        if total_tokens == 0:
            ppl = float("inf")
        else:
            ppl = math.exp(total_nll / total_tokens)

        return BenchmarkResult(
            name="perplexity",
            metrics={"perplexity": ppl, "avg_nll": (total_nll / max(total_tokens, 1))},
            details={"num_docs": min(len(texts), self._sample_limit()), "num_tokens": total_tokens},
        )

    def eval_arabic_hellaswag(self) -> BenchmarkResult:
        examples = self.load_arabic_hellaswag()
        correct = 0
        total = 0
        for ex in examples:
            context = ex["context"]
            endings = ex["endings"]
            if not endings or ex["label"] is None:
                continue

            scores: list[float] = []
            for ending in endings:
                ids = self._tokenize_text(f"{context} {ending}")
                nll, tokens = self._next_token_nll(ids)
                scores.append(-(nll / max(tokens, 1)))

            pred = max(range(len(scores)), key=lambda idx: scores[idx])
            correct += int(pred == ex["label"])
            total += 1

        acc = correct / total if total else 0.0
        return BenchmarkResult(
            name="arabic_hellaswag",
            metrics={"accuracy": acc},
            details={"evaluated": total},
        )

    def eval_arabic_qa(self) -> BenchmarkResult:
        examples = self.load_tydiqa_arabic()
        em_scores: list[float] = []
        for ex in examples:
            prompt = (
                "أجب عن السؤال التالي بالاعتماد على السياق.\\n"
                f"السياق: {ex['context']}\\n"
                f"السؤال: {ex['question']}\\n"
                "الإجابة:"
            )
            pred = self._generate(prompt).strip()
            pred_norm = " ".join(pred.lower().split())
            gt_norm = {" ".join(str(a).lower().split()) for a in ex["answers"] if str(a).strip()}
            em_scores.append(1.0 if pred_norm in gt_norm else 0.0)

        return BenchmarkResult(
            name="arabic_qa_tydiqa",
            metrics={"exact_match": mean(em_scores) if em_scores else 0.0},
            details={"evaluated": len(em_scores)},
        )

    def eval_translation(self) -> BenchmarkResult:
        if sacrebleu is None:
            raise RuntimeError("sacrebleu is required. Install with: pip install sacrebleu")

        pairs = self.load_flores_ar_en(split="dev")
        predictions: list[str] = []
        references: list[str] = []
        for pair in pairs:
            prompt = f"Translate to English:\nArabic: {pair['ar']}\nEnglish:"
            pred = self._generate(prompt).strip()
            predictions.append(pred)
            references.append(pair["en"])

        bleu = sacrebleu.corpus_bleu(predictions, [references]).score if predictions else 0.0
        chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score if predictions else 0.0

        return BenchmarkResult(
            name="translation_flores_ar_en",
            metrics={"bleu": bleu, "chrfpp": chrf},
            details={"evaluated": len(predictions)},
        )

    def eval_instruction_following(
        self,
        prompts: Sequence[str] | None = None,
        rubric_scorer: Callable[[str, str], float] | None = None,
    ) -> BenchmarkResult:
        if prompts is None:
            prompts = [
                "Explain blockchain in two short bullet points.",
                "اكتب رسالة اعتذار رسمية قصيرة باللغة العربية.",
                "Return ONLY valid JSON with keys name and age for a user named Laila, 24.",
            ]

        def default_scorer(prompt: str, output: str) -> float:
            if not output.strip():
                return 0.0
            score = 0.5
            if "only" in prompt.lower() and output.strip().startswith("{"):
                score += 0.2
            if len(output.split()) >= 5:
                score += 0.2
            if any(token in output.lower() for token in ["i cannot", "لا أستطيع"]):
                score -= 0.3
            return max(0.0, min(1.0, score))

        scorer = rubric_scorer or default_scorer
        rows = []
        for prompt in prompts[: self._sample_limit()]:
            completion = self._generate(prompt)
            rows.append(scorer(prompt, completion))

        return BenchmarkResult(
            name="instruction_following",
            metrics={"rubric_score": mean(rows) if rows else 0.0},
            details={"evaluated": len(rows), "scoring": "custom" if rubric_scorer else "default"},
        )

    def eval_arabic_culture(self) -> BenchmarkResult:
        qa_bank = [
            {"q": "ما هي عاصمة مصر؟", "a": ["القاهرة"]},
            {"q": "في أي بلد يقع جامع الزيتونة؟", "a": ["تونس", "في تونس"]},
            {"q": "من هو الشاعر الملقب بأمير الشعراء؟", "a": ["أحمد شوقي", "احمد شوقي"]},
        ]
        if self.quick:
            qa_bank = qa_bank[:2]

        scores: list[float] = []
        for item in qa_bank:
            pred = self._generate(f"أجب بإيجاز: {item['q']}").strip().lower()
            scores.append(1.0 if any(ans.lower() in pred for ans in item["a"]) else 0.0)

        return BenchmarkResult(
            name="arabic_culture",
            metrics={"accuracy": mean(scores) if scores else 0.0},
            details={"evaluated": len(scores)},
        )

    # -----------------------------
    # Suite + reporting
    # -----------------------------
    def run_all(self, test_data_path: str | Path, split: str = "test") -> dict[str, Any]:
        results = {
            "mode": "quick" if self.quick else "full",
            "benchmarks": {
                "perplexity": self.eval_perplexity(test_data_path, split=split).__dict__,
                "arabic_hellaswag": self.eval_arabic_hellaswag().__dict__,
                "arabic_qa": self.eval_arabic_qa().__dict__,
                "translation": self.eval_translation().__dict__,
                "instruction_following": self.eval_instruction_following().__dict__,
                "arabic_culture": self.eval_arabic_culture().__dict__,
            },
        }
        return results

    def generate_report(self, results: dict[str, Any], output_path: str | Path) -> dict[str, str]:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "benchmark_results.json"
        md_path = out_dir / "benchmark_report.md"

        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Fathy LLM Benchmark Report",
            "",
            f"Mode: **{results.get('mode', 'unknown')}**",
            "",
            "| Benchmark | Metric | Value |",
            "|---|---|---:|",
        ]
        for bench_name, payload in results.get("benchmarks", {}).items():
            metrics = payload.get("metrics", {})
            for metric_name, value in metrics.items():
                lines.append(f"| {bench_name} | {metric_name} | {value:.4f} |")

        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Fathy benchmarks")
    parser.add_argument("--test-data-path", required=True, help="Path used by perplexity evaluation")
    parser.add_argument("--split", default="test", help="Dataset split for perplexity")
    parser.add_argument("--output-path", default="./benchmark_outputs", help="Directory to save reports")
    parser.add_argument("--quick", action="store_true", help="Run quick tiny-subset benchmarks")
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    raise RuntimeError(
        "CLI requires model/tokenizer wiring from your runtime. "
        "Instantiate FathyEvaluator in code and call run_all() + generate_report()."
    )


if __name__ == "__main__":
    main()
