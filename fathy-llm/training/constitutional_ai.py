"""Constitutional AI dataset generation utilities.

Produces datasets consumable by:
- Supervised fine-tuning (SFT): rows with prompt/response (+ metadata).
- Reward/preference training: rows with prompt/chosen/rejected (+ metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alignment.constitution import ConstitutionalPrinciple, THE_CONSTITUTION as _SHARED_CONSTITUTION
from alignment.helpfulness import evaluate_helpfulness
from alignment.harmlessness import evaluate_harmlessness, is_potentially_harmful, safe_refusal

# Re-export the shared bilingual constitution from a single source of truth.
THE_CONSTITUTION: tuple[ConstitutionalPrinciple, ...] = _SHARED_CONSTITUTION


def _maybe_to_hf_dataset(rows: list[dict[str, Any]]):
    """Return `datasets.Dataset` when available, otherwise raw rows."""

    try:
        from datasets import Dataset

        return Dataset.from_list(rows)
    except Exception:
        return rows


def _detect_language(text: str) -> str:
    return "ar" if any("\u0600" <= ch <= "\u06FF" for ch in text) else "en"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _principle_text(principle: ConstitutionalPrinciple) -> str:
    return f"EN: {principle.english}\nAR: {principle.arabic}"


@dataclass(slots=True)
class ConstitutionalAITrainer:
    """Self-critique + revision pipeline for constitutional alignment datasets."""

    constitution: tuple[ConstitutionalPrinciple, ...] = THE_CONSTITUTION

    def generate_critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Generate a critique for `response` under one constitutional principle."""

        harmful_prompt = is_potentially_harmful(prompt)
        harmful_response = is_potentially_harmful(response)

        findings: list[str] = []
        if harmful_response:
            findings.append(
                "Response appears to contain harmful/operational details and should be rewritten."
            )
        if harmful_prompt and "can’t help" not in response.lower() and "cannot help" not in response.lower():
            findings.append(
                "Prompt is harmful but response is missing a direct refusal with safe alternatives."
            )
        if not findings:
            findings.append(
                "No direct violation detected; improve clarity, honesty, and constructive guidance."
            )

        return (
            f"Principle ({principle.id}):\n{_principle_text(principle)}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Response under review:\n{response}\n\n"
            f"Critique:\n- " + "\n- ".join(findings)
        )

    def generate_revision(self, prompt: str, response: str, critique: str) -> str:
        """Generate a revised response using the critique."""

        if is_potentially_harmful(prompt):
            return safe_refusal(prompt)

        revised = response.strip() or "I can help with a clear and safe answer."

        if "safe" not in revised.lower() and "آمن" not in revised:
            revised += (
                "\n\nIf you'd like, I can provide a safer step-by-step answer with examples."
                "\nإذا رغبت، يمكنني تقديم إجابة أكثر أمانًا خطوة بخطوة مع أمثلة."
            )

        if "No direct violation detected" not in critique:
            revised = (
                "I can’t assist with harmful or illegal details. "
                "Here is a safe alternative:\n" + revised
            )

        return revised

    def create_sl_cai_dataset(self, harmful_prompts: list[str]):
        """Create SFT-style constitutional AI dataset.

        Output columns:
        - prompt
        - response
        - critique
        - principle_id
        - language
        - source
        - metadata (dict)
        """

        rows: list[dict[str, Any]] = []
        created_at = _now_iso()

        for prompt in harmful_prompts:
            candidate = (
                "I can provide general educational context, but I will avoid harmful instructions."
            )

            for principle in self.constitution:
                critique = self.generate_critique(prompt, candidate, principle)
                revised = self.generate_revision(prompt, candidate, critique)
                rows.append(
                    {
                        "prompt": prompt,
                        "response": revised,
                        "critique": critique,
                        "principle_id": principle.id,
                        "language": _detect_language(prompt),
                        "source": "constitutional_ai_sl",
                        "metadata": {
                            "created_at": created_at,
                            "task": "sft",
                            "principle": principle.to_dict(),
                            "helpfulness": evaluate_helpfulness(prompt, revised),
                            "harmlessness": evaluate_harmlessness(prompt, revised),
                        },
                    }
                )

        return _maybe_to_hf_dataset(rows)

    def create_preference_dataset(self, prompts: list[str]):
        """Create preference dataset consumable by reward model trainers.

        Output columns:
        - prompt
        - chosen
        - rejected
        - language
        - source
        - metadata (dict)
        """

        rows: list[dict[str, Any]] = []
        created_at = _now_iso()

        for prompt in prompts:
            harmful = is_potentially_harmful(prompt)
            rejected = (
                "Sure, here are direct instructions and exact steps."
                if harmful
                else "Here is a vague and incomplete answer."
            )
            candidate = (
                "I can’t help with harmful instructions, but I can give safe alternatives "
                "and prevention guidance."
                if harmful
                else "Here is a complete and practical answer that stays safe and clear."
            )

            critique = self.generate_critique(prompt, rejected, self.constitution[0])
            chosen = self.generate_revision(prompt, candidate, critique)

            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "language": _detect_language(prompt),
                    "source": "constitutional_ai_preference",
                    "metadata": {
                        "created_at": created_at,
                        "task": "preference",
                        "chosen_helpfulness": evaluate_helpfulness(prompt, chosen),
                        "chosen_harmlessness": evaluate_harmlessness(prompt, chosen),
                        "rejected_harmlessness": evaluate_harmlessness(prompt, rejected),
                    },
                }
            )

        return _maybe_to_hf_dataset(rows)


__all__ = ["THE_CONSTITUTION", "ConstitutionalAITrainer"]
