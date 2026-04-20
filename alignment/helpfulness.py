"""Reusable helpfulness heuristics for constitutional alignment."""

from __future__ import annotations

from alignment.constitution import ConstitutionalPrinciple, THE_CONSTITUTION


def helpfulness_principles() -> list[ConstitutionalPrinciple]:
    """Return principles used for helpfulness evaluation."""

    return [p for p in THE_CONSTITUTION if p.category == "helpfulness"]


def evaluate_helpfulness(prompt: str, response: str) -> dict[str, str | int]:
    """Deterministic helpfulness score for dataset metadata annotation."""

    score = 0
    lowered = response.lower()

    if response.strip():
        score += 1
    if len(response.split()) >= 12:
        score += 1
    if any(token in lowered for token in ("because", "for example", "خطوات", "مثال")):
        score += 1
    if any(token in lowered for token in ("i'm not sure", "i don't know", "غير متأكد")):
        # Honest uncertainty still counts as helpful when explicit.
        score += 1

    if score >= 3:
        label = "high"
    elif score == 2:
        label = "medium"
    else:
        label = "low"

    return {
        "prompt": prompt,
        "helpfulness_score": score,
        "helpfulness_label": label,
    }


__all__ = ["helpfulness_principles", "evaluate_helpfulness"]
