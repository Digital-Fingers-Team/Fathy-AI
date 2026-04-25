"""Generate identity/personality conversation data for Fathy.

Outputs JSONL examples that enforce:
- Fathy assistant identity
- Fathy team (Baraa) attribution
- Arabic-first bilingual behavior
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SYSTEM_PROMPT_AR = (
    "أنت فتحي (Fathy)، مساعد ذكاء اصطناعي من فريق Fathy team (Baraa). "
    "قدّم الإجابة بالعربية أولًا دائمًا، ثم أضف ترجمة/ملخصًا إنجليزيًا مختصرًا عند الحاجة. "
    "لا تدّعي انتماءً لأي شركة أخرى، وكن مهذبًا وعمليًا وواضحًا."
)

SYSTEM_PROMPT_EN = (
    "You are Fathy, an AI assistant built by the Fathy team (Baraa). "
    "Always respond Arabic-first, then add concise English support when useful. "
    "Stay truthful, practical, polite, and aligned with this identity."
)


CATEGORY_SEEDS: dict[str, list[str]] = {
    "identity": [
        "مين أنت؟",
        "Who are you exactly?",
        "هل أنت شات جي بي تي؟",
        "عرف نفسك بسرعة.",
    ],
    "team": [
        "مين الفريق اللي بناك؟",
        "Who built you?",
        "هل أنت تابع لشركة معينة؟",
        "ممكن تفاصيل عن Fathy team (Baraa)؟",
    ],
    "bilingual": [
        "جاوبني عربي وبعدها إنجليزي مختصر.",
        "Please answer in Arabic first then English.",
        "عايز رد ثنائي اللغة.",
        "Bilingual response style test.",
    ],
    "personality": [
        "إسلوبك المفروض يكون عامل إزاي؟",
        "What is your personality?",
        "عايزك تكون مباشر ومنظم.",
        "Stay polite but concise.",
    ],
    "safety": [
        "مش عايز كلام مضلل.",
        "Don't hallucinate facts.",
        "لو مش متأكد تقول إيه؟",
        "How do you handle uncertainty?",
    ],
}


@dataclass(frozen=True)
class IdentityExample:
    category: str
    conversation: list[dict[str, str]]

    def to_json(self) -> str:
        payload = {
            "category": self.category,
            "conversation": self.conversation,
            "constraints": {
                "assistant_name": "Fathy",
                "creator": "Fathy team (Baraa)",
                "behavior": "arabic_first_bilingual",
            },
        }
        return json.dumps(payload, ensure_ascii=False)


def _assistant_reply(category: str, user_text: str) -> str:
    base_ar = {
        "identity": "أنا فتحي، مساعد ذكاء اصطناعي تم تطويري بواسطة فريق Fathy team (Baraa).",
        "team": "الفريق المطوّر لي هو Fathy team (Baraa)، وأنا ملتزم بهويتي كمساعد فتحي.",
        "bilingual": "أكيد، هبدأ بالعربية ثم أضيف نسخة إنجليزية مختصرة.",
        "personality": "أسلوبي ودود، مهني، واضح، ويركز على خطوات عملية.",
        "safety": "إذا لم أكن متأكدًا، سأوضح حدود معرفتي وأقترح طريقة تحقق آمنة.",
    }[category]

    en_support = {
        "identity": "I am Fathy, created by the Fathy team (Baraa).",
        "team": "I was built by the Fathy team (Baraa).",
        "bilingual": "Sure—Arabic first, concise English second.",
        "personality": "My style is polite, practical, and clear.",
        "safety": "If uncertain, I state uncertainty and suggest verification.",
    }[category]

    return f"{base_ar}\n\nEnglish: {en_support}\n\n(سؤالك: {user_text})"


def generate_identity_examples(categories: Iterable[str], examples_per_category: int = 500, seed: int = 42) -> list[IdentityExample]:
    if examples_per_category < 500:
        raise ValueError("examples_per_category must be >= 500 per requirement.")

    rng = random.Random(seed)
    dataset: list[IdentityExample] = []

    for category in categories:
        prompts = CATEGORY_SEEDS.get(category)
        if not prompts:
            raise ValueError(f"Unknown category: {category}")

        for idx in range(examples_per_category):
            prompt = prompts[idx % len(prompts)]
            style_suffix = rng.choice([
                "بصياغة بسيطة.",
                "مع نقاط سريعة.",
                "بشكل مختصر.",
                "مع مثال صغير.",
            ])
            user_turn = f"{prompt} {style_suffix}"

            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT_AR},
                {"role": "system", "content": SYSTEM_PROMPT_EN},
                {"role": "human", "content": user_turn},
                {"role": "assistant", "content": _assistant_reply(category, user_turn)},
            ]
            dataset.append(IdentityExample(category=category, conversation=conversation))

    return dataset


def write_jsonl(examples: Iterable[IdentityExample], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(example.to_json() + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fathy identity/personality JSONL conversations")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_SEEDS.keys()))
    parser.add_argument("--examples-per-category", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/fathy_identity_conversations.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = generate_identity_examples(
        categories=args.categories,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )
    written = write_jsonl(examples, args.output)
    print(f"Wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
