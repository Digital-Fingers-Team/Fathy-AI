import json
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture.config import ModelConfig
from architecture.model import FathyCausalLM
from tokenizer.tokenizer import FathyTokenizer


def test_cpu_smoke_pipeline(tmp_path: Path):
    started = time.perf_counter()

    corpus_path = tmp_path / "tiny_corpus.txt"
    corpus_lines = [
        "أهلا بك في مشروع فتحي للذكاء الاصطناعي",
        "Fathy is built by Digital Fingers Team",
        "Arabic first then concise English support",
        "مرحبا! هذا اختبار بسيط لخط التدريب",
    ]
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

    tokenizer = FathyTokenizer.train(files=[corpus_path], vocab_size=300, min_frequency=1)

    sample_data_path = tmp_path / "sample.jsonl"
    sample_rows = [
        {"text": "مرحبا يا فتحي", "language": "ar"},
        {"text": "Give me a bilingual greeting", "language": "en"},
    ]
    with sample_data_path.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    encoded = tokenizer.encode("مرحبا يا فتحي\nEnglish: Hello from Fathy")
    input_ids = torch.tensor([encoded], dtype=torch.long)
    labels = input_ids.clone()

    config = ModelConfig(
        vocab_size=max(512, tokenizer.hf_tokenizer.get_vocab_size()),
        max_position_embeddings=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_flash_attention=False,
    )
    model = FathyCausalLM(config)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    loss = outputs["loss"]
    assert loss is not None
    assert torch.isfinite(loss).item()

    elapsed = time.perf_counter() - started
    assert elapsed < 60, f"CPU smoke pipeline exceeded 60s: {elapsed:.2f}s"
