from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def _build_test_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        vocab_size=256,
    )
    corpus = [
        "مرحبًا بك في مشروع فتحي للذكاء الاصطناعي",
        "هذا اختبار بسيط لسلامة الترميز وفك الترميز",
        "Arabic tokenizer round trip sanity check",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def test_special_tokens_exist():
    tok = _build_test_tokenizer()
    vocab = tok.get_vocab()
    for special in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        assert special in vocab, f"Missing special token: {special}"


def test_arabic_round_trip_sanity():
    tok = _build_test_tokenizer()
    arabic_text = "مرحبًا بك في مشروع فتحي للذكاء الاصطناعي"

    encoded = tok.encode(arabic_text)
    decoded = tok.decode(encoded.ids)

    assert len(encoded.ids) > 0
    assert "فتحي" in decoded
    assert "الذكاء" in decoded
