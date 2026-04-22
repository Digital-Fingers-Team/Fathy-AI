import importlib

import pytest
import torch


MODULE_CANDIDATES = [
    "architecture.attention",
]


def _load_gqa_class():
    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        cls = getattr(module, "GroupedQueryAttention", None)
        if cls is not None:
            return cls
    pytest.fail(
        "GroupedQueryAttention not found. Expected one of "
        f"{MODULE_CANDIDATES} to expose GroupedQueryAttention."
    )


@pytest.fixture(scope="module")
def gqa_module():
    torch.manual_seed(0)
    GQA = _load_gqa_class()
    from architecture.config import ModelConfig
    return GQA(
        ModelConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
            use_flash_attention=False,
        )
    )


def test_gqa_output_shape(gqa_module):
    x = torch.randn(2, 16, 128)
    out, _ = gqa_module(x, use_cache=False)
    assert out.shape == x.shape


def test_gqa_causal_behavior(gqa_module):
    x = torch.randn(1, 8, 128)
    baseline, _ = gqa_module(x, use_cache=False)

    x_modified = x.clone()
    x_modified[:, -1, :] += 100.0  # perturb future token heavily
    modified, _ = gqa_module(x_modified, use_cache=False)

    # Causal masking should prevent token 0 from changing due to token 7 edits.
    assert torch.allclose(baseline[:, 0, :], modified[:, 0, :], atol=1e-5, rtol=1e-4)


def test_gqa_kv_cache_consistency(gqa_module):
    x = torch.randn(1, 10, 128)

    full_out, _ = gqa_module(x, use_cache=False)

    pref_out, past = gqa_module(x[:, :6, :], use_cache=True)
    step_out, past = gqa_module(x[:, 6:, :], use_cache=True, past_key_value=past)
    cat_out = torch.cat([pref_out, step_out], dim=1)

    assert cat_out.shape == full_out.shape
    assert torch.allclose(cat_out, full_out, atol=1e-4, rtol=1e-3)
