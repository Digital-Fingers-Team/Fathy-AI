import importlib

import pytest
import torch


MODULE_CANDIDATES = [
    "architecture.normalization",
]


def _load_rmsnorm_class():
    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        cls = getattr(module, "RMSNorm", None)
        if cls is not None:
            return cls
    pytest.fail(f"RMSNorm not found in candidates: {MODULE_CANDIDATES}")


def test_rmsnorm_zero_input_stability():
    RMSNorm = _load_rmsnorm_class()
    layer = RMSNorm(hidden_size=64, eps=1e-6)

    x = torch.zeros(2, 4, 64)
    out = layer(x)

    assert torch.isfinite(out).all(), "RMSNorm produced non-finite values for zeros"
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_rmsnorm_scale_invariance_direction():
    RMSNorm = _load_rmsnorm_class()
    layer = RMSNorm(hidden_size=64, eps=1e-6)

    x = torch.randn(3, 7, 64)
    y1 = layer(x)
    y2 = layer(3.0 * x)

    # RMSNorm should preserve direction under scalar scaling (up to eps effects).
    cos_sim = torch.nn.functional.cosine_similarity(y1.flatten(), y2.flatten(), dim=0)
    assert cos_sim > 0.999
