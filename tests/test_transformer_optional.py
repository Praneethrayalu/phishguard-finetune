"""Opt-in integration checks use local artifacts, never download in CI."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PHISHGUARD_TEST_ML") != "1",
    reason="Set PHISHGUARD_TEST_ML=1 after training",
)


def test_local_lora_and_quantized():
    torch = pytest.importorskip("torch")
    from phishguard.transformer import load_adapter, predict

    assert Path("artifacts/lora/adapter_model.safetensors").is_file()
    model, tokenizer = load_adapter()
    sample = ["Please review the project meeting agenda before tomorrow morning."]
    fp32 = predict(model, tokenizer, sample)
    quantized = torch.load(
        "artifacts/quantized.pt", map_location="cpu", weights_only=False
    )
    int8 = predict(quantized, tokenizer, sample)
    assert len(fp32) == len(int8) == 1
    assert 0 <= fp32[0] <= 1 and 0 <= int8[0] <= 1
