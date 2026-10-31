"""Tests for Speculative Decoding analytical capacity sizer."""

import pytest
from typer.testing import CliRunner

from infersize.analytics.speculative import (
    SpeculativeEstimateResult,
    estimate_speculative_decoding,
)
from infersize.cli import app
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import MODEL_PRESETS

runner = CliRunner()


def test_speculative_decoding_sizing_valid():
    target = MODEL_PRESETS["qwen2.5-32b"]
    draft = MODEL_PRESETS["qwen2.5-0.5b"]
    gpu = get_gpu_profile("a100-80gb")

    result = estimate_speculative_decoding(
        target_arch=target,
        draft_arch=draft,
        gpu=gpu,
        target_precision="int4",
        draft_precision="fp16",
        batch_size=1,
        context_tokens=2048,
        lookahead_k=4,
        acceptance_rate=0.75,
    )

    assert isinstance(result, SpeculativeEstimateResult)
    assert result.fits_in_vram is True
    assert result.headroom_gb > 0
    assert result.target_vram_gb > 0
    assert result.draft_vram_gb > 0
    assert result.total_vram_gb > result.target_vram_gb
    # Expected accepted tokens = 1 + (0.75 * 4) = 4.0
    assert result.expected_accepted_tokens == 4.0
    assert result.speedup_ratio > 1.5
    assert result.effective_tokens_per_sec > result.baseline_tokens_per_sec


def test_speculative_decoding_validation_errors():
    target = MODEL_PRESETS["qwen2.5-7b"]
    draft = MODEL_PRESETS["qwen2.5-0.5b"]
    gpu = get_gpu_profile("rtx3050")

    with pytest.raises(ValueError, match="acceptance_rate must be between 0.0 and 1.0"):
        estimate_speculative_decoding(
            target_arch=target,
            draft_arch=draft,
            gpu=gpu,
            acceptance_rate=1.5,
        )

    with pytest.raises(ValueError, match="lookahead_k must be >= 1"):
        estimate_speculative_decoding(
            target_arch=target,
            draft_arch=draft,
            gpu=gpu,
            lookahead_k=0,
        )


def test_speculative_decoding_oom_detection():
    # Attempting to fit 70B target + 7B draft on 6GB RTX 3050 must fail with fits_in_vram=False
    target = MODEL_PRESETS["llama3-70b"]
    draft = MODEL_PRESETS["qwen2.5-7b"]
    gpu = get_gpu_profile("rtx3050")

    result = estimate_speculative_decoding(
        target_arch=target,
        draft_arch=draft,
        gpu=gpu,
        target_precision="fp16",
        draft_precision="fp16",
    )

    assert result.fits_in_vram is False
    assert result.headroom_gb < 0


def test_cli_spec_command():
    result = runner.invoke(
        app,
        [
            "spec",
            "--target",
            "qwen2.5-7b",
            "--draft",
            "qwen2.5-0.5b",
            "--gpu",
            "rtx4090",
            "--target-quant",
            "int4",
            "--lookahead",
            "4",
            "--acceptance",
            "0.80",
        ],
    )
    assert result.exit_code == 0
    assert "SPECULATIVE DECODING CAPACITY" in result.output
    assert "NET SPECULATIVE SPEEDUP" in result.output
    assert "qwen2.5-7b" in result.output
