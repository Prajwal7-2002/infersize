"""Comprehensive tests for VRAM analytics, parameter calculations, and capacity limits."""

import pytest
from typer.testing import CliRunner

from infersize import __version__
from infersize.analytics.vram import (
    calculate_kv_cache_bytes_per_token,
    calculate_weight_memory_gb,
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)
from infersize.cli import app
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import get_model_spec

runner = CliRunner()


def test_model_parameter_calculation():
    """Verify analytical parameter computation against known architecture sizes."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    params = qwen_7b.calculate_total_parameters()
    # Qwen-2.5-7B has ~7.6B physical parameters
    assert 7.0e9 < params < 8.0e9

    gpt2 = get_model_spec("gpt2")
    gpt2_params = gpt2.calculate_total_parameters()
    # GPT-2 small is ~124M parameters
    assert 120e6 < gpt2_params < 130e6


def test_kv_cache_per_token_math():
    """Verify exact formula: 2 * layers * kv_heads * head_dim * bytes_per_elem."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    # 2 * 28 layers * 4 kv_heads * 128 head_dim * 2 bytes = 57,344 bytes
    fp16_bytes = calculate_kv_cache_bytes_per_token(qwen_7b, kv_cache_dtype="fp16")
    assert fp16_bytes == 57344

    # INT8 should be exactly half (28,672 bytes)
    int8_bytes = calculate_kv_cache_bytes_per_token(qwen_7b, kv_cache_dtype="int8")
    assert int8_bytes == 28672

    # INT4 should be exactly a quarter (14,336 bytes)
    int4_bytes = calculate_kv_cache_bytes_per_token(qwen_7b, kv_cache_dtype="int4")
    assert int4_bytes == 14336


def test_weight_memory_precisions():
    """Verify weight memory scales proportionally across precisions."""
    llama_8b = get_model_spec("llama3-8b")
    fp32_gb = calculate_weight_memory_gb(llama_8b, "fp32")
    fp16_gb = calculate_weight_memory_gb(llama_8b, "fp16")
    int8_gb = calculate_weight_memory_gb(llama_8b, "int8")
    int4_gb = calculate_weight_memory_gb(llama_8b, "int4")
    bitnet_gb = calculate_weight_memory_gb(llama_8b, "bitnet")

    assert pytest.approx(fp32_gb, rel=1e-3) == fp16_gb * 2
    assert pytest.approx(fp16_gb, rel=1e-3) == int8_gb * 2
    assert pytest.approx(int8_gb, rel=1e-3) == int4_gb * 2
    assert pytest.approx(int4_gb, rel=1e-3) == bitnet_gb * 2


def test_estimate_vram_usage_fit_and_oom():
    """Test realistic deployment scenarios: RTX 3050 (6GB) vs A100 (80GB)."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx3050 = get_gpu_profile("rtx3050")
    a100 = get_gpu_profile("a100-80gb")

    # Qwen-7B in FP16 (requires ~15GB weights) MUST OOM on 6GB RTX 3050
    oom_res = estimate_vram_usage(
        model=qwen_7b,
        gpu=rtx3050,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
    )
    assert not oom_res.fits_in_vram
    assert oom_res.free_vram_headroom_gb < 0

    # Qwen-7B in INT4 (~3.8GB weights) MUST FIT on 6GB RTX 3050 with small context
    fit_res = estimate_vram_usage(
        model=qwen_7b,
        gpu=rtx3050,
        precision="int4",
        kv_cache_dtype="int8",
        batch_size=1,
        context_tokens=1024,
    )
    assert fit_res.fits_in_vram
    assert fit_res.free_vram_headroom_gb > 0

    # A100 (80GB) should comfortably hold Qwen-7B FP16 with 8k context
    a100_res = estimate_vram_usage(
        model=qwen_7b,
        gpu=a100,
        precision="fp16",
        batch_size=4,
        context_tokens=8192,
    )
    assert a100_res.fits_in_vram
    assert a100_res.vram_utilization_pct < 50.0


def test_capacity_solvers():
    """Verify maximum context window and batch size solvers."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx3050 = get_gpu_profile("rtx3050")

    # In INT4 on 6GB GPU, max context should be positive and bounded
    max_ctx = find_max_context_length(qwen_7b, rtx3050, precision="int4")
    assert max_ctx > 5000

    # In FP16 on 6GB GPU, it's impossible to host the weights, so max context must be 0
    max_ctx_fp16 = find_max_context_length(qwen_7b, rtx3050, precision="fp16")
    assert max_ctx_fp16 == 0

    # Test max concurrent batch size
    max_batch = find_max_concurrent_batch_size(
        qwen_7b, rtx3050, precision="int4", context_tokens=2048
    )
    assert max_batch >= 1


def test_cli_commands():
    """Verify all CLI subcommands execute with exit code 0."""
    # version
    v_res = runner.invoke(app, ["version"])
    assert v_res.exit_code == 0
    assert __version__ in v_res.stdout

    # models
    m_res = runner.invoke(app, ["models"])
    assert m_res.exit_code == 0
    assert "qwen2.5-7b" in m_res.stdout

    # gpus
    g_res = runner.invoke(app, ["gpus"])
    assert g_res.exit_code == 0
    assert "rtx3050" in g_res.stdout

    # vram
    vram_res = runner.invoke(app, ["vram", "-m", "qwen2.5-7b", "-g", "rtx3050", "-q", "int4"])
    assert vram_res.exit_code == 0
    assert "VRAM CAPACITY ANALYSIS" in vram_res.stdout

    # max-batch
    batch_res = runner.invoke(app, ["max-batch", "-m", "qwen2.5-7b", "-g", "rtx3050", "-q", "int4"])
    assert batch_res.exit_code == 0
    assert "Maximum Safe Concurrent Streams" in batch_res.stdout

    # max-context
    ctx_res = runner.invoke(app, ["max-context", "-m", "qwen2.5-7b", "-g", "rtx3050", "-q", "int4"])
    assert ctx_res.exit_code == 0
    assert "Maximum Supported Context Length" in ctx_res.stdout
