"""Tests for local silicon hardware detection and zero-OOM verification."""

from typer.testing import CliRunner

from infersize.cli import app
from infersize.hardware.detect import (
    DetectedHardware,
    LocalFitResult,
    _get_system_ram_gb,
    _match_gpu_preset,
    detect_local_hardware,
    verify_local_fit,
)
from infersize.models.specs import MODEL_PRESETS

runner = CliRunner()


def test_system_ram_detection():
    total, free = _get_system_ram_gb()
    assert total > 0.0
    assert free >= 0.0


def test_match_gpu_preset():
    assert _match_gpu_preset("NVIDIA GeForce RTX 3050 Laptop GPU", 6.0) == "rtx3050"
    assert _match_gpu_preset("NVIDIA GeForce RTX 4090", 24.0) == "rtx4090"
    assert _match_gpu_preset("NVIDIA A100-SXM4-80GB", 80.0) == "a100-80gb"
    assert _match_gpu_preset("NVIDIA A100-PCIE-40GB", 40.0) == "a100-40gb"
    assert _match_gpu_preset("NVIDIA H100 80GB HBM3", 80.0) == "h100-sxm"


def test_detect_local_hardware():
    hw = detect_local_hardware()
    assert isinstance(hw, DetectedHardware)
    assert hw.total_memory_gb > 0.0
    assert hw.free_memory_gb >= 0.0
    assert hw.device_type in ["cuda", "cpu", "apple_silicon"]


def test_verify_local_fit():
    arch = MODEL_PRESETS["qwen2.5-0.5b"]
    # Mock hardware with 8GB total and 6GB free
    mock_hw = DetectedHardware(
        device_type="cuda",
        device_name="Mock NVIDIA GPU",
        total_memory_gb=8.0,
        free_memory_gb=6.0,
        matched_gpu_id="rtx4060",
    )

    fit_res = verify_local_fit(
        arch=arch,
        precision="int4",
        hardware=mock_hw,
    )

    assert isinstance(fit_res, LocalFitResult)
    assert fit_res.fits_total is True
    assert fit_res.fits_currently is True
    assert fit_res.headroom_total_gb > 0


def test_cli_fit_command():
    result = runner.invoke(
        app,
        [
            "fit",
            "--model",
            "qwen2.5-0.5b",
            "--quant",
            "int4",
        ],
    )
    assert result.exit_code == 0
    assert "LIVE HARDWARE ZERO-OOM VERIFIER" in result.output
    assert "qwen2.5-0.5b" in result.output
