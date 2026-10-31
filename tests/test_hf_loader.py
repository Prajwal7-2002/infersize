"""Tests for dynamic Hugging Face config loading, MoE active parameter math, and CLI options."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from infersize.analytics.cost import estimate_serving_cost
from infersize.cli import app
from infersize.hardware.specs import get_gpu_profile
from infersize.models.loader import (
    load_model_arch_from_config,
    load_model_arch_from_hf,
    parse_hf_config_dict,
    resolve_model_arch,
)
from infersize.models.specs import ModelArch

runner = CliRunner()


def test_parse_hf_config_dict_dense():
    """Verify parsing standard dense Hugging Face config dictionary."""
    raw_cfg = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "model_type": "llama",
    }
    spec = parse_hf_config_dict(raw_cfg, model_name="test-llama")
    assert spec.name == "test-llama"
    assert spec.num_layers == 32
    assert spec.hidden_size == 4096
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.gqa_ratio == 4.0
    assert spec.head_dim == 128
    assert spec.is_moe is False
    assert spec.mlp_type == "swiglu"
    assert spec.calculate_total_parameters() == spec.calculate_active_parameters()


def test_parse_hf_config_dict_moe():
    """Verify parsing Mixture of Experts config with active parameter separation."""
    raw_moe_cfg = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "model_type": "mixtral",
    }
    spec = parse_hf_config_dict(raw_moe_cfg, model_name="mixtral-8x7b")
    assert spec.is_moe is True
    assert spec.num_experts == 8
    assert spec.num_experts_per_tok == 2

    total_p = spec.calculate_total_parameters()
    active_p = spec.calculate_active_parameters()
    # Total parameters should store all 8 experts, active per token should only route 2
    assert total_p > active_p
    assert 40e9 < total_p < 55e9  # Mixtral 8x7B total ~46.7B
    assert 10e9 < active_p < 16e9  # Mixtral 8x7B active ~12.9B


def test_moe_cost_uses_active_parameters():
    """Verify that MoE serving cost calculation calculates memory traffic from active parameters."""
    moe_spec = ModelArch(
        name="test-moe-8x7b",
        family="moe",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
    )
    gpu = get_gpu_profile("h100-80gb")
    # In FP8 precision, 46.7B weights ~ 46.7 GB, which comfortably fits on 80GB H100
    res = estimate_serving_cost(
        model=moe_spec,
        gpu=gpu,
        precision="fp8",
        batch_size=1,
        context_tokens=2048,
        tensor_parallel=1,
    )
    # Weight bytes read per decode step should correspond to active parameters (~12.9B * 1.0 byte)
    expected_active_weight_bytes = int(moe_spec.calculate_active_parameters() * 1.0)
    assert res.weight_bytes_per_step == expected_active_weight_bytes
    assert res.fits_in_vram is True
    assert res.tokens_per_sec_per_user > 50  # Fast decode due to streaming only 2 experts


def test_load_model_arch_from_local_config(tmp_path):
    """Verify loading architecture from a local config.json file."""
    cfg = {
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 5120,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 64000,
        "model_type": "custom",
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    spec = load_model_arch_from_config(cfg_file)
    assert spec.num_layers == 16
    assert spec.hidden_size == 2048
    assert spec.gqa_ratio == 4.0

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_arch_from_config(tmp_path / "nonexistent_config.json")


def test_load_model_arch_from_hf_mock():
    """Verify HTTP fetching logic and token authorization header."""
    mock_payload = {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "model_type": "qwen2",
    }
    raw_json = json.dumps(mock_payload).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = raw_json
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        spec = load_model_arch_from_hf("Qwen/Qwen2.5-7B", hf_token="hf_test_secret_token")
        assert spec.name == "Qwen/Qwen2.5-7B"
        assert spec.hidden_size == 3584

        req = mock_urlopen.call_args[0][0]
        assert "https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json" in req.full_url
        assert req.headers["Authorization"] == "Bearer hf_test_secret_token"


def test_load_model_arch_from_hf_errors():
    """Verify HTTP error handling for 404 not found, 401 gated access, and URLError."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        # 404
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://hf.co/invalid", code=404, msg="Not Found", hdrs={}, fp=io.BytesIO()
        )
        with pytest.raises(ValueError, match="not found"):
            load_model_arch_from_hf("nonexistent/model")

        # 401 / 403 gated
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://hf.co/gated", code=403, msg="Forbidden", hdrs={}, fp=io.BytesIO()
        )
        with pytest.raises(PermissionError, match="Access denied"):
            load_model_arch_from_hf("meta-llama/Llama-2-7b")

        # Connection error
        mock_urlopen.side_effect = urllib.error.URLError(reason="DNS resolution failed")
        with pytest.raises(ConnectionError, match="Failed to connect"):
            load_model_arch_from_hf("any/model")


def test_resolve_model_arch(tmp_path):
    """Verify resolving model specs through presets, file paths, and HF identifiers."""
    # 1. Preset
    spec1 = resolve_model_arch("llama3-8b")
    assert spec1.hidden_size == 4096

    # 2. Local config file path
    cfg = {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "vocab_size": 50257,
    }
    cfg_file = tmp_path / "custom.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    spec2 = resolve_model_arch(str(cfg_file))
    assert spec2.num_layers == 12

    # 3. Via --config explicit argument
    spec3 = resolve_model_arch("ignored", config_path=str(cfg_file))
    assert spec3.num_layers == 12

    # 4. Unknown preset without network mock
    with patch("urllib.request.urlopen", side_effect=Exception("network down")):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_arch("unknown-preset-xyz")


def test_cli_commands_with_config(tmp_path):
    """Verify all CLI commands execute with dynamic --config path."""
    cfg = {
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "model_type": "llama",
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    # 1. vram
    res_vram = runner.invoke(app, ["vram", "--config", str(cfg_file), "--gpu", "rtx4090"])
    assert res_vram.exit_code == 0
    assert "VRAM CAPACITY ANALYSIS" in res_vram.stdout

    # 2. compare
    res_cmp = runner.invoke(
        app, ["compare", "--config", str(cfg_file), "--gpus", "rtx4090,a100-80gb"]
    )
    assert res_cmp.exit_code == 0
    assert "HARDWARE COMPARISON MATRIX" in res_cmp.stdout

    # 3. cost
    res_cost = runner.invoke(app, ["cost", "--config", str(cfg_file), "--gpu", "rtx4090"])
    assert res_cost.exit_code == 0
    assert "CLOUD SERVING ROOFLINE & TCO ANALYSIS" in res_cost.stdout
    assert "COST PER 1M TOKENS" in res_cost.stdout

    # 4. report
    out_file = tmp_path / "test_rfc.md"
    res_rep = runner.invoke(
        app, ["report", "--config", str(cfg_file), "--gpu", "rtx4090", "--output", str(out_file)]
    )
    assert res_rep.exit_code == 0
    assert out_file.exists()
    assert "Architecture RFC" in out_file.read_text(encoding="utf-8")

    # 5. max-batch
    res_mb = runner.invoke(app, ["max-batch", "--config", str(cfg_file), "--gpu", "rtx4090"])
    assert res_mb.exit_code == 0
    assert "CONCURRENT BATCH SIZING" in res_mb.stdout

    # 6. max-context
    res_mc = runner.invoke(app, ["max-context", "--config", str(cfg_file), "--gpu", "rtx4090"])
    assert res_mc.exit_code == 0
    assert "CONTEXT LENGTH CAPACITY" in res_mc.stdout
