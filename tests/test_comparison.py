"""Comprehensive tests for multi-GPU comparison matrix, recommendations, and CLI."""

import json

from typer.testing import CliRunner

from infersize.analytics.comparison import (
    DEFAULT_COMPARISON_GPUS,
    compare_gpus_for_workload,
)
from infersize.cli import app
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import get_model_spec

runner = CliRunner()


def test_compare_gpus_default_matrix():
    """Verify default GPU candidate matrix for Qwen 2.5 7B FP16."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    report = compare_gpus_for_workload(
        model=qwen_7b,
        precision="fp16",
        kv_cache_dtype="fp16",
        batch_size=1,
        context_tokens=2048,
    )

    assert report.model_name == "qwen2.5-7b"
    assert len(report.items) == len(DEFAULT_COMPARISON_GPUS)

    item_map = {it.gpu_id: it for it in report.items}

    # RTX 3050 (6GB) and RTX 4060 (8GB) MUST OOM on 7B FP16 (~15GB weights)
    assert not item_map["rtx3050"].fits
    assert item_map["rtx3050"].status == "OOM"
    assert item_map["rtx3050"].max_concurrent_batch == 0

    assert not item_map["rtx4060"].fits
    assert item_map["rtx4060"].status == "OOM"

    # RTX 4090 (24GB) fits comfortably
    assert item_map["rtx4090"].fits
    assert item_map["rtx4090"].headroom_gb > 5.0
    assert item_map["rtx4090"].max_concurrent_batch > 10

    # A100 80GB and H100 80GB fit with excess headroom for batch=1
    assert item_map["h100-80gb"].fits
    assert item_map["h100-80gb"].status == "UNDERUTILIZED"

    # Sizing recommendations
    assert report.recommended_optimal_gpu == "rtx4090"
    assert report.recommended_performance_gpu in ["h100-80gb", "a100-80gb"]


def test_compare_custom_candidates_and_all_oom():
    """Verify custom candidate lists and handling when all candidates OOM."""
    llama_8b = get_model_spec("llama3-8b")
    rtx3050 = get_gpu_profile("rtx3050")
    rtx4060 = get_gpu_profile("rtx4060")

    # Both 6GB and 8GB OOM for FP16 8B model
    report = compare_gpus_for_workload(
        model=llama_8b,
        candidate_gpus=[rtx3050, rtx4060],
        precision="fp16",
    )
    assert len(report.items) == 2
    assert all(not it.fits for it in report.items)
    assert report.recommended_optimal_gpu is None
    assert report.recommended_performance_gpu is None


def test_compare_with_tensor_parallelism():
    """Verify multi-GPU tensor parallelism analytical memory partitioning."""
    qwen_72b = get_model_spec("qwen2.5-72b")
    a100_80gb = get_gpu_profile("a100-80gb")

    # Qwen 72B FP16 is ~144GB weights -> OOM on single 80GB GPU
    single_gpu = compare_gpus_for_workload(
        model=qwen_72b,
        candidate_gpus=[a100_80gb],
        precision="fp16",
        tensor_parallel=1,
    )
    assert not single_gpu.items[0].fits
    assert single_gpu.items[0].status == "OOM"

    # With TP=2, weights are ~72GB per GPU -> might be tight/OOM with KV cache and driver
    # With TP=4, weights are ~36GB per GPU -> easily fits on 80GB GPU
    tp4_report = compare_gpus_for_workload(
        model=qwen_72b,
        candidate_gpus=[a100_80gb],
        precision="fp16",
        tensor_parallel=4,
    )
    assert tp4_report.items[0].fits
    assert tp4_report.items[0].headroom_gb > 20.0
    assert tp4_report.recommended_optimal_gpu == "a100-80gb"


def test_markdown_table_export():
    """Verify Markdown table generation contains appropriate RFC headers and rows."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    report = compare_gpus_for_workload(
        model=qwen_7b,
        precision="int4",
        context_tokens=4096,
    )
    md = report.to_markdown_table()
    assert "### InferSize Hardware Comparison" in md
    assert "| GPU | VRAM Total | VRAM Req |" in md
    assert "Right-Sized / Cost-Effective" in md
    assert "rtx3050" in md


def test_cli_compare_command():
    """Verify the CLI `compare` command output formatting and exit codes."""
    res = runner.invoke(app, ["compare", "--model", "qwen2.5-7b", "--quant", "fp16"])
    assert res.exit_code == 0
    assert "HARDWARE COMPARISON MATRIX" in res.output
    assert "rtx4090" in res.output
    assert "RECOMMENDATION" in res.output


def test_cli_compare_json_output():
    """Verify machine-readable JSON output for CI and infrastructure automation."""
    res = runner.invoke(
        app,
        [
            "compare",
            "--model",
            "qwen2.5-7b",
            "--quant",
            "int4",
            "--gpus",
            "rtx3050,rtx4090",
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["model_name"] == "qwen2.5-7b"
    assert data["precision"] == "int4"
    assert len(data["items"]) == 2
    assert data["items"][0]["gpu_id"] == "rtx3050"
    assert data["items"][1]["gpu_id"] == "rtx4090"


def test_cli_compare_invalid_inputs():
    """Verify robust error reporting when invalid model or GPU names are passed."""
    bad_model = runner.invoke(app, ["compare", "--model", "unknown-transformer-model"])
    assert bad_model.exit_code == 1
    assert "Error" in bad_model.output

    bad_gpu = runner.invoke(app, ["compare", "--gpus", "nonexistent-accelerator"])
    assert bad_gpu.exit_code == 1
    assert "Error" in bad_gpu.output
