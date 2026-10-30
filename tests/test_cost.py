"""Comprehensive tests for cloud cost estimation, roofline throughput, and CLI."""

import json

from typer.testing import CliRunner

from infersize.analytics.cost import (
    compare_cloud_costs_for_workload,
    estimate_serving_cost,
)
from infersize.cli import app
from infersize.hardware.pricing import get_hourly_rate
from infersize.hardware.specs import get_gpu_profile
from infersize.models.specs import get_model_spec

runner = CliRunner()


def test_estimate_serving_cost_fit():
    """Verify decode roofline throughput and $/1M tokens on fitting GPU."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx4090 = get_gpu_profile("rtx4090")

    res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
        provider="runpod",
    )

    assert res.fits_in_vram
    assert res.oom_deficit_gb == 0.0
    assert res.hourly_rate_usd == 0.44
    assert res.tokens_per_sec_per_user > 40.0
    assert res.aggregate_tokens_per_sec > 40.0
    assert res.time_per_output_token_ms < 30.0
    assert res.tokens_per_hour > 100_000
    assert 1.0 < res.cost_per_million_tokens_usd < 5.0
    assert res.monthly_dedicated_cost_usd == round(0.44 * 730, 2)


def test_estimate_serving_cost_oom():
    """Verify OOM workloads are safely flagged without calculating invalid throughput."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx3050 = get_gpu_profile("rtx3050")

    res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx3050,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
    )

    assert not res.fits_in_vram
    assert res.oom_deficit_gb > 9.0
    assert res.aggregate_tokens_per_sec == 0.0
    assert res.cost_per_million_tokens_usd == 0.0


def test_batch_throughput_economics():
    """Verify concurrent batching amortizes fixed weight transfers and drops $/1M tokens."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx4090 = get_gpu_profile("rtx4090")

    b1_res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
    )

    b16_res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=16,
        context_tokens=2048,
    )

    # Aggregate throughput at B=16 must be vastly higher than B=1
    assert b16_res.aggregate_tokens_per_sec > b1_res.aggregate_tokens_per_sec * 10
    # Cost per 1M tokens at B=16 must be less than 15% of B=1 cost
    assert b16_res.cost_per_million_tokens_usd < b1_res.cost_per_million_tokens_usd * 0.15


def test_tensor_parallel_cluster_cost():
    """Verify multi-GPU Tensor Parallelism cluster rental cost scaling."""
    qwen_72b = get_model_spec("qwen2.5-72b")
    a100_80gb = get_gpu_profile("a100-80gb")

    tp4_res = estimate_serving_cost(
        model=qwen_72b,
        gpu=a100_80gb,
        precision="fp16",
        batch_size=4,
        context_tokens=2048,
        tensor_parallel=4,
        provider="runpod",
    )

    assert tp4_res.fits_in_vram
    assert tp4_res.hourly_rate_usd == 1.89
    assert tp4_res.total_cluster_hourly_usd == pytest_approx(1.89 * 4)


def test_pricing_lookup_and_override():
    """Verify provider pricing lookup and custom rate precedence."""
    aws_t4 = get_hourly_rate("t4", provider="aws")
    assert aws_t4 == 0.526

    runpod_4090 = get_hourly_rate("rtx4090", provider="runpod")
    assert runpod_4090 == 0.44

    custom = get_hourly_rate("rtx4090", custom_rate=0.35)
    assert custom == 0.35


def test_compare_cloud_costs_for_workload():
    """Verify multi-GPU cost comparison report generation."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    report = compare_cloud_costs_for_workload(
        model=qwen_7b,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
        provider="runpod",
    )

    assert report.model_name == "qwen2.5-7b"
    assert len(report.results) >= 4
    assert report.lowest_cost_setup is not None
    assert report.lowest_cost_setup.cost_per_million_tokens_usd > 0
    assert report.fastest_setup is not None


def test_cli_cost_command():
    """Verify CLI `cost` basic execution."""
    res = runner.invoke(app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090"])
    assert res.exit_code == 0
    assert "ROOFLINE & TCO ANALYSIS" in res.output
    assert "COST PER 1M TOKENS" in res.output


def test_cli_cost_compare_all_and_clouds():
    """Verify CLI `cost --compare-all` and `--compare-clouds` flags."""
    res_all = runner.invoke(app, ["cost", "--model", "qwen2.5-7b", "--compare-all"])
    assert res_all.exit_code == 0
    assert "CLOUD HARDWARE COST & THROUGHPUT MATRIX" in res_all.output

    res_clouds = runner.invoke(
        app, ["cost", "--model", "qwen2.5-7b", "--gpu", "a100-80gb", "--compare-clouds"]
    )
    assert res_clouds.exit_code == 0
    assert "Cloud Provider Price Comparison" in res_clouds.output
    assert "LAMBDA" in res_clouds.output
    assert "AWS" in res_clouds.output


def test_cli_cost_json_output():
    """Verify CLI `cost --json` outputs valid parseable JSON."""
    res = runner.invoke(app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["model_name"] == "qwen2.5-7b"
    assert data["gpu_id"] == "rtx4090"
    assert data["cost_per_million_tokens_usd"] > 0


def pytest_approx(val):
    import pytest

    return pytest.approx(val, rel=1e-3)
