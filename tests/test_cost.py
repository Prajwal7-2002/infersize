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
    """Verify decode roofline throughput without rate, and $/1M tokens with explicit rate."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    rtx4090 = get_gpu_profile("rtx4090")

    res_no_rate = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
    )

    assert res_no_rate.fits_in_vram
    assert res_no_rate.oom_deficit_gb == 0.0
    assert res_no_rate.hourly_rate_usd == 0.0
    assert res_no_rate.cost_per_million_tokens_usd == 0.0
    assert res_no_rate.monthly_dedicated_cost_usd == 0.0
    assert res_no_rate.tokens_per_sec_per_user > 40.0
    assert res_no_rate.aggregate_tokens_per_sec > 40.0
    assert res_no_rate.time_per_output_token_ms < 30.0
    assert res_no_rate.tokens_per_hour > 100_000

    res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
        custom_hourly_rate=0.44,
    )
    assert res.hourly_rate_usd == 0.44
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
        custom_hourly_rate=0.44,
    )

    b16_res = estimate_serving_cost(
        model=qwen_7b,
        gpu=rtx4090,
        precision="fp16",
        batch_size=16,
        context_tokens=2048,
        custom_hourly_rate=0.44,
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
        custom_hourly_rate=1.89,
    )

    assert tp4_res.fits_in_vram
    assert tp4_res.hourly_rate_usd == 1.89
    assert tp4_res.total_cluster_hourly_usd == pytest_approx(1.89 * 4)


def test_pricing_lookup_and_override():
    """Verify no built-in guessed cloud rates remain; only explicit custom_rate is returned."""
    assert get_hourly_rate("t4", provider="aws") == 0.0
    assert get_hourly_rate("rtx4090", provider="runpod") == 0.0
    assert get_hourly_rate("rtx4090", custom_rate=0.35) == 0.35


def test_compare_cloud_costs_for_workload():
    """Verify multi-GPU roofline comparison does not invent rates unless supplied."""
    qwen_7b = get_model_spec("qwen2.5-7b")
    report = compare_cloud_costs_for_workload(
        model=qwen_7b,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
    )

    assert report.model_name == "qwen2.5-7b"
    assert len(report.results) >= 4
    assert report.lowest_cost_setup is None
    assert report.fastest_setup is not None
    assert all(r.hourly_rate_usd == 0.0 for r in report.results)

    report_with_rate = compare_cloud_costs_for_workload(
        model=qwen_7b,
        precision="fp16",
        batch_size=1,
        context_tokens=2048,
        custom_hourly_rate=1.0,
    )
    assert report_with_rate.lowest_cost_setup is not None
    assert report_with_rate.lowest_cost_setup.cost_per_million_tokens_usd > 0


def test_cli_cost_command():
    """Verify CLI `cost` omits unverified dollar rates unless `--rate` is provided."""
    res_no_rate = runner.invoke(app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090"])
    assert res_no_rate.exit_code == 0
    assert "ROOFLINE" in res_no_rate.output
    assert "COST PER 1M TOKENS" not in res_no_rate.output

    res_with_rate = runner.invoke(
        app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090", "--rate", "0.44"]
    )
    assert res_with_rate.exit_code == 0
    assert "COST PER 1M TOKENS" in res_with_rate.output


def test_cli_cost_compare_all_and_clouds():
    """Verify `--compare-all` compares roofline throughput and `--compare-clouds` is removed."""
    res_all = runner.invoke(app, ["cost", "--model", "qwen2.5-7b", "--compare-all"])
    assert res_all.exit_code == 0
    assert "HARDWARE ROOF" in res_all.output or "THROUGHPUT MATRIX" in res_all.output

    res_clouds = runner.invoke(
        app, ["cost", "--model", "qwen2.5-7b", "--gpu", "a100-80gb", "--compare-clouds"]
    )
    assert res_clouds.exit_code != 0


def test_cli_cost_json_output():
    """Verify CLI `cost --json` emits valid JSON with 0 cost unless `--rate` is supplied."""
    res_no_rate = runner.invoke(
        app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090", "--json"]
    )
    assert res_no_rate.exit_code == 0
    data_no_rate = json.loads(res_no_rate.output)
    assert data_no_rate["model_name"] == "qwen2.5-7b"
    assert data_no_rate["gpu_id"] == "rtx4090"
    assert data_no_rate["cost_per_million_tokens_usd"] == 0.0

    res = runner.invoke(
        app, ["cost", "--model", "qwen2.5-7b", "--gpu", "rtx4090", "--rate", "0.44", "--json"]
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["cost_per_million_tokens_usd"] > 0


def pytest_approx(val):
    import pytest

    return pytest.approx(val, rel=1e-3)


def test_bandwidth_override_and_overhead():
    model, gpu = get_model_spec("qwen2.5-1.5b"), get_gpu_profile("rtx3050")
    base = estimate_serving_cost(model, gpu, precision="q4_k_m", context_tokens=256)
    half_bw = estimate_serving_cost(
        model, gpu, precision="q4_k_m", context_tokens=256, bandwidth_gb_s=84.0
    )
    assert half_bw.memory_bandwidth_gb_s == 84.0
    assert abs(half_bw.time_per_step_ms - 2 * base.time_per_step_ms) < 0.01
    slow = estimate_serving_cost(
        model, gpu, precision="q4_k_m", context_tokens=256, decode_overhead_ms=4.0
    )
    assert abs(slow.time_per_step_ms - (base.time_per_step_ms + 4.0)) < 0.01
    assert slow.decode_overhead_ms == 4.0


def test_unknown_precision_is_an_error_not_fp16():
    import pytest

    with pytest.raises(ValueError, match="Unknown weight precision"):
        estimate_serving_cost(get_model_spec("qwen2.5-0.5b"), get_gpu_profile("t4"), "q4")
    result = runner.invoke(app, ["cost", "-m", "qwen2.5-0.5b", "-g", "t4", "-q", "q4"])
    assert result.exit_code != 0
    assert "q4_k_m" in result.output


def test_cost_output_labels_ceiling():
    result = runner.invoke(app, ["cost", "-m", "qwen2.5-0.5b", "-g", "rtx3050", "-q", "q4_k_m"])
    assert result.exit_code == 0
    assert "ceiling" in result.output
