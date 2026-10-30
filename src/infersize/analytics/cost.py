"""Cloud TCO, memory bandwidth roofline modeling, and $/1M tokens cost analytics."""

from __future__ import annotations

from pydantic import BaseModel, Field

from infersize.analytics.vram import (
    BYTES_PER_PARAM,
    calculate_kv_cache_bytes_per_token,
    estimate_vram_usage,
)
from infersize.hardware.pricing import get_hourly_rate
from infersize.hardware.specs import GPU_PRESETS, GPUProfile
from infersize.models.specs import ModelArch


class CostEstimateResult(BaseModel):
    """Detailed analytical throughput, hardware roofline, and cloud cost metrics."""

    model_name: str
    gpu_name: str
    gpu_id: str
    provider: str
    hourly_rate_usd: float = Field(description="Hourly rental rate per GPU in USD")
    total_cluster_hourly_usd: float = Field(
        description="Total hourly rate across all TP GPUs in USD"
    )
    batch_size: int
    context_tokens: int
    precision: str
    kv_cache_dtype: str
    tensor_parallel: int

    # Memory traffic per decode step
    weight_bytes_per_step: int = Field(description="DRAM bytes read for weights per decode step")
    kv_bytes_per_step: int = Field(description="DRAM bytes read for KV-cache per decode step")
    total_dram_bytes_per_step: int = Field(description="Total DRAM bytes transferred per step")

    # Roofline throughput & latency
    memory_bandwidth_gb_s: float = Field(description="Theoretical peak memory bandwidth in GB/s")
    bus_efficiency: float = Field(
        description="Assumed memory bus utilization efficiency (e.g. 0.80)"
    )
    effective_bandwidth_gb_s: float = Field(description="Effective achievable bandwidth in GB/s")

    time_per_step_ms: float = Field(description="Time taken per decode step in milliseconds")
    time_per_output_token_ms: float = Field(
        description="Time Per Output Token (TPOT) per user stream in ms"
    )
    tokens_per_sec_per_user: float = Field(
        description="Generation speed per individual user stream"
    )
    aggregate_tokens_per_sec: float = Field(
        description="Total token generation throughput across entire batch"
    )
    tokens_per_hour: float = Field(description="Theoretical maximum token capacity per hour")

    # Cloud Unit Economics
    cost_per_million_tokens_usd: float = Field(
        description="Unit cost in USD to generate 1,000,000 tokens"
    )
    monthly_dedicated_cost_usd: float = Field(
        description="24/7 monthly dedicated cluster rental cost (730 hours)"
    )
    fits_in_vram: bool = Field(description="Whether the workload fits in GPU memory")
    oom_deficit_gb: float = Field(
        default=0.0, description="VRAM memory deficit in GB if workload exceeds capacity"
    )


def estimate_serving_cost(
    model: ModelArch,
    gpu: GPUProfile,
    precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    tensor_parallel: int = 1,
    provider: str = "runpod",
    custom_hourly_rate: float | None = None,
    bus_efficiency: float = 0.80,
) -> CostEstimateResult:
    """Analytically calculate decode throughput, TPOT, and cost per million tokens.

    Models the autoregressive decode phase as memory-bandwidth bound:
    Step Time = (Weight Bytes + KV Cache Bytes) / Effective Bandwidth
    """
    tp = max(1, tensor_parallel)
    vram_res = estimate_vram_usage(
        model=model,
        gpu=gpu,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        tensor_parallel=tp,
    )

    hourly_rate = get_hourly_rate(
        gpu_id=gpu.name,
        provider=provider,
        custom_rate=custom_hourly_rate,
    )
    cluster_hourly = hourly_rate * tp

    # Calculate DRAM bytes transferred per decode token step
    total_params = model.calculate_total_parameters()
    bytes_per_param = BYTES_PER_PARAM.get(precision.lower(), 2.0)
    # Per GPU weight slice
    weight_bytes = int((total_params * bytes_per_param) / tp)

    # Per GPU KV-cache slice: kv_bytes_per_token * batch_size * context_tokens / tp
    kv_bytes_per_token = calculate_kv_cache_bytes_per_token(model, kv_cache_dtype)
    kv_bytes = int((kv_bytes_per_token * batch_size * context_tokens) / tp)

    total_dram_step = weight_bytes + kv_bytes

    eff_bw_gb_s = gpu.memory_bandwidth_gb_s * bus_efficiency
    eff_bw_bytes_s = eff_bw_gb_s * 1e9

    if not vram_res.fits_in_vram:
        deficit = abs(vram_res.free_vram_headroom_gb)
        return CostEstimateResult(
            model_name=model.name,
            gpu_name=gpu.display_name,
            gpu_id=gpu.name,
            provider=provider,
            hourly_rate_usd=hourly_rate,
            total_cluster_hourly_usd=cluster_hourly,
            batch_size=batch_size,
            context_tokens=context_tokens,
            precision=precision.lower(),
            kv_cache_dtype=kv_cache_dtype.lower(),
            tensor_parallel=tp,
            weight_bytes_per_step=weight_bytes,
            kv_bytes_per_step=kv_bytes,
            total_dram_bytes_per_step=total_dram_step,
            memory_bandwidth_gb_s=gpu.memory_bandwidth_gb_s,
            bus_efficiency=bus_efficiency,
            effective_bandwidth_gb_s=round(eff_bw_gb_s, 1),
            time_per_step_ms=0.0,
            time_per_output_token_ms=0.0,
            tokens_per_sec_per_user=0.0,
            aggregate_tokens_per_sec=0.0,
            tokens_per_hour=0.0,
            cost_per_million_tokens_usd=0.0,
            monthly_dedicated_cost_usd=round(cluster_hourly * 730, 2),
            fits_in_vram=False,
            oom_deficit_gb=round(deficit, 2),
        )

    # Time per decode step in seconds: DRAM bytes / Effective Bandwidth
    step_time_s = total_dram_step / eff_bw_bytes_s
    step_time_ms = step_time_s * 1000.0

    # User-perceived latency (TPOT)
    tpot_ms = step_time_ms
    user_tok_s = 1.0 / step_time_s if step_time_s > 0 else 0.0

    # Aggregate throughput across all B concurrent streams
    aggregate_tok_s = user_tok_s * batch_size
    tokens_per_hour = aggregate_tok_s * 3600.0

    # $/1M Tokens = (Hourly Cost / Tokens Per Hour) * 1,000,000
    if tokens_per_hour > 0 and cluster_hourly > 0:
        cost_per_1m = (cluster_hourly / tokens_per_hour) * 1e6
    else:
        cost_per_1m = 0.0

    monthly_cost = cluster_hourly * 730.0

    return CostEstimateResult(
        model_name=model.name,
        gpu_name=gpu.display_name,
        gpu_id=gpu.name,
        provider=provider,
        hourly_rate_usd=round(hourly_rate, 3),
        total_cluster_hourly_usd=round(cluster_hourly, 3),
        batch_size=batch_size,
        context_tokens=context_tokens,
        precision=precision.lower(),
        kv_cache_dtype=kv_cache_dtype.lower(),
        tensor_parallel=tp,
        weight_bytes_per_step=weight_bytes,
        kv_bytes_per_step=kv_bytes,
        total_dram_bytes_per_step=total_dram_step,
        memory_bandwidth_gb_s=gpu.memory_bandwidth_gb_s,
        bus_efficiency=bus_efficiency,
        effective_bandwidth_gb_s=round(eff_bw_gb_s, 1),
        time_per_step_ms=round(step_time_ms, 2),
        time_per_output_token_ms=round(tpot_ms, 2),
        tokens_per_sec_per_user=round(user_tok_s, 1),
        aggregate_tokens_per_sec=round(aggregate_tok_s, 1),
        tokens_per_hour=round(tokens_per_hour, 0),
        cost_per_million_tokens_usd=round(cost_per_1m, 4),
        monthly_dedicated_cost_usd=round(monthly_cost, 2),
        fits_in_vram=True,
        oom_deficit_gb=0.0,
    )


class MultiGPUCostReport(BaseModel):
    """Side-by-side cost and unit economics comparison across multiple cloud setups."""

    model_name: str
    precision: str
    kv_cache_dtype: str
    batch_size: int
    context_tokens: int
    results: list[CostEstimateResult]
    lowest_cost_setup: CostEstimateResult | None = None
    fastest_setup: CostEstimateResult | None = None


def compare_cloud_costs_for_workload(
    model: ModelArch,
    candidate_gpus: list[GPUProfile] | None = None,
    precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    provider: str = "runpod",
    tensor_parallel: int = 1,
) -> MultiGPUCostReport:
    """Compare throughput, TPOT, and $/1M tokens across multiple GPU candidates."""
    if candidate_gpus is None:
        candidate_ids = ["rtx4090", "l4", "a100-40gb", "a100-80gb", "h100-80gb"]
        candidate_gpus = [GPU_PRESETS[gid] for gid in candidate_ids if gid in GPU_PRESETS]

    results: list[CostEstimateResult] = []
    for gpu in candidate_gpus:
        res = estimate_serving_cost(
            model=model,
            gpu=gpu,
            precision=precision,
            kv_cache_dtype=kv_cache_dtype,
            batch_size=batch_size,
            context_tokens=context_tokens,
            tensor_parallel=tensor_parallel,
            provider=provider,
        )
        results.append(res)

    fitting = [r for r in results if r.fits_in_vram and r.cost_per_million_tokens_usd > 0]
    lowest = min(fitting, key=lambda x: x.cost_per_million_tokens_usd) if fitting else None
    fastest = max(fitting, key=lambda x: x.aggregate_tokens_per_sec) if fitting else None

    return MultiGPUCostReport(
        model_name=model.name,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        results=results,
        lowest_cost_setup=lowest,
        fastest_setup=fastest,
    )
