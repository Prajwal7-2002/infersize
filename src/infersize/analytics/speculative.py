"""Analytical modeling for Speculative Decoding VRAM sizing and memory bandwidth speedup."""

from __future__ import annotations

from pydantic import BaseModel, Field

from infersize.analytics.vram import (
    BYTES_PER_PARAM,
    calculate_kv_cache_bytes_per_token,
    estimate_vram_usage,
)
from infersize.hardware.specs import GPUProfile
from infersize.models.specs import ModelArch


class SpeculativeEstimateResult(BaseModel):
    """Detailed analytical footprint, latency, and speedup for speculative decoding."""

    target_model_name: str
    draft_model_name: str
    gpu_name: str
    target_precision: str
    draft_precision: str
    batch_size: int
    context_tokens: int
    lookahead_k: int = Field(
        description="Number of speculative tokens proposed per draft iteration (K)"
    )
    acceptance_rate: float = Field(
        description="Empirical average verification acceptance rate (alpha)"
    )

    # VRAM components (GB)
    target_vram_gb: float = Field(
        description="VRAM allocated for target model weights, KV, and activations"
    )
    draft_vram_gb: float = Field(
        description="VRAM allocated for draft model weights, KV, and activations"
    )
    total_vram_gb: float = Field(
        description="Joint VRAM required for target + draft models simultaneously"
    )
    gpu_total_vram_gb: float = Field(description="Total physical VRAM available on the GPU")
    headroom_gb: float = Field(description="Remaining free VRAM headroom in GB")
    fits_in_vram: bool = Field(
        description="True if joint target + draft footprint fits within GPU VRAM"
    )

    # Latency physics (milliseconds per forward pass / step)
    draft_step_latency_ms: float = Field(
        description="DRAM memory-bandwidth latency to generate 1 draft token"
    )
    target_verify_latency_ms: float = Field(
        description="Latency for target model to verify K tokens in a single parallel forward pass"
    )
    baseline_autoregressive_latency_ms: float = Field(
        description="Standard single-token decode latency of the target model alone"
    )

    # Speculative dynamics
    expected_accepted_tokens: float = Field(
        description="Expected accepted tokens per speculative cycle (1 + alpha * K)"
    )
    speculative_cycle_latency_ms: float = Field(
        description="Total duration of one speculative cycle (K * draft_step + target_verify)"
    )
    effective_tpot_ms: float = Field(
        description="Effective Time Per Output Token with speculative acceleration (ms)"
    )
    effective_tokens_per_sec: float = Field(
        description="Effective generated tokens per second per stream"
    )
    baseline_tokens_per_sec: float = Field(
        description="Baseline target autoregressive tokens per second"
    )
    speedup_ratio: float = Field(
        description="Net latency speedup factor (baseline_tpot / effective_tpot)"
    )


def estimate_speculative_decoding(
    target_arch: ModelArch,
    draft_arch: ModelArch,
    gpu: GPUProfile,
    target_precision: str = "fp16",
    draft_precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    lookahead_k: int = 4,
    acceptance_rate: float = 0.70,
    tensor_parallel: int = 1,
    bus_efficiency: float = 0.80,
) -> SpeculativeEstimateResult:
    """Analytically models joint VRAM consumption, memory bandwidth latency, and Leviathan speedup.

    Args:
        target_arch: Architecture specification of the large target model.
        draft_arch: Architecture specification of the small draft model.
        gpu: Target GPU hardware specification.
        target_precision: Quantization precision for target weights.
        draft_precision: Quantization precision for draft weights.
        kv_cache_dtype: Data type used for KV caches.
        batch_size: Number of concurrent request streams.
        context_tokens: Average active sequence length (prompt + output tokens).
        lookahead_k: Number of speculative candidate tokens proposed per step (K).
        acceptance_rate: Empirical token acceptance probability (alpha in [0.0, 1.0]).
        tensor_parallel: Number of GPUs in the tensor-parallel cluster (TP).
        bus_efficiency: Achievable fraction of theoretical memory bandwidth.

    Returns:
        SpeculativeEstimateResult with joint VRAM breakdown and effective speedup.
    """
    if not (0.0 <= acceptance_rate <= 1.0):
        raise ValueError(f"acceptance_rate must be between 0.0 and 1.0, got {acceptance_rate}")
    if lookahead_k < 1:
        raise ValueError(f"lookahead_k must be >= 1, got {lookahead_k}")

    # 1. Target model VRAM sizing
    target_vram = estimate_vram_usage(
        model=target_arch,
        gpu=gpu,
        precision=target_precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        tensor_parallel=tensor_parallel,
    )

    # 2. Draft model VRAM sizing
    draft_vram = estimate_vram_usage(
        model=draft_arch,
        gpu=gpu,
        precision=draft_precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        tensor_parallel=tensor_parallel,
    )

    # Joint VRAM = Target weights + Target KV + Draft weights + Draft KV + Shared CUDA Context
    # Shared CUDA runtime overhead is accounted for once
    joint_vram_gb = (
        target_vram.weight_memory_gb
        + target_vram.kv_cache_memory_gb
        + target_vram.activation_memory_gb
        + draft_vram.weight_memory_gb
        + draft_vram.kv_cache_memory_gb
        + draft_vram.activation_memory_gb
        + target_vram.cuda_overhead_gb
    )

    headroom_gb = round(gpu.vram_gb - joint_vram_gb, 3)
    fits = headroom_gb >= 0.0

    # 3. DRAM Memory Bandwidth Roofline Latency Modeling
    effective_bw_bytes_sec = (gpu.memory_bandwidth_gb_s * 1e9) * bus_efficiency * tensor_parallel

    # Target model single-token baseline decode latency
    # In decode, every step reads active weights + KV cache
    target_active_weights = target_arch.calculate_active_parameters()
    target_bytes_per_step = (
        int(target_active_weights * BYTES_PER_PARAM.get(target_precision, 2.0))
        + calculate_kv_cache_bytes_per_token(target_arch, kv_cache_dtype)
        * context_tokens
        * batch_size
    )
    baseline_target_lat_ms = (target_bytes_per_step / effective_bw_bytes_sec) * 1000.0

    # Draft model single-token step latency
    draft_active_weights = draft_arch.calculate_active_parameters()
    draft_bytes_per_step = (
        int(draft_active_weights * BYTES_PER_PARAM.get(draft_precision, 2.0))
        + calculate_kv_cache_bytes_per_token(draft_arch, kv_cache_dtype)
        * context_tokens
        * batch_size
    )
    draft_step_lat_ms = (draft_bytes_per_step / effective_bw_bytes_sec) * 1000.0

    # Target parallel verification pass:
    # Target verifies all K tokens in a single parallel GEMM forward pass.
    # Reading weights takes the exact same memory-fetch time as 1 token!
    # Additional overhead is negligible compute for small batch sizes (K <= 8).
    # We model a minor 5% attention score broadcast overhead for K tokens:
    target_verify_lat_ms = baseline_target_lat_ms * (1.0 + 0.03 * lookahead_k)

    # 4. Speculative Decoding Speedup Dynamics (Leviathan et al., 2023)
    # Expected accepted tokens per cycle = 1 (guaranteed) + alpha * K
    expected_accepted = 1.0 + (acceptance_rate * lookahead_k)

    # Duration of 1 complete speculative cycle = K draft steps + 1 target verification pass
    cycle_latency_ms = (lookahead_k * draft_step_lat_ms) + target_verify_lat_ms

    # Effective TPOT (Time Per Output Token in ms)
    effective_tpot_ms = cycle_latency_ms / expected_accepted

    # Tokens per second per stream
    effective_tok_s = (1000.0 / effective_tpot_ms) if effective_tpot_ms > 0 else 0.0
    baseline_tok_s = (1000.0 / baseline_target_lat_ms) if baseline_target_lat_ms > 0 else 0.0

    # Speedup ratio
    speedup = baseline_target_lat_ms / effective_tpot_ms if effective_tpot_ms > 0 else 1.0

    return SpeculativeEstimateResult(
        target_model_name=target_arch.name,
        draft_model_name=draft_arch.name,
        gpu_name=gpu.name,
        target_precision=target_precision,
        draft_precision=draft_precision,
        batch_size=batch_size,
        context_tokens=context_tokens,
        lookahead_k=lookahead_k,
        acceptance_rate=acceptance_rate,
        target_vram_gb=round(target_vram.total_required_vram_gb, 3),
        draft_vram_gb=round(draft_vram.total_required_vram_gb, 3),
        total_vram_gb=round(joint_vram_gb, 3),
        gpu_total_vram_gb=gpu.vram_gb,
        headroom_gb=headroom_gb,
        fits_in_vram=fits,
        draft_step_latency_ms=round(draft_step_lat_ms, 2),
        target_verify_latency_ms=round(target_verify_lat_ms, 2),
        baseline_autoregressive_latency_ms=round(baseline_target_lat_ms, 2),
        expected_accepted_tokens=round(expected_accepted, 2),
        speculative_cycle_latency_ms=round(cycle_latency_ms, 2),
        effective_tpot_ms=round(effective_tpot_ms, 2),
        effective_tokens_per_sec=round(effective_tok_s, 1),
        baseline_tokens_per_sec=round(baseline_tok_s, 1),
        speedup_ratio=round(speedup, 2),
    )
