"""Analytical VRAM Capacity Planning & Memory Wall modeling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from infersize.hardware.specs import GPUProfile
from infersize.models.specs import ModelArch

PrecisionType = Literal["fp32", "fp16", "bf16", "int8", "fp8", "int4", "fp4", "bitnet"]
KVCacheDtype = Literal["fp16", "bf16", "int8", "fp8", "int4"]

BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "bitnet": 0.25,
}

BYTES_PER_KV_ELEM: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "int4": 0.5,
}


class VRAMEstimateResult(BaseModel):
    """Detailed analytical VRAM footprint and capacity breakdown."""

    model_name: str
    gpu_name: str
    precision: str
    kv_cache_dtype: str
    batch_size: int
    context_tokens: int
    total_parameters: int

    # Memory components in Gigabytes (1 GB = 10^9 bytes)
    weight_memory_gb: float = Field(description="Memory required for model weights in GB")
    kv_cache_memory_gb: float = Field(description="Memory required for KV-cache in GB")
    activation_memory_gb: float = Field(description="Temporary buffer memory for activations in GB")
    cuda_overhead_gb: float = Field(description="Driver and CUDA context overhead in GB")
    total_required_vram_gb: float = Field(description="Total VRAM required in GB")

    gpu_total_vram_gb: float = Field(description="Total physical GPU VRAM in GB")
    free_vram_headroom_gb: float = Field(description="Remaining free VRAM in GB (negative if OOM)")
    fits_in_vram: bool = Field(description="Whether the workload fits in GPU VRAM without OOM")
    vram_utilization_pct: float = Field(description="Percentage of total VRAM used")

    # Detailed metrics
    kv_bytes_per_token: int = Field(description="Bytes of KV-cache consumed per generated token")
    weight_compression_ratio: float = Field(description="Compression factor relative to FP16")


def calculate_kv_cache_bytes_per_token(
    model: ModelArch,
    kv_cache_dtype: str = "fp16",
) -> int:
    """Calculate exact KV-cache bytes required per token sequence step.

    Formula: 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    The factor of 2 accounts for storing both Key (K) and Value (V) tensors.
    """
    dtype_key = kv_cache_dtype.lower().strip()
    bytes_per_elem = BYTES_PER_KV_ELEM.get(dtype_key, 2.0)
    bytes_per_token = int(
        2 * model.num_layers * model.num_key_value_heads * model.head_dim * bytes_per_elem
    )
    return bytes_per_token


def calculate_weight_memory_gb(
    model: ModelArch,
    precision: str = "fp16",
) -> float:
    """Calculate physical VRAM footprint in GB for model weights."""
    prec_key = precision.lower().strip()
    bytes_per_param = BYTES_PER_PARAM.get(prec_key, 2.0)
    total_params = model.calculate_total_parameters()
    total_bytes = total_params * bytes_per_param
    return total_bytes / 1e9


def calculate_kv_cache_total_gb(
    model: ModelArch,
    batch_size: int,
    context_tokens: int,
    kv_cache_dtype: str = "fp16",
) -> float:
    """Calculate total KV-cache footprint in GB for a given batch and context."""
    bytes_per_token = calculate_kv_cache_bytes_per_token(model, kv_cache_dtype)
    total_bytes = bytes_per_token * batch_size * context_tokens
    return total_bytes / 1e9


def calculate_activation_memory_gb(
    model: ModelArch,
    batch_size: int,
    context_tokens: int,
    use_flash_attention: bool = True,
) -> float:
    """Estimate temporary forward-pass activation memory in GB.

    With FlashAttention, activation memory is linear O(B * S * hidden_size) rather
    than quadratic O(B * heads * S^2) because intermediate attention matrices are never materialized.
    """
    if use_flash_attention:
        # Standard FlashAttention-2 working buffer per layer: ~4 * B * S * hidden_size * 2 bytes
        layer_act_bytes = 4 * batch_size * context_tokens * model.hidden_size * 2
        # PyTorch allocator keeps activations for the currently executing layer
        total_act_bytes = layer_act_bytes * 2  # Double buffer safety margin
    else:
        # Naive SDPA materializes attention score matrix (B, heads, S, S)
        attn_matrix_bytes = batch_size * model.num_attention_heads * (context_tokens**2) * 2
        total_act_bytes = attn_matrix_bytes + (batch_size * context_tokens * model.hidden_size * 4)

    return max(0.05, total_act_bytes / 1e9)


def estimate_vram_usage(
    model: ModelArch,
    gpu: GPUProfile,
    precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    use_flash_attention: bool = True,
) -> VRAMEstimateResult:
    """Perform rigorous analytical VRAM capacity estimation for a serving workload."""
    total_params = model.calculate_total_parameters()
    weight_gb = calculate_weight_memory_gb(model, precision)
    kv_gb = calculate_kv_cache_total_gb(model, batch_size, context_tokens, kv_cache_dtype)
    act_gb = calculate_activation_memory_gb(model, batch_size, context_tokens, use_flash_attention)
    overhead_gb = gpu.cuda_overhead_gb

    total_required = weight_gb + kv_gb + act_gb + overhead_gb
    free_headroom = gpu.vram_gb - total_required
    fits = free_headroom >= 0.0
    utilization_pct = min(100.0, (total_required / gpu.vram_gb) * 100.0)

    kv_bytes_token = calculate_kv_cache_bytes_per_token(model, kv_cache_dtype)
    bytes_per_param = BYTES_PER_PARAM.get(precision.lower(), 2.0)
    compression_ratio = 2.0 / bytes_per_param

    return VRAMEstimateResult(
        model_name=model.name,
        gpu_name=gpu.display_name,
        precision=precision.lower(),
        kv_cache_dtype=kv_cache_dtype.lower(),
        batch_size=batch_size,
        context_tokens=context_tokens,
        total_parameters=total_params,
        weight_memory_gb=round(weight_gb, 3),
        kv_cache_memory_gb=round(kv_gb, 3),
        activation_memory_gb=round(act_gb, 3),
        cuda_overhead_gb=round(overhead_gb, 3),
        total_required_vram_gb=round(total_required, 3),
        gpu_total_vram_gb=round(gpu.vram_gb, 3),
        free_vram_headroom_gb=round(free_headroom, 3),
        fits_in_vram=fits,
        vram_utilization_pct=round(utilization_pct, 1),
        kv_bytes_per_token=kv_bytes_token,
        weight_compression_ratio=round(compression_ratio, 2),
    )


def find_max_context_length(
    model: ModelArch,
    gpu: GPUProfile,
    precision: str = "fp16",
    batch_size: int = 1,
    kv_cache_dtype: str = "fp16",
    safety_margin_gb: float = 0.5,
) -> int:
    """Analytically solve for the maximum context window before OOM."""
    weight_gb = calculate_weight_memory_gb(model, precision)
    overhead_gb = gpu.cuda_overhead_gb + safety_margin_gb
    fixed_gb = weight_gb + overhead_gb

    available_for_kv_act = gpu.vram_gb - fixed_gb
    if available_for_kv_act <= 0:
        return 0

    bytes_per_token = calculate_kv_cache_bytes_per_token(model, kv_cache_dtype) * batch_size
    # Solve linear equation: context * bytes_per_token <= available_bytes
    max_tokens = int((available_for_kv_act * 1e9) / bytes_per_token)
    return max(0, max_tokens)


def find_max_concurrent_batch_size(
    model: ModelArch,
    gpu: GPUProfile,
    precision: str = "fp16",
    context_tokens: int = 2048,
    kv_cache_dtype: str = "fp16",
    safety_margin_gb: float = 0.5,
) -> int:
    """Analytically solve for the maximum concurrent request batch size before OOM."""
    weight_gb = calculate_weight_memory_gb(model, precision)
    overhead_gb = gpu.cuda_overhead_gb + safety_margin_gb
    fixed_gb = weight_gb + overhead_gb

    available_for_kv = gpu.vram_gb - fixed_gb
    if available_for_kv <= 0:
        return 0

    kv_per_stream_bytes = calculate_kv_cache_bytes_per_token(model, kv_cache_dtype) * context_tokens
    max_batch = int((available_for_kv * 1e9) / kv_per_stream_bytes)
    return max(0, max_batch)
