"""Analytics, VRAM capacity modeling, and latency estimations."""

from infersize.analytics.vram import (
    VRAMEstimateResult,
    calculate_activation_memory_gb,
    calculate_kv_cache_bytes_per_token,
    calculate_kv_cache_total_gb,
    calculate_weight_memory_gb,
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)

__all__ = [
    "VRAMEstimateResult",
    "calculate_kv_cache_bytes_per_token",
    "calculate_weight_memory_gb",
    "calculate_kv_cache_total_gb",
    "calculate_activation_memory_gb",
    "estimate_vram_usage",
    "find_max_context_length",
    "find_max_concurrent_batch_size",
]

