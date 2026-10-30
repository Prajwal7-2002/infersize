"""Cloud GPU hourly rental rates and provider pricing profiles."""

from __future__ import annotations

from typing import Literal

CloudProvider = Literal["runpod", "lambda", "aws", "gcp"]

# Representative hourly spot / on-demand rental rates (USD per single GPU)
# Updated based on prevailing cloud provider market rates
CLOUD_GPU_PRICING: dict[str, dict[str, float]] = {
    "runpod": {
        "rtx4090": 0.44,
        "t4": 0.20,
        "l4": 0.38,
        "a100-40gb": 1.64,
        "a100-80gb": 1.89,
        "h100-80gb": 2.89,
        "h200-141gb": 3.89,
    },
    "lambda": {
        "l4": 0.50,
        "a100-40gb": 1.10,
        "a100-80gb": 1.49,
        "h100-80gb": 2.49,
        "h200-141gb": 3.50,
    },
    "aws": {
        "t4": 0.526,  # g4dn.xlarge
        "l4": 0.804,  # g6.xlarge
        "a100-40gb": 4.10,  # p4d.24xlarge normalized per-GPU
        "a100-80gb": 5.12,  # p4de.24xlarge normalized per-GPU
        "h100-80gb": 12.29,  # p5.48xlarge normalized per-GPU
    },
    "gcp": {
        "t4": 0.35,
        "l4": 0.70,
        "a100-40gb": 2.93,
        "a100-80gb": 3.67,
        "h100-80gb": 4.50,
    },
}

# Baseline default fallback rate per GPU when provider isn't explicitly pinned
DEFAULT_HOURLY_RATES: dict[str, float] = {
    "rtx3050": 0.0,  # Local consumer edge hardware
    "rtx4060": 0.0,  # Local consumer edge hardware
    "rtx4090": 0.44,  # RunPod / Vast community cloud rate
    "t4": 0.25,
    "l4": 0.50,
    "a100-40gb": 1.64,
    "a100-80gb": 1.89,
    "h100-80gb": 2.89,
    "h200-141gb": 3.89,
    "m3-max-36gb": 0.0,  # Apple Silicon workstation
    "m3-max-128gb": 0.0,  # Apple Silicon workstation
}


def get_hourly_rate(
    gpu_id: str,
    provider: str = "runpod",
    custom_rate: float | None = None,
) -> float:
    """Retrieve hourly rental rate in USD for a given GPU and cloud provider.

    If a custom rate is passed and > 0, it takes highest precedence.
    """
    if custom_rate is not None and custom_rate >= 0.0:
        return custom_rate

    prov_key = provider.lower().strip()
    gpu_key = gpu_id.lower().strip().replace(" ", "").replace("_", "-")

    if prov_key in CLOUD_GPU_PRICING and gpu_key in CLOUD_GPU_PRICING[prov_key]:
        return CLOUD_GPU_PRICING[prov_key][gpu_key]

    if gpu_key in DEFAULT_HOURLY_RATES:
        return DEFAULT_HOURLY_RATES[gpu_key]

    return 1.50  # Generic default cloud GPU rate
