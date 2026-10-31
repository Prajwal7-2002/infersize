"""infersize: Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving."""

__version__ = "0.3.0"
__author__ = "Prajwal Pujari"

from infersize.analytics.vram import VRAMEstimateResult, estimate_vram_usage
from infersize.hardware.specs import GPUProfile, get_gpu_profile, list_gpus
from infersize.models.specs import ModelArch, get_model_spec, list_models

__all__ = [
    "__version__",
    "ModelArch",
    "GPUProfile",
    "VRAMEstimateResult",
    "estimate_vram_usage",
    "get_model_spec",
    "get_gpu_profile",
    "list_models",
    "list_gpus",
]
