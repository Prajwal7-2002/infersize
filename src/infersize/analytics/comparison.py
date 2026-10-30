"""Analytical hardware comparison engine for multi-GPU evaluation and recommendation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from infersize.analytics.vram import (
    VRAMEstimateResult,
    estimate_vram_usage,
    find_max_concurrent_batch_size,
    find_max_context_length,
)
from infersize.hardware.specs import GPU_PRESETS, GPUProfile
from infersize.models.specs import ModelArch

StatusVerdict = Literal["OPTIMAL", "FITS", "TIGHT", "UNDERUTILIZED", "OOM"]


class GPUComparisonItem(BaseModel):
    """Detailed capacity and suitability verdict for a single GPU profile."""

    gpu_id: str = Field(description="Preset key or identifier of the GPU")
    gpu_name: str = Field(description="Full display name of the GPU")
    vram_total_gb: float = Field(description="Total addressable GPU memory in GB")
    memory_bandwidth_gb_s: float = Field(description="Peak memory bandwidth in GB/s")
    required_vram_gb: float = Field(description="Total VRAM demand for workload in GB")
    headroom_gb: float = Field(description="Remaining free headroom in GB")
    utilization_pct: float = Field(description="Percentage of VRAM occupied by workload")
    fits: bool = Field(description="True if workload fits within physical memory")
    status: StatusVerdict = Field(description="Categorical suitability status")
    max_concurrent_batch: int = Field(description="Max concurrent streams supported at context")
    max_context_tokens: int = Field(description="Max context supported at current batch")
    recommendation_note: str = Field(description="Actionable procurement recommendation")
    vram_estimate: VRAMEstimateResult = Field(description="Full breakdown of VRAM components")


class GPUComparisonReport(BaseModel):
    """Comprehensive comparison matrix across multiple candidate accelerators."""

    model_name: str
    precision: str
    kv_cache_dtype: str
    batch_size: int
    context_tokens: int
    tensor_parallel: int
    items: list[GPUComparisonItem]
    recommended_optimal_gpu: str | None = Field(
        default=None, description="Most cost-effective GPU that safely satisfies workload"
    )
    recommended_performance_gpu: str | None = Field(
        default=None, description="Highest-throughput GPU among all fitting candidates"
    )

    def to_markdown_table(self) -> str:
        """Render the comparison report into a clean Markdown table for RFCs."""
        lines = [
            f"### InferSize Hardware Comparison: `{self.model_name}`",
            f"- **Workload**: Batch={self.batch_size}, Context={self.context_tokens:,} tokens, "
            f"Weights={self.precision.upper()}, KV-Cache={self.kv_cache_dtype.upper()}, TP={self.tensor_parallel}",
            "",
            "| GPU | VRAM Total | VRAM Req | Util (%) | Headroom | Max Batch | Bandwidth | Status | Verdict |",
            "|:---|---:|---:|---:|---:|---:|---:|:---:|:---|",
        ]
        for item in self.items:
            sign = "+" if item.headroom_gb >= 0 else ""
            status_symbol = {
                "OPTIMAL": "⭐ Optimal",
                "FITS": "✔ Fits",
                "TIGHT": "⚠ Tight",
                "UNDERUTILIZED": "ℹ Overkill",
                "OOM": "❌ OOM",
            }.get(item.status, item.status)
            lines.append(
                f"| **{item.gpu_id}** | {item.vram_total_gb:.1f} GB | {item.required_vram_gb:.1f} GB | "
                f"{item.utilization_pct:.1f}% | {sign}{item.headroom_gb:.1f} GB | {item.max_concurrent_batch} | "
                f"{item.memory_bandwidth_gb_s:.0f} GB/s | {status_symbol} | {item.recommendation_note} |"
            )

        lines.append("")
        if self.recommended_optimal_gpu:
            lines.append(f"**Right-Sized / Cost-Effective**: `{self.recommended_optimal_gpu}`")
        if self.recommended_performance_gpu:
            lines.append(f"**Peak Throughput**: `{self.recommended_performance_gpu}`")

        return "\n".join(lines)


# Default representative GPUs for general evaluation
DEFAULT_COMPARISON_GPUS = [
    "rtx3050",
    "rtx4060",
    "rtx4090",
    "t4",
    "l4",
    "a100-40gb",
    "a100-80gb",
    "h100-80gb",
]


def compare_gpus_for_workload(
    model: ModelArch,
    candidate_gpus: list[GPUProfile] | None = None,
    precision: str = "fp16",
    kv_cache_dtype: str = "fp16",
    batch_size: int = 1,
    context_tokens: int = 2048,
    tensor_parallel: int = 1,
) -> GPUComparisonReport:
    """Compare a model serving workload across multiple GPUs side-by-side.

    Analytically solves memory footprint, max batch headroom, and yields
    procurement recommendations (Right-Sized vs Peak Performance).
    """
    if candidate_gpus is None:
        candidate_gpus = [GPU_PRESETS[gid] for gid in DEFAULT_COMPARISON_GPUS if gid in GPU_PRESETS]

    items: list[GPUComparisonItem] = []

    for gpu in candidate_gpus:
        est = estimate_vram_usage(
            model=model,
            gpu=gpu,
            precision=precision,
            kv_cache_dtype=kv_cache_dtype,
            batch_size=batch_size,
            context_tokens=context_tokens,
            tensor_parallel=tensor_parallel,
        )

        max_b = find_max_concurrent_batch_size(
            model=model,
            gpu=gpu,
            precision=precision,
            context_tokens=context_tokens,
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallel=tensor_parallel,
        )

        max_c = find_max_context_length(
            model=model,
            gpu=gpu,
            precision=precision,
            batch_size=batch_size,
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallel=tensor_parallel,
        )

        # Categorize suitability
        if not est.fits_in_vram:
            status: StatusVerdict = "OOM"
            note = (
                f"Deficit of {abs(est.free_vram_headroom_gb):.1f} GB. Requires quantization or TP."
            )
        elif est.free_vram_headroom_gb < 1.0 or est.vram_utilization_pct > 92.0:
            status = "TIGHT"
            note = "High preemption risk under dynamic KV-cache growth."
        elif est.vram_utilization_pct < 30.0:
            status = "UNDERUTILIZED"
            note = "Excess VRAM headroom. Ideal for large batching, costly for B=1."
        else:
            status = "FITS"
            note = "Healthy capacity headroom with low preemption risk."

        items.append(
            GPUComparisonItem(
                gpu_id=gpu.name,
                gpu_name=gpu.display_name,
                vram_total_gb=gpu.vram_gb,
                memory_bandwidth_gb_s=gpu.memory_bandwidth_gb_s,
                required_vram_gb=est.total_required_vram_gb,
                headroom_gb=est.free_vram_headroom_gb,
                utilization_pct=est.vram_utilization_pct,
                fits=est.fits_in_vram,
                status=status,
                max_concurrent_batch=max_b,
                max_context_tokens=max_c,
                recommendation_note=note,
                vram_estimate=est,
            )
        )

    # Determine Best Budget / Optimal GPU:
    # Smallest total VRAM among candidates that fit with at least 8% headroom
    fitting_items = [it for it in items if it.fits and it.utilization_pct <= 92.0]

    optimal_gpu_id: str | None = None
    performance_gpu_id: str | None = None

    if fitting_items:
        # Sort by VRAM ascending to find smallest capable GPU
        sorted_by_vram = sorted(fitting_items, key=lambda x: x.vram_total_gb)
        optimal_item = sorted_by_vram[0]
        optimal_gpu_id = optimal_item.gpu_id
        # Update its status and note to OPTIMAL
        for item in items:
            if item.gpu_id == optimal_gpu_id:
                item.status = "OPTIMAL"
                item.recommendation_note = "Right-sized: Minimal idle VRAM waste, optimal cost."

        # Sort by Memory Bandwidth descending to find peak throughput GPU
        sorted_by_bw = sorted(fitting_items, key=lambda x: x.memory_bandwidth_gb_s, reverse=True)
        perf_item = sorted_by_bw[0]
        performance_gpu_id = perf_item.gpu_id

    return GPUComparisonReport(
        model_name=model.name,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        batch_size=batch_size,
        context_tokens=context_tokens,
        tensor_parallel=tensor_parallel,
        items=items,
        recommended_optimal_gpu=optimal_gpu_id,
        recommended_performance_gpu=performance_gpu_id,
    )
