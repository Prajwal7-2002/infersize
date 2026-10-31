"""Unit tests for infersize Streamlit UI logic."""

from infersize.analytics.comparison import compare_gpus_for_workload
from infersize.analytics.cost import compare_cloud_costs_for_workload, estimate_serving_cost
from infersize.analytics.speculative import estimate_speculative_decoding
from infersize.analytics.vram import estimate_vram_usage
from infersize.hardware.detect import detect_local_hardware, verify_local_fit
from infersize.hardware.specs import get_gpu_profile
from infersize.models.loader import resolve_model_arch


def test_ui_backend_pipeline_execution():
    """Verify all analytic solvers called by the Web Studio execute cleanly."""
    arch = resolve_model_arch("llama3-8b")
    draft_arch = resolve_model_arch("qwen2.5-0.5b")
    gpu = get_gpu_profile("rtx4090")

    # Mode 1: Multi-GPU matrix and primary estimate
    comp_report = compare_gpus_for_workload(
        model=arch,
        precision="int4",
        kv_cache_dtype="fp16",
        batch_size=4,
        context_tokens=2048,
        tensor_parallel=1,
    )
    assert len(comp_report.items) > 0

    vram = estimate_vram_usage(
        model=arch,
        gpu=gpu,
        precision="int4",
        kv_cache_dtype="fp16",
        batch_size=4,
        context_tokens=2048,
        tensor_parallel=1,
    )
    assert vram.total_required_vram_gb > 0

    # Mode 2: Speculative decoding
    spec = estimate_speculative_decoding(
        target_arch=arch,
        draft_arch=draft_arch,
        gpu=gpu,
        target_precision="int4",
        draft_precision="int4",
        kv_cache_dtype="fp16",
        batch_size=4,
        context_tokens=2048,
        lookahead_k=4,
        acceptance_rate=0.75,
        tensor_parallel=1,
    )
    assert spec.speedup_ratio > 1.0

    # Mode 3: Cost and cloud comparison
    cost = estimate_serving_cost(
        model=arch,
        gpu=gpu,
        precision="int4",
        kv_cache_dtype="fp16",
        batch_size=4,
        context_tokens=2048,
        tensor_parallel=1,
    )
    assert cost.cost_per_million_tokens_usd > 0
    assert cost.monthly_dedicated_cost_usd > 0

    cloud_comp = compare_cloud_costs_for_workload(
        model=arch,
        precision="int4",
        kv_cache_dtype="fp16",
        batch_size=4,
        context_tokens=2048,
        tensor_parallel=1,
    )
    assert len(cloud_comp.results) > 0
    assert cloud_comp.results[0].monthly_dedicated_cost_usd > 0

    # Mode 4: Local fit
    hw = detect_local_hardware()
    fit = verify_local_fit(arch=arch, precision="int4", hardware=hw)
    assert fit.required_memory_gb > 0
