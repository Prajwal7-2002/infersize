"""Streamlit Web UI for infersize: Interactive LLM Capacity Planner & GPU Sizing Studio."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so app runs directly from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    print(
        "Streamlit is required to run the infersize web studio. Install with: pip install 'infersize[ui]' or pip install streamlit"
    )
    sys.exit(1)

from infersize import __version__
from infersize.analytics.comparison import compare_gpus_for_workload
from infersize.analytics.cost import compare_cloud_costs_for_workload, estimate_serving_cost
from infersize.analytics.speculative import estimate_speculative_decoding
from infersize.analytics.vram import estimate_vram_usage
from infersize.hardware.detect import DetectedHardware, detect_local_hardware, verify_local_fit
from infersize.hardware.specs import GPU_PRESETS, get_gpu_profile
from infersize.models.loader import resolve_model_arch
from infersize.models.specs import MODEL_PRESETS

# Page Setup


def main():
    st.set_page_config(
        page_title="infersize ⚡ LLM Capacity & Sizing Studio",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling (Dark Modern CAD)
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #0d1b2a;
            border: 1px solid #10b981;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin-bottom: 12px;
        }
        .metric-title {
            font-size: 0.85rem;
            color: #94a3b8;
            font-family: monospace;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #6ee7b7;
        }
        .status-fit {
            color: #10b981;
            font-weight: 700;
        }
        .status-oom {
            color: #ef4444;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Workload Inputs
    st.sidebar.markdown(f"### ⚡ infersize `v{__version__}`")
    st.sidebar.markdown("**First-Principles Capacity Sizing Studio**")

    mode = st.sidebar.radio(
        "Choose Sizing Mode:",
        [
            "⚡ Multi-GPU VRAM & Capacity",
            "🚀 Speculative Decoding Sizer",
            "💰 Cloud TCO & $/1M Tokens",
            "🖥️ Live Local Hardware Check",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎛️ Workload Parameters")

    model_choice = st.sidebar.selectbox(
        "Model Architecture:",
        list(MODEL_PRESETS.keys()) + ["Custom Hugging Face ID"],
        index=0,
    )

    if model_choice == "Custom Hugging Face ID":
        custom_hf_id = st.sidebar.text_input(
            "Hugging Face Repo ID:", value="Qwen/Qwen2.5-7B-Instruct"
        )
        target_model_name = custom_hf_id
    else:
        target_model_name = model_choice

    quant_choice = st.sidebar.selectbox(
        "Weight Quantization:",
        ["fp16", "bf16", "int8", "fp8", "int4", "bitnet"],
        index=0,
    )

    kv_choice = st.sidebar.selectbox(
        "KV-Cache Precision:",
        ["fp16", "fp8", "int4"],
        index=0,
    )

    context_len = st.sidebar.slider(
        "Sequence Length (Prompt + Output):",
        min_value=512,
        max_value=65536,
        value=2048,
        step=512,
    )

    batch_sz = st.sidebar.slider(
        "Concurrent Batch Size (Streams):",
        min_value=1,
        max_value=128,
        value=4,
        step=1,
    )

    tp_degree = st.sidebar.selectbox(
        "Tensor Parallelism (TP):",
        [1, 2, 4, 8],
        index=0,
    )

    # Load Model Architecture
    try:
        arch = resolve_model_arch(target_model_name)
    except Exception as e:
        st.error(f"Error loading model architecture: {e}")
        st.stop()

    # Main View: Dispatch by Mode
    if mode == "⚡ Multi-GPU VRAM & Capacity":
        st.title("⚡ Multi-GPU VRAM Footprint & Capacity Matrix")
        st.markdown(
            f"Evaluating **{arch.name}** ({arch.calculate_total_parameters() / 1e9:.2f}B parameters) with **{quant_choice.upper()}** weights at **batch {batch_sz}** and **{context_len:,} context tokens**."
        )

        # Multi-GPU Table
        comparison_report = compare_gpus_for_workload(
            model=arch,
            precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            tensor_parallel=tp_degree,
        )

        # Summary metrics for default primary GPU (e.g. RTX 4090 or A100)
        primary_gpu = get_gpu_profile("rtx4090")
        vram_calc = estimate_vram_usage(
            model=arch,
            gpu=primary_gpu,
            precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            tensor_parallel=tp_degree,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Required VRAM", f"{vram_calc.total_required_vram_gb:.2f} GB")
        with col2:
            st.metric("Model Weights", f"{vram_calc.weight_memory_gb:.2f} GB")
        with col3:
            st.metric("KV-Cache (Paged)", f"{vram_calc.kv_cache_memory_gb:.2f} GB")
        with col4:
            st.metric("Activation Buffers", f"{vram_calc.activation_memory_gb:.2f} GB")

        st.markdown("### 🖥️ Hardware Comparison Matrix")

        # Format table for display
        display_rows = []
        for r in comparison_report.items:
            status_symbol = "✔ FITS" if r.fits else "✘ OOM"
            display_rows.append(
                {
                    "GPU Accelerator": r.gpu_name,
                    "Total VRAM": f"{r.vram_total_gb:.1f} GB",
                    "Workload Required": f"{r.required_vram_gb:.2f} GB",
                    "Headroom": f"{r.headroom_gb:+.2f} GB",
                    "VRAM Util": f"{r.utilization_pct:.1f}%",
                    "Max Concurrent Batch": f"{r.max_concurrent_batch} reqs",
                    "Memory Bandwidth": f"{r.memory_bandwidth_gb_s:.0f} GB/s",
                    "Status": status_symbol,
                    "Verdict": r.status,
                }
            )
        st.dataframe(display_rows, use_container_width=True)

        # CLI Command Helper
        st.markdown("---")
        st.markdown("##### ⚡ Equivalent Terminal CLI Command:")
        st.code(
            f"infersize compare --model {arch.name} --quant {quant_choice} --batch-size {batch_sz} --context {context_len} --tp {tp_degree}",
            language="bash",
        )

    elif mode == "🚀 Speculative Decoding Sizer":
        st.title("🚀 Speculative Decoding Analytical Capacity Sizer")
        st.markdown(
            "Model joint VRAM footprint, verification overhead, and effective decode acceleration under Leviathan speculative verification."
        )

        draft_choice = st.selectbox(
            "Draft Model Architecture:",
            [m for m in MODEL_PRESETS.keys() if m != arch.name],
            index=0,
        )
        draft_arch = resolve_model_arch(draft_choice)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            spec_gpu_id = st.selectbox(
                "Serving GPU:",
                list(GPU_PRESETS.keys()),
                index=list(GPU_PRESETS.keys()).index("a100-80gb"),
            )
            spec_gpu = get_gpu_profile(spec_gpu_id)
        with col_s2:
            lookahead = st.slider("Lookahead Tokens (K):", min_value=1, max_value=8, value=4)
        with col_s3:
            acceptance = st.slider(
                "Verification Acceptance Rate (α):",
                min_value=0.1,
                max_value=1.0,
                value=0.75,
                step=0.05,
            )

        spec_res = estimate_speculative_decoding(
            target_arch=arch,
            draft_arch=draft_arch,
            gpu=spec_gpu,
            target_precision=quant_choice,
            draft_precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            lookahead_k=lookahead,
            acceptance_rate=acceptance,
            tensor_parallel=tp_degree,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Effective Speedup",
                f"{spec_res.speedup_ratio:.2f}x",
                delta=f"{spec_res.speedup_ratio - 1.0:+.2f}x",
            )
        with c2:
            st.metric(
                "Effective Throughput",
                f"{spec_res.effective_tokens_per_sec:.1f} tok/s",
                delta=f"{spec_res.effective_tokens_per_sec - spec_res.baseline_tokens_per_sec:+.1f} tok/s",
            )
        with c3:
            st.metric("Joint VRAM Required", f"{spec_res.total_vram_gb:.2f} GB")
        with c4:
            fit_label = "✔ FITS" if spec_res.fits_in_vram else "✘ OOM"
            st.metric("Hardware Fit", fit_label, f"{spec_res.headroom_gb:+.2f} GB free")

        st.markdown("### 📊 Speculative Latency Dynamics")
        st.write(
            f"* Target Model Autoregressive Baseline: **{spec_res.baseline_autoregressive_latency_ms:.1f} ms** per token ({spec_res.baseline_tokens_per_sec:.1f} tok/s)\n"
            f"* Draft Model Step Latency: **{spec_res.draft_step_latency_ms:.1f} ms** per draft token\n"
            f"* Target Parallel Verification Pass: **{spec_res.target_verify_latency_ms:.1f} ms** for all {lookahead} candidate tokens\n"
            f"* Expected Accepted Tokens per Cycle: **{spec_res.expected_accepted_tokens:.2f} tokens** ($1 + {acceptance} \\times {lookahead}$)\n"
            f"* Effective Latency per Accepted Token: **{spec_res.effective_tpot_ms:.1f} ms/token**"
        )

        st.code(
            f"infersize spec --target {arch.name} --draft {draft_arch.name} --gpu {spec_gpu_id} --quant {quant_choice} --acceptance {acceptance} --lookahead {lookahead}",
            language="bash",
        )

    elif mode == "💰 Cloud TCO & $/1M Tokens":
        st.title("💰 Cloud TCO & Unit Economics Sizer ($/1M Tokens)")
        st.markdown(
            "Calculate memory-bandwidth decode throughput, TPOT latency, and $/1M tokens across RunPod, Lambda, AWS, and GCP."
        )

        gpu_id = st.selectbox(
            "Select Cloud Accelerator:",
            list(GPU_PRESETS.keys()),
            index=list(GPU_PRESETS.keys()).index("rtx4090"),
        )
        gpu_prof = get_gpu_profile(gpu_id)

        cost_res = estimate_serving_cost(
            model=arch,
            gpu=gpu_prof,
            precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            tensor_parallel=tp_degree,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Unit Cost / 1M Tokens", f"${cost_res.cost_per_million_tokens_usd:.4f}")
        with m2:
            st.metric("TPOT Latency", f"{cost_res.time_per_output_token_ms:.1f} ms")
        with m3:
            st.metric("Throughput / Stream", f"{cost_res.tokens_per_sec_per_user:.1f} tok/s")
        with m4:
            st.metric("Monthly 24/7 Hosting", f"${cost_res.monthly_dedicated_cost_usd:,.2f}")

        st.markdown("### ☁️ Cross-Cloud Comparison Matrix")
        cloud_report = compare_cloud_costs_for_workload(
            model=arch,
            precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            tensor_parallel=tp_degree,
        )
        cloud_rows = []
        for c in cloud_report.results:
            cloud_rows.append(
                {
                    "GPU Accelerator": c.gpu_name,
                    "Hourly Rate ($/hr)": f"${c.hourly_rate_usd:.2f}",
                    "Cluster Hourly ($/hr)": f"${c.total_cluster_hourly_usd:.2f}",
                    "TPOT Latency": f"{c.time_per_output_token_ms:.1f} ms",
                    "Throughput / Stream": f"{c.tokens_per_sec_per_user:.1f} tok/s",
                    "$/1M Tokens": f"${c.cost_per_million_tokens_usd:.4f}",
                    "Monthly 24/7 Hosting": f"${c.monthly_dedicated_cost_usd:,.2f}",
                }
            )
        st.dataframe(cloud_rows, use_container_width=True)

    elif mode == "🖥️ Live Local Hardware Check":
        st.title("🖥️ Local Hardware Zero-OOM Verification")
        st.markdown(
            "Verify whether this model configuration will fit into physical VRAM without crashing with an Out-Of-Memory (OOM) error."
        )

        st.info(
            "💡 **Hosted Web Demo Note:** Web browsers run sandboxed and cannot access client hardware or GPU drivers. "
            "Below, you can either **simulate your local GPU** (RTX 4090, 3060, Mac Unified Memory, etc.) or inspect the live cloud server process."
        )

        source_tab = st.radio(
            "Target Device Source:",
            [
                "🎮 Simulate My Physical Rig / GPU Preset",
                "🖥️ Live Host Auto-Detection (Server Process / CLI)",
            ],
            horizontal=True,
        )

        if source_tab == "🎮 Simulate My Physical Rig / GPU Preset":
            sim_col1, sim_col2 = st.columns([2, 1])
            with sim_col1:
                preset_options = [
                    "NVIDIA GeForce RTX 4090 (24 GB VRAM)",
                    "NVIDIA GeForce RTX 4080 (16 GB VRAM)",
                    "NVIDIA GeForce RTX 3060 (12 GB VRAM)",
                    "NVIDIA GeForce RTX 4060 (8 GB VRAM)",
                    "NVIDIA GeForce RTX 3050 Laptop (6 GB VRAM)",
                    "Apple M3 Max Unified Memory (36 GB)",
                    "Apple M3 Max Unified Memory (128 GB)",
                    "Custom Hardware VRAM Budget",
                ]
                selected_device = st.selectbox(
                    "Choose Your GPU / Silicon Model:", preset_options, index=0
                )

            if "RTX 4090" in selected_device:
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name="NVIDIA GeForce RTX 4090 (24GB)",
                    total_memory_gb=24.0,
                    free_memory_gb=22.2,
                    matched_gpu_id="rtx4090",
                )
            elif "RTX 4080" in selected_device:
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name="NVIDIA GeForce RTX 4080 (16GB)",
                    total_memory_gb=16.0,
                    free_memory_gb=14.8,
                    matched_gpu_id="rtx4080",
                )
            elif "RTX 3060" in selected_device:
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name="NVIDIA GeForce RTX 3060 (12GB)",
                    total_memory_gb=12.0,
                    free_memory_gb=11.0,
                    matched_gpu_id="rtx3060",
                )
            elif "RTX 4060" in selected_device:
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name="NVIDIA GeForce RTX 4060 (8GB)",
                    total_memory_gb=8.0,
                    free_memory_gb=7.2,
                    matched_gpu_id="rtx4060",
                )
            elif "RTX 3050" in selected_device:
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name="NVIDIA GeForce RTX 3050 Laptop (6GB)",
                    total_memory_gb=6.0,
                    free_memory_gb=5.3,
                    matched_gpu_id="rtx3050",
                )
            elif "36 GB" in selected_device:
                hw = DetectedHardware(
                    device_type="apple_silicon",
                    device_name="Apple M3 Max Unified Memory (36GB)",
                    total_memory_gb=36.0,
                    free_memory_gb=30.0,
                    matched_gpu_id="m3-max-36gb",
                )
            elif "128 GB" in selected_device:
                hw = DetectedHardware(
                    device_type="apple_silicon",
                    device_name="Apple M3 Max Unified Memory (128GB)",
                    total_memory_gb=128.0,
                    free_memory_gb=112.0,
                    matched_gpu_id="m3-max-128gb",
                )
            else:
                with sim_col2:
                    custom_vram = st.number_input(
                        "Custom VRAM (GB):",
                        min_value=1.0,
                        max_value=256.0,
                        value=16.0,
                        step=1.0,
                    )
                hw = DetectedHardware(
                    device_type="cuda",
                    device_name=f"Custom Device ({custom_vram:.0f} GB)",
                    total_memory_gb=float(custom_vram),
                    free_memory_gb=round(float(custom_vram) * 0.9, 1),
                    matched_gpu_id=None,
                )
        else:
            hw = detect_local_hardware()
            if hw.device_type == "cpu":
                st.warning(
                    f"⚠️ **Server Host CPU Detected ({hw.device_name})**: The active Python process has no CUDA or Metal accelerator attached. "
                    "When accessed via Streamlit Cloud, this represents the hosting container's system RAM, not your personal computer."
                )

        fit = verify_local_fit(
            arch=arch,
            precision=quant_choice,
            kv_cache_dtype=kv_choice,
            batch_size=batch_sz,
            context_tokens=context_len,
            hardware=hw,
        )

        st.markdown(
            f"**Target Hardware:** `{hw.device_name}` &nbsp;|&nbsp; "
            f"**Total Capacity:** `{hw.total_memory_gb:.1f} GB` &nbsp;|&nbsp; "
            f"**Estimated Free:** `{hw.free_memory_gb:.1f} GB`"
        )

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Workload VRAM", f"{fit.required_memory_gb:.2f} GB")
        with h2:
            st.metric("Total Headroom", f"{fit.headroom_total_gb:+.2f} GB")
        with h3:
            st.metric("Currently Free Headroom", f"{fit.headroom_free_gb:+.2f} GB")

        if fit.fits_currently:
            st.success(f"### {fit.verdict}")
        elif fit.fits_total:
            st.warning(f"### {fit.verdict}")
        else:
            st.error(f"### {fit.verdict}")

        st.divider()
        st.markdown(
            """
            ### 💻 Run Zero-Config Detection on Your Real Machine
            To let `infersize` directly probe your local NVIDIA drivers or Apple Silicon with zero manual input:
            ```bash
            # 1. Install infersize via pip
            pip install infersize

            # 2. Check your physical GPU and live free VRAM directly
            infersize check

            # 3. Or launch this exact Studio running locally on your hardware
            infersize ui
            ```
            """
        )


if __name__ == "__main__":
    main()
