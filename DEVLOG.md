# DEVLOG: infersize

## 2026-10-29: Project Inception & Phase 1 VRAM Capacity Modeling

- Initialized Project #5 `infersize` (Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving).
- Strategic positioning: Pure-Python, zero-dependency analytical performance modeling tool predicting VRAM limits, KV-cache footprints, and hardware roofline saturation before spending money on cloud GPU compute.
- Initialized Apache 2.0 license, `.gitignore`, and `pyproject.toml` with `hatchling` build backend and `uv` package manager.
- Implemented Phase 1 Analytical VRAM Engine:
  - Exact transformer parameter accounting from model configuration (Embeddings, Q/K/V/O projections, SwiGLU dense/MoE FFNs, RMSNorms).
  - Open-weights model presets: Qwen 2.5 (0.5B, 1.5B, 7B, 14B, 32B, 72B), Llama 3 (8B, 70B), Mistral 7B, Phi-3 Mini, GPT-2.
  - Hardware profiles: NVIDIA RTX 3050, RTX 4060, RTX 4090, Tesla T4, L4, A100 (40GB/80GB), H100 (80GB), H200 (141GB), Apple M3 Max (36GB/128GB).
  - Multi-precision weight footprint calculation (FP32, FP16/BF16, INT8/FP8, INT4/FP4, BitNet 1.58b).
  - Exact KV-cache memory equation: $2 \times L \times H_{\text{kv}} \times D \times B$.
  - Maximum context window solver (`find_max_context_length`) and concurrent stream solver (`find_max_concurrent_batch_size`).
- Implemented rich Typer CLI (`infersize`):
  - `vram`: Detailed component memory breakdown (weights, KV, activations, driver) with visual ASCII allocation bar.
  - `max-batch`: Safe concurrent serving stream sizing.
  - `max-context`: Context window capacity before OOM.
  - `models` & `gpus`: Catalog tables with architecture stats and Silicon ridge points.
- 6/6 unit test suites passing with 100% assertions verified. All Ruff checks passing.

---

## 2026-10-30: PR #1 - Multi-GPU Hardware Comparison Matrix Engine

- **Branch**: `feat/gpu-comparison-matrix` -> **Merged via PR #1**.
- Implemented `infersize.analytics.comparison.compare_gpus_for_workload`:
  - Evaluates target serving workloads (model, quant, KV dtype, batch size, context length, TP) across all candidate accelerators in parallel.
  - Computes exact memory required, headroom deficit, utilization percentage, max concurrent batch size, and memory bandwidth for each GPU.
  - Automated procurement sizing heuristics:
    - **Right-Sized / Cost-Effective GPU**: Minimal idle VRAM waste, lowest hardware tier fitting the workload safely.
    - **Peak Throughput GPU**: Highest memory bandwidth among fitting GPUs.
- Implemented CLI command `infersize compare`:
  - Rich-formatted side-by-side terminal comparison table with status indicators (`⭐ OPTIMAL`, `✔ FITS`, `ℹ OVERKILL`, `✘ OOM`).
  - `--gpus` filtering (e.g. `--gpus rtx4090,a100-80gb,l4`).
  - `--json` machine-readable output format for infrastructure CI gates.
  - `to_markdown()` Markdown table generator for documentation export.
- Added 7 comprehensive unit tests in `tests/test_comparison.py`.

---

## 2026-10-30: PR #2 - Cloud Cost & Memory-Bandwidth Decode Roofline Engine

- **Branch**: `feat/cloud-cost-calculator` -> **Merged via PR #2**.
- Implemented `infersize.analytics.cost`:
  - Memory-bandwidth bound autoregressive decode throughput roofline modeling:
    $$\text{Decode Throughput (tok/s)} = \frac{B \cdot \text{BW} \cdot \eta}{W_{\text{bytes}} + \text{KV}_{\text{bytes}}}$$
  - Computes Time Per Output Token (TPOT latency per user stream in ms).
  - Computes exact unit operational economics: **\$ / 1 Million Tokens** and monthly 24/7 dedicated hosting cost (730 continuous hours).
  - Built-in cloud GPU pricing registry (`infersize.hardware.pricing`) for **RunPod**, **Lambda Labs**, **AWS EC2**, and **Google Cloud (GCP)**.
- Implemented CLI command `infersize cost`:
  - Panel header with workload summary, cluster cost, and fitting status.
  - Full metrics breakdown table with physical and financial engineering insights.
  - `--compare-clouds`: Side-by-side price comparison across all supported cloud providers for a single GPU.
  - `--compare-all`: Multi-accelerator cost, throughput, and \\\$/1M tokens comparison matrix.
  - `--rate`: Custom hourly rental rate override per GPU.
- Added 9 unit tests in `tests/test_cost.py` verifying throughput, TP cluster cost, and cloud price lookups.

---

## 2026-10-30: PR #3 - Architecture RFC Report Generator & v0.2.0 Bump

- **Branch**: `feat/report-export` -> **Merged via PR #3**.
- Implemented `infersize.analytics.report.generate_architecture_report`:
  - Synthesizes workload configuration, physical memory breakdowns, multi-GPU comparison matrix, and cloud TCO economics into an engineering specification.
  - Exports publication-ready Markdown RFC specifications (`to_markdown()`) ready to drop into GitHub PRs, Engineering RFCs, or infrastructure tickets.
  - Formulates LaTeX equations for KV-cache, decode roofline throughput, and unit economics.
- Implemented CLI command `infersize report`:
  - Renders syntax-highlighted Markdown directly to terminal console or writes directly to file via `--output <path.md>`.
  - Machine-readable `--json` schema output for automated DevOps pipelines.
- Bumped package version to `v0.2.0` in `pyproject.toml` and `src/infersize/__init__.py`.
- Added 6 unit tests in `tests/test_report.py`.

---

## 2026-10-31: PR #4 - Dynamic Hugging Face Hub Auto-Ingestion, MoE Active Parameter Sizing & PyPI Release CI

- **Branch**: `feat/hf-auto-spec` -> **Merged via PR #4**.
- Implemented `src/infersize/models/loader.py`:
  - Dynamic Hugging Face Hub `config.json` ingestion using Python standard library `urllib.request`. Zero external heavy dependencies (no PyTorch, Transformers, or Hugging Face Hub SDK required).
  - Fetches and parses any open-weights architecture in <300ms from `https://huggingface.co/<org>/<repo>/raw/main/config.json`.
  - Local custom file support via `load_model_arch_from_config(path)`.
  - Automatic repo name detection: string containing `/` (e.g. `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B`) triggers automatic Hub resolution.
  - Gated model authorization via `--hf-token` or `HF_TOKEN` environment variable.
- Mixture-of-Experts (MoE) Physical Sizing:
  - Differentiated **Total Parameters** ($E$ experts held in static VRAM) from **Active Parameters** (top-$K$ routed experts streaming per token step).
  - Added `ModelArch.calculate_active_parameters()` in `specs.py`.
  - Updated decode memory traffic in `cost.py` to stream active parameters across the memory bus, matching true vLLM/SGLang/TensorRT-LLM decode physics.
- Wired `--hf`, `--config`, and `--hf-token` across all CLI commands (`vram`, `compare`, `cost`, `report`, `max-batch`, `max-context`).
- Added PyPI Trusted Publisher automated workflow in `.github/workflows/release.yml`.
- Added 8 unit tests in `tests/test_hf_loader.py`. Test suite expanded to 36 tests with 95% code coverage and zero Ruff warnings.
