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

