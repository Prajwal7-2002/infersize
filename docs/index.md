# infersize 📐

[![PyPI](https://img.shields.io/pypi/v/infersize.svg?color=blue)](https://pypi.org/project/infersize/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://infersize.streamlit.app/)
[![CI](https://github.com/Prajwal7-2002/infersize/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/infersize/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/CPU-Zero_GPU_Required-green.svg)](#)

A **deterministic, first-principles analytical capacity planner, VRAM sizing engine, speculative decoding budgeter, and multi-GPU matrix comparator** for Large Language Model (LLM) inference.

---

## Why infersize?

Deploying large language models often results in expensive surprises:
1. **CUDA Out-of-Memory (OOM) crashes** after waiting minutes to download 30GB+ model checkpoints.
2. **Massive KV-cache footprint** at long context lengths (e.g. 32K or 128K) causing serving latency spikes.
3. **Speculative Decoding sizing guesses** where target and draft models exceed total GPU VRAM headroom.
4. **Cloud TCO over-provisioning** by renting expensive 8x H100 clusters when a single quantized 4x L40S or A10G pool was mathematically optimal.

`infersize` solves this by computing exact weight bytes, SwiGLU expansion dimensions, GQA KV-cache scaling, activation tensors, and CUDA runtime context overhead in **under 10 milliseconds on your local CPU** — before downloading weights or renting clusters.

---

## ⚡ Core Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              infersize                                 │
│      (Analytical Performance Modeling & VRAM Capacity Planning)        │
│                                                                        │
│  ┌───────────────────────────────────┐ ┌─────────────────────────────┐ │
│  │ 1. Parameter & Weight Sizer       │ │ 2. GQA KV-Cache Engine      │ │
│  │ • Exact SwiGLU 3x gate+up proj    │ │ • Sub-byte FP16/INT8/INT4   │ │
│  │ • MoE active routing (DeepSeek)   │ │ • Context scaling (1K..128K)│ │
│  │ • FP16, INT8, INT4, FP8, BitNet   │ │ • Linear sequence budgeting │ │
│  └─────────────────┬─────────────────┘ └──────────────┬──────────────┘ │
│                    │                                  │                │
│                    └─────────────────┬────────────────┘                │
│                                      ▼                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ 3. Memory & Hardware Matrix Solvers                               │ │
│  │ • Multi-GPU side-by-side comparison (A100, H100, L40S, RTX 3050)  │ │
│  │ • Right-Sized vs Peak Throughput recommendations                  │ │
│  │ • Speculative Decoding Leviathan Speedup & Joint VRAM Planner     │ │
│  │ • Local Physical Hardware Zero-OOM Verifier (NVML / PyTorch)      │ │
│  │ • Cloud TCO (\$ / 1M tokens) across AWS, GCP, Azure, RunPod, Lambda│ │
│  └───────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Quick Installation

```bash
# Core CLI engine
pip install infersize

# Or via uv
uv add infersize

# With interactive Streamlit Web Studio
pip install "infersize[ui]"
```

---

## 🚀 Quick Navigation

* **[VRAM Analytical Physics](vram-physics.md)** — First-principles mathematical derivation of parameter counting, SwiGLU, and GQA KV-cache virtual memory.
* **[Speculative Sizer](speculative-sizer.md)** — Leviathan speedup derivations ($S = \frac{1 + \alpha K}{1 + (K+1)/\gamma}$) and joint target + draft VRAM budgeting.
* **[Hardware Zero-OOM Verifier](hardware-fit.md)** — Real-time NVML hardware querying to verify zero-OOM safety on physical silicon.
* **[Web Studio & Streamlit](web-studio.md)** — Launching and using the interactive browser GUI (`infersize ui`).
* **[CLI Reference](cli-reference.md)** — Complete command-line manual (`estimate`, `compare`, `spec`, `fit`, `ui`).
