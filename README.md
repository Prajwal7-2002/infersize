# infersize ⚡

> **Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving.**  
> Predict exact memory footprints, KV-cache scaling, and hardware limits before downloading a model or spending thousands on GPU clusters.

[![CI](https://github.com/Prajwal7-2002/infersize/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/infersize/actions)
[![PyPI version](https://img.shields.io/pypi/v/infersize.svg?color=blue)](https://pypi.org/project/infersize/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Zero GPU Required](https://img.shields.io/badge/Compute-Zero%20GPU%20Required-success.svg)](https://github.com/Prajwal7-2002/infersize)

---

## 🎯 Why `infersize`?

Every engineer deploying local or cloud LLMs asks the same questions:
* *"Can I run Qwen-2.5-7B on my 6GB RTX 3050 laptop?"*
* *"How many concurrent users can an 80GB A100 serve at 4,000 context length before crashing with Out-Of-Memory (OOM)?"*
* *"How much VRAM does INT8 KV-cache actually save compared to FP16?"*

Renting GPU clusters just to discover a model runs out of memory costs time and money.  
**`infersize` solves this with rigorous analytical physics**: it computes exact parameter bytes, KV-cache tensor dimensions, FlashAttention activation peaks, and hardware roofline saturation in **< 10 milliseconds on any CPU**.

---

## 🚀 Key Features

* **Rigorous Mathematical Accounting**: Derives exact parameter counts and tensor bytes directly from architectural specs (layers, hidden dims, GQA heads, SwiGLU FFNs).
* **Multi-Precision Support**: Models memory footprint across `FP32`, `FP16`/`BF16`, `INT8`/`FP8`, `INT4`/`FP4`, and `BitNet 1.58b`.
* **Sub-Byte KV-Cache Scaling**: Accurately models KV-cache savings for `FP16`, `INT8`, and `INT4` KV quantization.
* **Capacity Solvers**:
  * `max-batch`: Solves for the maximum concurrent users a GPU can host before VRAM exhaustion.
  * `max-context`: Solves for the maximum supported sequence length before OOM.
* **Hardware Profiles Built-in**: NVIDIA RTX 3050, 4060, 4090, Tesla T4, L4, A100 (40GB/80GB), H100 (80GB), H200 (141GB), and Apple Silicon M3 Max (36GB/128GB).
* **Zero GPU Dependencies**: Pure Python, runs instantaneously on Windows, macOS, and Linux.

---

## 📦 Quickstart

```bash
# Install via pip / uv
pip install infersize
```

### 1. Predict VRAM Capacity & Headroom

```bash
infersize vram --model qwen2.5-7b --gpu rtx3050 --quant int4 --context 4096
```

Output:
```text
╭────────────────── ⚡ LLM-PERFMODEL: VRAM CAPACITY ANALYSIS ──────────────────╮
│ Model: qwen2.5-7b (7.62B params) | GPU: NVIDIA GeForce RTX 3050 Laptop (6GB) │
│ Precision: Weights: INT4 | KV-Cache: FP16 | Batch: 1 | Context: 4096 tokens  │
│ Verdict: ✔ WORKLOAD FITS IN VRAM (78.3% utilized)                            │
╰──────────────────────────────────────────────────────────────────────────────╯
 Component      Memory (GB)  Share (%)  Key Physical Insight
────────────────────────────────────────────────────────────────────────────────
 Model Weights      3.81 GB      63.5%  Compressed 4.0x from FP16 baseline
 KV Cache           0.23 GB       3.8%  57,344 bytes/token (B=1, S=4096)
 Activations        0.06 GB       1.0%  FlashAttention linear peak memory buffer
 CUDA Context       0.60 GB      10.0%  Driver runtime reservation & workspace
────────────────────────────────────────────────────────────────────────────────
 TOTAL REQUIRED     4.70 GB      78.3%  Total physical VRAM demand
 HEADROOM          +1.30 GB      21.7%  VRAM remaining for additional batch

Memory Allocation: [█████████████████████████▉▌░░░░░░░░]
Legend: ■ Weights  ■ KV-Cache  ■ Activations  ■ Driver  ░ Free
```

### 2. Solve for Maximum Concurrent Serving Batch Size

```bash
infersize max-batch --model qwen2.5-7b --gpu rtx3050 --quant int4 --context 2048
```

### 3. Solve for Maximum Context Window

```bash
infersize max-context --model qwen2.5-7b --gpu rtx3050 --quant int4 --batch-size 1
```

### 4. Inspect Built-in Hardware & Models

```bash
infersize models
infersize gpus
```

---

## 📐 The Mathematical Mechanics

### 1. KV-Cache Memory Equation
$$\text{KV Bytes per Token} = 2 \times L \times H_{\text{kv}} \times D \times B$$
* $L$: Number of transformer layers
* $H_{\text{kv}}$: Number of Key-Value attention heads (Grouped-Query Attention)
* $D$: Head dimension ($\frac{d_{\text{model}}}{H_{\text{q}}}$)
* $B$: Bytes per element ($2$ for FP16, $1$ for INT8, $0.5$ for INT4)

### 2. Maximum Concurrent Batch Size Solver
$$B_{\max} = \left\lfloor \frac{\text{VRAM}_{\text{usable}} - \text{Memory}_{\text{weights}} - \text{Memory}_{\text{act}}}{\text{KV Bytes per Token} \times S} \right\rfloor$$

---

## 🧪 Testing

```bash
uv run pytest
```

---

## 📄 License

Apache 2.0. Authored by **Prajwal Pujari** (`prajwalpujari16@gmail.com`).

