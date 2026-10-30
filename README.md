# infersize ⚡

> **Analytical Performance Modeling & VRAM Capacity Planner for LLM Serving.**  
> Predict exact memory footprints, KV-cache scaling, multi-GPU comparison matrices, and hardware limits before downloading a model or spending thousands on GPU clusters.

[![CI](https://github.com/Prajwal7-2002/infersize/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/infersize/actions)
[![PyPI version](https://img.shields.io/pypi/v/infersize.svg?color=blue)](https://pypi.org/project/infersize/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Zero GPU Required](https://img.shields.io/badge/Compute-Zero%20GPU%20Required-success.svg)](https://github.com/Prajwal7-2002/infersize)

---

## 🎯 Why `infersize`?

Every engineer deploying local or cloud LLMs asks the same questions:
* *"Can I run Qwen-2.5-7B on my 6GB RTX 3050 laptop?"*
* *"Which GPU should I rent on RunPod or AWS for the lowest cost without running out of memory?"*
* *"How many concurrent users can an 80GB A100 serve at 4,000 context length before crashing with Out-Of-Memory (OOM)?"*
* *"How much VRAM does INT8 KV-cache actually save compared to FP16?"*

Renting GPU clusters just to discover a model runs out of memory costs time and money.  
**`infersize` solves this with rigorous analytical physics**: it computes exact parameter bytes, KV-cache tensor dimensions, FlashAttention activation peaks, and multi-accelerator capacity rankings in **< 10 milliseconds on any CPU**.

---

## 🚀 Key Features

* **Rigorous Mathematical Accounting**: Derives exact parameter counts and tensor bytes directly from architectural specs (layers, hidden dims, GQA heads, SwiGLU FFNs).
* **Multi-GPU Comparison Matrix (`infersize compare`)**: Evaluates a model workload across candidate accelerators side-by-side, flags OOM deficits, and identifies the **Right-Sized / Cost-Effective GPU** vs **Peak Throughput GPU**.
* **Tensor Parallelism (TP) Modeling**: Partitions weights, KV cache, and activations across $N$ GPUs analytically.
* **Multi-Precision Support**: Models memory footprint across `FP32`, `FP16`/`BF16`, `INT8`/`FP8`, `INT4`/`FP4`, and `BitNet 1.58b`.
* **Sub-Byte KV-Cache Scaling**: Accurately models KV-cache savings for `FP16`, `INT8`, and `INT4` KV quantization.
* **Capacity Solvers**:
  * `max-batch`: Solves for the maximum concurrent users a GPU can host before VRAM exhaustion.
  * `max-context`: Solves for the maximum supported sequence length before OOM.
* **Machine-Readable Automation (`--json`)**: Emits structured JSON schemas for CI/CD gates and infrastructure automation.
* **Hardware Profiles Built-in**: NVIDIA RTX 3050, 4060, 4090, Tesla T4, L4, A100 (40GB/80GB), H100 (80GB), H200 (141GB), and Apple Silicon M3 Max (36GB/128GB).
* **Zero GPU Dependencies**: Pure Python, runs instantaneously on Windows, macOS, and Linux.

---

## 📦 Quickstart

```bash
# Install via pip / uv
pip install infersize
```

### 1. Multi-GPU Comparison Matrix

Find the optimal accelerator for a workload and eliminate over-provisioning:

```bash
infersize compare --model qwen2.5-7b --quant fp16 --batch-size 1 --context 2048
```

Output:
```text
╭───────────── ⚡ INFERSIZE: MULTI-GPU HARDWARE COMPARISON MATRIX ─────────────╮
│ Model: qwen2.5-7b | Weights: FP16 | KV-Cache: FP16 | Batch: 1 | Context:     │
│ 2,048 tokens | TP: 1                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────┬──────────┬──────────┬──────────┬──────────┬───────────┬───────────┬───────────┬──────────────────────────────╮
│ GPU      │     VRAM │ Required │ Util (%) │ Headroom │ Max Batch │ Bandwidth │ Status    │ Sizing Verdict               │
├──────────┼──────────┼──────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────────────────────────┤
│ rtx3050  │   6.0 GB │  16.1 GB │   100.0% │ -10.1 GB │    0 reqs │  288 GB/s │ ✘ OOM     │ Deficit of 10.1 GB.          │
│ rtx4060  │   8.0 GB │  16.2 GB │   100.0% │  -8.2 GB │    0 reqs │  272 GB/s │ ✘ OOM     │ Deficit of 8.2 GB.           │
│ rtx4090  │  24.0 GB │  16.4 GB │    68.2% │  +7.6 GB │   62 reqs │ 1008 GB/s │ ⭐ OPTIMAL│ Right-sized: Optimal cost.   │
│ t4       │  16.0 GB │  16.2 GB │   100.0% │  -0.2 GB │    0 reqs │  320 GB/s │ ✘ OOM     │ Deficit of 0.2 GB.           │
│ l4       │  24.0 GB │  16.3 GB │    67.8% │  +7.7 GB │   63 reqs │  300 GB/s │ ✔ FITS    │ Healthy capacity headroom.   │
│ a100-40gb│  40.0 GB │  16.6 GB │    41.4% │ +23.4 GB │  197 reqs │ 1555 GB/s │ ✔ FITS    │ Healthy capacity headroom.   │
│ a100-80gb│  80.0 GB │  16.7 GB │    20.8% │ +63.3 GB │  537 reqs │ 2039 GB/s │ ℹ OVERKILL│ Excess VRAM headroom.        │
│ h100-80gb│  80.0 GB │  16.9 GB │    21.1% │ +63.1 GB │  535 reqs │ 3350 GB/s │ ℹ OVERKILL│ Excess VRAM headroom.        │
╰──────────┴──────────┴──────────┴──────────┴──────────┴───────────┴───────────┴───────────┴──────────────────────────────╯
╭─────────────────── 🎯 PROCUREMENT & SIZING RECOMMENDATION ───────────────────╮
│ ⭐ Recommended (Right-Sized / Cost-Effective): rtx4090 - Safest fit with     │
│ minimal idle VRAM waste and lowest hardware rental cost.                     │
│ 🚀 Recommended (Peak Throughput): h100-80gb - Highest memory bandwidth       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### 2. Predict Detailed VRAM Footprint & Headroom

```bash
infersize vram --model qwen2.5-7b --gpu rtx3050 --quant int4 --context 4096
```

Output:
```text
╭────────────────── ⚡ INFERSIZE: VRAM CAPACITY ANALYSIS ──────────────────────╮
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

### 3. Solve for Maximum Concurrent Serving Batch Size

```bash
infersize max-batch --model qwen2.5-7b --gpu rtx3050 --quant int4 --context 2048
```

### 4. Solve for Maximum Context Window

```bash
infersize max-context --model qwen2.5-7b --gpu rtx3050 --quant int4 --batch-size 1
```

### 5. Inspect Built-in Hardware & Models

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
uv run pytest -v
```

---

## 📄 License

Apache 2.0. Authored by **Prajwal Pujari** (`prajwalpujari16@gmail.com`).
