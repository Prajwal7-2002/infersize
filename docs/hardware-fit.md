# Hardware Zero-OOM Verifier (`infersize fit`)

Theoretical benchmarks assume ideal GPU clusters. In practice, developers test models on their local workstations or laptops (RTX 3050, RTX 4090, Apple Silicon, or CPU RAM).

`infersize fit` provides a **live zero-OOM verification probe** that inspects your physical hardware directly.

---

## 1. How It Works

1. **NVML / CUDA Probe**: Queries the active NVIDIA driver and PyTorch CUDA context to determine total VRAM, currently allocated VRAM, and actual unreserved headroom.
2. **System RAM Fallback**: If no GPU is present or CUDA is uninitialized, probes total host RAM using OS primitives.
3. **Safety Headroom Analysis**:
   * Evaluates if model weights + KV-cache + activation buffer fits within physical memory.
   * Emits an amber warning if headroom is $<15\%$.
   * Emits a red Out-of-Memory alert if required VRAM exceeds physical capacity.

---

## 2. CLI Usage

Verify if a model can run on your current physical machine:

```bash
# Verify Qwen 2.5 7B at INT4 precision on your local GPU
infersize fit --model qwen2.5-7b --quant int4 --context-length 4096
```

### Sample Output:
```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            LIVE HARDWARE ZERO-OOM VERIFIER                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Detected Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total VRAM:        6.00 GB
Free Headroom:     4.82 GB (1.18 GB used by display / OS)

Workload Analysis:
• Model:           qwen2.5-7b (int4 blockwise)
• Context Length:  4,096 tokens (batch size = 1)
• Required VRAM:   4.42 GB
• Headroom Margin: +0.40 GB (8.3% safety margin)

Result:            [PASS] Model fits within local physical VRAM.
Recommendation:    Keep batch size <= 1 to avoid KV-cache allocation thrashing.
```
