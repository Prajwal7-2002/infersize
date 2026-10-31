# Speculative Decoding Sizer (`infersize spec`)

Speculative decoding leverages a smaller, high-throughput **Draft Model** to propose candidate tokens in parallel, which are then verified simultaneously by a larger **Target Model** in a single batched forward pass.

---

## 1. Leviathan Theoretical Speedup Equation

Based on the canonical derivation from *Leviathan et al. (Fast Inference from Transformers via Speculative Decoding)*, the wall-clock speedup $S$ achieved over autoregressive generation is given by:

$$S = \frac{1 + \alpha \cdot K}{1 + \frac{K + 1}{\gamma}}$$

Where:
* $K$: Number of speculative draft tokens generated per verification step (lookahead window).
* $\alpha \in [0, 1]$: Empirical token acceptance rate of the draft model.
* $\gamma = \frac{\text{Target Model Step Latency}}{\text{Draft Model Step Latency}}$: Latency ratio between target and draft forward steps.

### Expected Accepted Tokens Per Step
$$\mathbb{E}[\text{Accepted Tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$
For high acceptance rates ($\alpha \ge 0.7$) and optimal draft lengths ($K \in [3, 6]$), speculative decoding routinely delivers **$1.8\times\text{--}2.6\times$ empirical speedups**.

---

## 2. Joint VRAM Memory Budgeting

While speculative decoding improves latency, it places strict memory demands on the accelerator: **both models must fit into physical VRAM simultaneously**.

$$\text{Total VRAM}_{\text{spec}} = \text{VRAM}_{\text{target}} + \text{VRAM}_{\text{draft}}$$

`infersize spec` calculates the combined memory footprint across precisions (e.g. Target FP16 + Draft INT4) to ensure the draft model does not trigger a catastrophic CUDA OOM.

---

## 3. CLI Example

```bash
infersize spec \
  --target qwen2.5-7b --target-quant fp16 \
  --draft qwen2.5-0.5b --draft-quant int4 \
  --gpu a100 --context-length 4096 --acceptance-rate 0.72 --draft-k 4
```

### Sample Output:
```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      SPECULATIVE DECODING JOINT SIZER & LEVIATHAN SPEEDUP   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Target Model:       qwen2.5-7b (fp16) -> 14.82 GB VRAM
Draft Model:        qwen2.5-0.5b (int4) -> 0.98 GB VRAM
Combined VRAM:      15.80 GB (Safe on NVIDIA A100 80GB, 19.8% utilized)
Acceptance Rate α:  0.72
Draft Length K:     4
Latency Ratio γ:    12.4x
Leviathan Speedup:  2.78x wall-clock acceleration
```
