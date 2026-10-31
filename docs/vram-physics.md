# VRAM Analytical Physics & First-Principles Modeling

`infersize` models GPU memory requirements using first-principles analytical formulas calibrated against empirical measurements from modern inference runtimes.

---

## 1. Parameter Footprint Derivation

For a Transformer model with:
* Hidden dimension $d_{\text{model}}$
* Intermediate MLP dimension $d_{\text{ffn}}$
* Attention heads $N_{\text{heads}}$
* Key/Value heads $N_{\text{kv}}$ (Grouped-Query Attention)
* Number of layers $L$
* Vocabulary size $V$

### Attention Layer Parameters
$$\text{Params}_{\text{attn}} = 2 \cdot d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot (N_{\text{kv}} \cdot D_{\text{head}})$$
where $D_{\text{head}} = \frac{d_{\text{model}}}{N_{\text{heads}}}$. In standard Multi-Head Attention (MHA), $N_{\text{kv}} = N_{\text{heads}}$, giving $4 \cdot d_{\text{model}}^2$.

### SwiGLU Feed-Forward Network (FFN)
Standard ReLU/GELU MLPs utilize 2 projection matrices ($W_{\text{gate}}$ and $W_{\text{down}}$):
$$\text{Params}_{\text{FFN}} = 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$$

Modern architectures (Llama, Mistral, Qwen, DeepSeek) utilize gated **SwiGLU** activations, requiring **3 linear projection matrices** ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$):
$$\text{Params}_{\text{SwiGLU}} = 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}}$$

### Total Parameter Memory
$$\text{Memory}_{\text{weights}} = \text{Total Parameters} \times \text{Bytes Per Weight}$$

* **FP16 / BF16**: 2.0 bytes/param
* **INT8 / FP8**: 1.0 byte/param
* **INT4 / AWQ / GPTQ**: 0.5 bytes/param
* **BitNet 1.58b**: ~0.20 bytes/param

---

## 2. Grouped-Query Attention (GQA) KV-Cache Footprint

During autoregressive generation, past Key and Value vectors are cached to avoid redundant matrix operations. For sequence length $S$ and batch size $B$:

$$\text{KV Bytes Per Token} = 2 \cdot L \cdot N_{\text{kv}} \cdot D_{\text{head}} \cdot \text{dtype\_bytes}$$

$$\text{Total KV-Cache Memory} = B \cdot S \cdot \text{KV Bytes Per Token}$$

### The GQA Memory Compression Advantage
Comparing Llama-2-70B (MHA, $N_{\text{kv}} = 64$) vs Llama-3-70B (GQA, $N_{\text{kv}} = 8$):
* **MHA ($N_{\text{kv}} = 64$)**: 1.28 MB per token slot across 80 layers. At 32K context ($B=1$), KV cache requires **40.96 GB VRAM**!
* **GQA ($N_{\text{kv}} = 8$)**: 0.16 MB per token slot. At 32K context ($B=1$), KV cache requires only **5.12 GB VRAM** (**8x reduction**).

---

## 3. Activation & Runtime Overhead

1. **FlashAttention Peak Activation Memory**:
   $$\text{Memory}_{\text{act}} \approx B \cdot S \cdot d_{\text{model}} \cdot \text{dtype\_bytes}$$
2. **CUDA Context & Driver Reserve**:
   $$\text{Memory}_{\text{CUDA}} \approx 0.6\text{ GB to }1.2\text{ GB}$$
   Pre-allocated by PyTorch and the NVIDIA driver upon initializing the CUDA runtime.

$$\text{Total Required VRAM} = \frac{\text{Memory}_{\text{weights}}}{\text{Tensor Parallel}} + \text{Memory}_{\text{KV}} + \text{Memory}_{\text{act}} + \text{Memory}_{\text{CUDA}}$$
