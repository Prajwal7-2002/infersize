# CLI Command Reference

`infersize` provides a command-line interface powered by Typer and Rich.

---

## 1. `infersize estimate`
Estimate VRAM usage, weight footprint, and KV-cache breakdown for a single model:
```bash
infersize estimate --model qwen2.5-7b --quant fp16 --batch-size 4 --context-length 4096
```

### Options:
* `--model`, `-m`: Model preset name or local path (`qwen2.5-0.5b`, `qwen2.5-7b`, `llama-3-8b`, `deepseek-v2-lite`).
* `--config`, `-c`: Path to a custom Hugging Face `config.json`.
* `--quant`, `-q`: Precision (`fp16`, `bf16`, `int8`, `int4`, `fp8`, `bitnet`).
* `--batch-size`, `-b`: Concurrent batch size (default: `1`).
* `--context-length`, `-s`: Sequence length in tokens (default: `2048`).
* `--tensor-parallel`, `-tp`: Number of GPUs in tensor-parallel group (default: `1`).
* `--json-out`: Output pure JSON payload for CI/CD scripting.

---

## 2. `infersize compare`
Compare VRAM utilization and determine fit across multiple GPU architectures simultaneously:
```bash
infersize compare --model qwen2.5-7b --quant int4 --gpus rtx3050,rtx4090,a10g,l40s,a100-80gb
```

---

## 3. `infersize spec`
Calculate joint target + draft model VRAM footprint and theoretical Leviathan speedup:
```bash
infersize spec --target qwen2.5-7b --draft qwen2.5-0.5b --acceptance-rate 0.75 --draft-k 4
```

---

## 4. `infersize fit`
Verify zero-OOM safety against your current physical workstation hardware:
```bash
infersize fit --model qwen2.5-7b --quant int4 --context-length 4096
```

---

## 5. `infersize cost`
Calculate cloud TCO, hourly rental expenses, and cost per 1M tokens across providers:
```bash
infersize cost --model qwen2.5-7b --quant fp16 --clouds aws,runpod,lambda
```

---

## 6. `infersize report`
Generate a comprehensive Architecture Decision RFC document in Markdown or JSON:
```bash
infersize report --model qwen2.5-7b --quant fp16 --output rfc.md
```

---

## 7. `infersize ui`
Launch the interactive Streamlit Web Studio:
```bash
infersize ui --port 8501
```
