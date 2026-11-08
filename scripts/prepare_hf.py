"""Prepare Hugging Face upload artifacts for infersize Streamlit Space."""

import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    space_dir = root / "hf_space"

    if space_dir.exists():
        shutil.rmtree(space_dir)
    space_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy app.py and requirements.txt
    shutil.copy2(root / "app.py", space_dir / "app.py")
    shutil.copy2(root / "requirements.txt", space_dir / "requirements.txt")

    if (root / "LICENSE").exists():
        shutil.copy2(root / "LICENSE", space_dir / "LICENSE")

    # 2. Copy src/ directory (ignoring __pycache__)
    src_dst = space_dir / "src"
    shutil.copytree(
        root / "src",
        src_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )

    # 3. Create Space README.md with YAML metadata
    space_readme = """---
title: Infersize - LLM Capacity & Sizing Studio
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
short_description: Interactive LLM VRAM sizing, multi-GPU comparison & cloud TCO.
---

# infersize ⚡ LLM Capacity & Sizing Studio

Interactive pre-flight analytical performance modeling, multi-GPU comparison matrix, speculative decoding sizer, and cloud TCO ($/1M tokens) capacity planner.

[![PyPI](https://img.shields.io/pypi/v/infersize.svg?color=blue)](https://pypi.org/project/infersize/)
[![GitHub](https://img.shields.io/badge/GitHub-infersize-181717?logo=github)](https://github.com/Prajwal7-2002/infersize)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://prajwal7-2002.github.io/infersize/)

### 🚀 Capabilities
- **VRAM Capacity Planning:** Analytical parameter calculation, KV-cache sizing (MHA, GQA, DeepSeek MLA), activation memory, and context scaling up to 128k.
- **Multi-GPU Comparison Matrix:** Instant fitting across NVIDIA H100, A100, L40S, RTX 4090, T4, and Apple Silicon with tensor parallelism ($TP=1, 2, 4, 8$).
- **Speculative Decoding Sizer:** Draft-target pair validation, memory overhead, and Leviathan acceptance speedup estimation.
- **Cloud TCO & Economics:** Real-time $/1M tokens cost modeling across AWS, GCP, Lambda Labs, and RunPod.
- **Dynamic Hub Ingestion:** Load any Hugging Face `config.json` on the fly.

**Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))
"""
    (space_dir / "README.md").write_text(space_readme, encoding="utf-8")
    print("✓ Successfully prepared hf_space directory for infersize!")


if __name__ == "__main__":
    main()

