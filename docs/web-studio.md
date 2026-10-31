# Web Studio & Streamlit App (`infersize ui`)

> 🚀 **Live Cloud Instance**: You can test the Web Studio directly in your browser without local installation at **[infersize.streamlit.app](https://infersize.streamlit.app/)**.

For interactive capacity planning, architecture design meetings, and visual RFC reviews, `infersize` includes a standalone **Streamlit Web Studio**.

---

## 1. Launching the Studio

Ensure the UI extra is installed:
```bash
pip install "infersize[ui]"
```

Launch the interactive studio:
```bash
# Direct CLI shortcut
infersize ui

# Or via standard streamlit
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## 2. Interactive Features

* **Real-Time Sliders**:
  * Adjust sequence length ($1\text{K}\text{--}128\text{K}$ tokens).
  * Adjust concurrency batch size ($1\text{--}64$ active streams).
  * Toggle Tensor Parallelism ($TP = 1, 2, 4, 8$).
  * Change weight precision (FP16, INT8, INT4, FP8, BitNet).
* **Multi-GPU Comparison Matrix**:
  * Live comparison table across RTX 3050, RTX 4090, A10G, L40S, A100 (40GB/80GB), and H100.
  * Automatic badge indicators: `[FIT]`, `[OOM]`, and `[RIGHT-SIZED]`.
* **Architecture RFC Exporter**:
  * One-click download of a production-ready Markdown report (`architecture_rfc.md`) detailing memory bounds, roofline decode throughput, and cloud TCO estimates (\$ / 1M tokens) for engineering stakeholders.
