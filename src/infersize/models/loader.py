"""Dynamic Hugging Face and local config.json architecture loader."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from infersize import __version__
from infersize.models.specs import MODEL_PRESETS, ModelArch, get_model_spec


def parse_hf_config_dict(data: dict[str, Any], model_name: str = "custom-model") -> ModelArch:
    """Parse a Hugging Face Transformers config dictionary into a ModelArch specification.

    Handles architectural differences across Llama, Qwen, Mistral, Gemma, DeepSeek,
    and Mixture-of-Experts (MoE) naming variations.
    """
    # 1. Number of layers
    num_layers = (
        data.get("num_hidden_layers")
        or data.get("n_layer")
        or data.get("num_layers")
        or data.get("n_layers")
        or 32
    )

    # 2. Hidden dimension
    hidden_size = (
        data.get("hidden_size")
        or data.get("n_embd")
        or data.get("d_model")
        or data.get("dim")
        or 4096
    )

    # 3. Attention heads
    num_heads = (
        data.get("num_attention_heads")
        or data.get("n_head")
        or data.get("num_heads")
        or data.get("n_heads")
        or 32
    )

    # 4. Key/Value heads (GQA/MQA)
    num_kv_heads = (
        data.get("num_key_value_heads")
        or data.get("num_kv_heads")
        or data.get("n_head_kv")
        or data.get("num_key_value_head")
        or num_heads  # Default to MHA if not specified
    )

    # 5. Head dimension
    head_dim = data.get("head_dim")
    if head_dim is None and num_heads > 0:
        head_dim = hidden_size // num_heads

    # 6. Intermediate FFN size
    intermediate_size = (
        data.get("intermediate_size")
        or data.get("n_inner")
        or data.get("intermediate_dim")
        or data.get("ffn_dim")
    )
    if intermediate_size is None:
        # Standard MLP heuristic: 4 * hidden, SwiGLU heuristic: ~2.7 * hidden
        intermediate_size = int(hidden_size * 2.7)

    # 7. Vocabulary size
    vocab_size = data.get("vocab_size", 32000)

    # 8. Embedding tying
    tie_word_embeddings = bool(data.get("tie_word_embeddings", False))

    # 9. MLP Activation Type (SwiGLU vs standard MLP)
    hidden_act = str(data.get("hidden_act", "")).lower()
    model_type = str(data.get("model_type", "")).lower()
    if (
        "silu" in hidden_act
        or "swiglu" in hidden_act
        or any(m in model_type for m in ["llama", "qwen", "mistral", "mixtral", "deepseek", "phi3"])
    ):
        mlp_type = "swiglu"
    else:
        mlp_type = "standard"

    # 10. Mixture-of-Experts (MoE) detection
    num_experts = (
        data.get("num_local_experts")
        or data.get("num_experts")
        or data.get("n_routed_experts")
        or data.get("num_routed_experts")
        or 1
    )
    num_experts_per_tok = (
        data.get("num_experts_per_tok")
        or data.get("num_selected_experts")
        or data.get("n_active_experts")
        or data.get("num_active_experts")
        or 1
    )
    is_moe = num_experts > 1

    family = model_type if model_type else "transformer"

    return ModelArch(
        name=model_name,
        family=family,
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(num_kv_heads),
        head_dim=int(head_dim) if head_dim else None,
        vocab_size=int(vocab_size),
        tie_word_embeddings=tie_word_embeddings,
        mlp_type=mlp_type,
        is_moe=is_moe,
        num_experts=int(num_experts),
        num_experts_per_tok=int(num_experts_per_tok),
    )


def load_model_arch_from_hf(
    repo_id: str,
    hf_token: str | None = None,
    timeout_s: float = 10.0,
) -> ModelArch:
    """Fetch and parse config.json for any model on Hugging Face using standard urllib.

    Zero external dependencies required (no transformers or huggingface_hub needed).
    """
    clean_repo = repo_id.strip()
    if clean_repo.startswith("https://huggingface.co/"):
        clean_repo = clean_repo.replace("https://huggingface.co/", "")
    clean_repo = clean_repo.strip("/")

    url = f"https://huggingface.co/{clean_repo}/raw/main/config.json"
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    headers = {
        "User-Agent": f"infersize/{__version__} (https://github.com/Prajwal7-2002/infersize)",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            raw_bytes = response.read()
            data = json.loads(raw_bytes.decode("utf-8"))
            return parse_hf_config_dict(data, model_name=clean_repo)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"Hugging Face repository '{clean_repo}' not found or config.json does not exist. "
                f"Please verify repository name on https://huggingface.co/{clean_repo}"
            ) from e
        elif e.code in (401, 403):
            raise PermissionError(
                f"Access denied for '{clean_repo}'. This model may be gated (e.g. Meta Llama). "
                "Please set your Hugging Face access token via the HF_TOKEN environment variable."
            ) from e
        raise RuntimeError(
            f"HTTP error {e.code} fetching config for '{clean_repo}': {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Failed to connect to Hugging Face hub for '{clean_repo}': {e.reason}. "
            "Please check network connection or provide a local config with --config."
        ) from e


def load_model_arch_from_config(path: str | Path) -> ModelArch:
    """Load and parse a model architecture from a local config.json file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local config file '{path}' not found.")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = p.parent.name if p.name == "config.json" else p.stem
    return parse_hf_config_dict(data, model_name=model_name)


def resolve_model_arch(
    model_identifier: str,
    hf_token: str | None = None,
    is_hf_flag: bool = False,
    config_path: str | None = None,
) -> ModelArch:
    """Intelligently resolve a model spec from local files, presets, or Hugging Face hub."""
    if config_path:
        return load_model_arch_from_config(config_path)

    # Check local path
    possible_path = Path(model_identifier)
    if possible_path.exists() and possible_path.is_file():
        return load_model_arch_from_config(possible_path)

    # If user explicitly passed --hf or identifier looks like an HF repo ('org/model')
    if is_hf_flag or "/" in model_identifier:
        return load_model_arch_from_hf(model_identifier, hf_token=hf_token)

    # Try presets
    try:
        return get_model_spec(model_identifier)
    except ValueError:
        # Fallback to HF attempt if preset not found
        try:
            return load_model_arch_from_hf(model_identifier, hf_token=hf_token)
        except Exception:
            available = list(MODEL_PRESETS.keys())
            raise ValueError(
                f"Unknown model '{model_identifier}'. Not in built-in presets: {available}. "
                "To load from Hugging Face hub, pass the repo id (e.g. --model meta-llama/Llama-3.1-8B-Instruct) "
                "or specify a local config with --config <path>."
            )
