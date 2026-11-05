"""Model architecture definitions and standard open-weights presets."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelArch(BaseModel):
    """Architectural parameters of a Transformer Language Model."""

    name: str = Field(description="Unique model identifier")
    family: str = Field(
        default="transformer", description="Architecture family (qwen, llama, mistral, moe)"
    )
    num_layers: int = Field(description="Number of transformer layers")
    hidden_size: int = Field(description="Hidden dimension size (d_model)")
    intermediate_size: int = Field(description="FFN intermediate layer dimension")
    num_attention_heads: int = Field(description="Number of query attention heads")
    num_key_value_heads: int = Field(description="Number of key/value heads for GQA/MQA")
    head_dim: int | None = Field(
        default=None,
        description="Head dimension size. Defaults to hidden_size // num_attention_heads",
    )
    vocab_size: int = Field(default=152064, description="Vocabulary token count")
    tie_word_embeddings: bool = Field(
        default=False, description="Whether input and output embeddings share weights"
    )
    mlp_type: str = Field(
        default="swiglu",
        description="MLP activation architecture: swiglu (3 matrices) or standard (2 matrices)",
    )
    is_moe: bool = Field(
        default=False, description="Whether the architecture uses Mixture of Experts"
    )
    num_experts: int = Field(default=1, description="Total experts in MoE layer")
    num_experts_per_tok: int = Field(default=1, description="Active routed experts per token")
    n_shared_experts: int = Field(
        default=0, description="Number of always-active shared experts (DeepSeek / Qwen MoE)"
    )
    moe_intermediate_size: int | None = Field(
        default=None,
        description="FFN intermediate dimension for routed experts (if distinct from dense FFN)",
    )
    first_k_dense_replace: int = Field(
        default=0,
        description="Number of initial bottom layers that remain dense before MoE layers begin",
    )

    # Multi-Head Latent Attention (MLA - DeepSeek-V2 / V3)
    is_mla: bool = Field(
        default=False,
        description="Whether the architecture uses Multi-Head Latent Attention (DeepSeek MLA)",
    )
    kv_lora_rank: int | None = Field(
        default=None,
        description="Compressed KV latent dimension (d_c in DeepSeek MLA, e.g. 512)",
    )
    q_lora_rank: int | None = Field(
        default=None,
        description="Compressed Query latent dimension in MLA (e.g. 1536)",
    )
    qk_rope_head_dim: int | None = Field(
        default=None,
        description="Decoupled RoPE head dimension for MLA (d_r in DeepSeek, e.g. 64)",
    )

    def model_post_init(self, __context: object) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def gqa_ratio(self) -> float:
        """Ratio of Query heads to KV heads (1.0 for MHA, >1.0 for GQA)."""
        return self.num_attention_heads / self.num_key_value_heads

    def calculate_total_parameters(self) -> int:
        """Analytically compute the total parameter count of the model."""
        # 1. Embeddings: vocab_size * hidden_size
        embed_params = self.vocab_size * self.hidden_size
        lm_head_params = 0 if self.tie_word_embeddings else (self.vocab_size * self.hidden_size)

        # 2. Per-layer Self-Attention:
        if self.is_mla:
            q_rank = self.q_lora_rank or 1536
            kv_rank = self.kv_lora_rank or 512
            rope_dim = self.qk_rope_head_dim or 64
            h_dim = self.head_dim or 128

            q_down = self.hidden_size * q_rank
            q_up = q_rank * (self.num_attention_heads * (h_dim + rope_dim))
            kv_down = self.hidden_size * (kv_rank + rope_dim)
            kv_up = kv_rank * (self.num_attention_heads * (h_dim + h_dim))
            o_proj = (self.num_attention_heads * h_dim) * self.hidden_size
            attn_per_layer = q_down + q_up + kv_down + kv_up + o_proj
        else:
            q_params = self.hidden_size * (self.num_attention_heads * self.head_dim)
            k_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
            v_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
            o_params = (self.num_attention_heads * self.head_dim) * self.hidden_size
            attn_per_layer = q_params + k_params + v_params + o_params

        # 3. Feed-Forward Network:
        ffn_matrix_factor = 3 if self.mlp_type.lower() == "swiglu" else 2
        dense_ffn_per_layer = ffn_matrix_factor * self.hidden_size * self.intermediate_size

        if self.is_moe:
            dense_layers = min(self.first_k_dense_replace, self.num_layers)
            moe_layers = max(0, self.num_layers - dense_layers)
            moe_inter = self.moe_intermediate_size or self.intermediate_size

            # Routed experts across MoE layers
            routed_ffn = (
                moe_layers * self.num_experts * (ffn_matrix_factor * self.hidden_size * moe_inter)
            )
            # Shared experts across MoE layers
            shared_ffn = (
                moe_layers
                * self.n_shared_experts
                * (ffn_matrix_factor * self.hidden_size * moe_inter)
            )
            # Initial dense layers
            initial_dense_ffn = dense_layers * dense_ffn_per_layer

            total_ffn_params = routed_ffn + shared_ffn + initial_dense_ffn
        else:
            total_ffn_params = dense_ffn_per_layer * self.num_layers

        # 4. Layer Norms (input layernorm + post_attention layernorm per layer + final norm)
        norm_params = (2 * self.hidden_size) * self.num_layers + self.hidden_size

        total_attn_params = attn_per_layer * self.num_layers

        return embed_params + lm_head_params + total_attn_params + total_ffn_params + norm_params

    def calculate_active_parameters(self) -> int:
        """Analytically compute active parameters evaluated per token (top-K experts for MoE)."""
        if not self.is_moe or self.num_experts <= 1:
            return self.calculate_total_parameters()

        embed_params = self.vocab_size * self.hidden_size
        lm_head_params = 0 if self.tie_word_embeddings else (self.vocab_size * self.hidden_size)

        if self.is_mla:
            q_rank = self.q_lora_rank or 1536
            kv_rank = self.kv_lora_rank or 512
            rope_dim = self.qk_rope_head_dim or 64
            h_dim = self.head_dim or 128

            q_down = self.hidden_size * q_rank
            q_up = q_rank * (self.num_attention_heads * (h_dim + rope_dim))
            kv_down = self.hidden_size * (kv_rank + rope_dim)
            kv_up = kv_rank * (self.num_attention_heads * (h_dim + h_dim))
            o_proj = (self.num_attention_heads * h_dim) * self.hidden_size
            attn_per_layer = q_down + q_up + kv_down + kv_up + o_proj
        else:
            q_params = self.hidden_size * (self.num_attention_heads * self.head_dim)
            k_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
            v_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
            o_params = (self.num_attention_heads * self.head_dim) * self.hidden_size
            attn_per_layer = q_params + k_params + v_params + o_params

        ffn_matrix_factor = 3 if self.mlp_type.lower() == "swiglu" else 2
        dense_ffn_per_layer = ffn_matrix_factor * self.hidden_size * self.intermediate_size

        dense_layers = min(self.first_k_dense_replace, self.num_layers)
        moe_layers = max(0, self.num_layers - dense_layers)
        moe_inter = self.moe_intermediate_size or self.intermediate_size

        active_routed_ffn = (
            moe_layers
            * self.num_experts_per_tok
            * (ffn_matrix_factor * self.hidden_size * moe_inter)
        )
        shared_ffn = (
            moe_layers * self.n_shared_experts * (ffn_matrix_factor * self.hidden_size * moe_inter)
        )
        initial_dense_ffn = dense_layers * dense_ffn_per_layer

        active_ffn_params = active_routed_ffn + shared_ffn + initial_dense_ffn
        norm_params = (2 * self.hidden_size) * self.num_layers + self.hidden_size
        total_attn_params = attn_per_layer * self.num_layers

        return embed_params + lm_head_params + total_attn_params + active_ffn_params + norm_params


# Standard Open-Weights Presets
MODEL_PRESETS: dict[str, ModelArch] = {
    # Qwen 2.5 series
    "qwen2.5-0.5b": ModelArch(
        name="qwen2.5-0.5b",
        family="qwen",
        num_layers=24,
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    "qwen2.5-1.5b": ModelArch(
        name="qwen2.5-1.5b",
        family="qwen",
        num_layers=28,
        hidden_size=1536,
        intermediate_size=8960,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    "qwen2.5-3b": ModelArch(
        name="qwen2.5-3b",
        family="qwen",
        num_layers=36,
        hidden_size=2048,
        intermediate_size=11008,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    "qwen2.5-7b": ModelArch(
        name="qwen2.5-7b",
        family="qwen",
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
    "qwen2.5-14b": ModelArch(
        name="qwen2.5-14b",
        family="qwen",
        num_layers=48,
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
    "qwen2.5-32b": ModelArch(
        name="qwen2.5-32b",
        family="qwen",
        num_layers=64,
        hidden_size=5120,
        intermediate_size=27648,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
    "qwen2.5-72b": ModelArch(
        name="qwen2.5-72b",
        family="qwen",
        num_layers=80,
        hidden_size=8192,
        intermediate_size=29568,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
    # Llama 3 series
    "llama3-8b": ModelArch(
        name="llama3-8b",
        family="llama",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=128256,
        tie_word_embeddings=False,
    ),
    "llama3-70b": ModelArch(
        name="llama3-70b",
        family="llama",
        num_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=128256,
        tie_word_embeddings=False,
    ),
    # Mistral 7B
    "mistral-7b": ModelArch(
        name="mistral-7b",
        family="mistral",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=32000,
        tie_word_embeddings=False,
    ),
    # Phi-3 Mini
    "phi-3-mini": ModelArch(
        name="phi-3-mini",
        family="phi",
        num_layers=32,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=96,
        vocab_size=32064,
        tie_word_embeddings=False,
    ),
    # Mixture of Experts (MoE) & Multi-Head Latent Attention (MLA)
    "deepseek-v3": ModelArch(
        name="deepseek-v3",
        family="deepseek",
        num_layers=61,
        hidden_size=7168,
        intermediate_size=18432,
        num_attention_heads=128,
        num_key_value_heads=128,
        head_dim=128,
        vocab_size=129280,
        tie_word_embeddings=False,
        is_moe=True,
        num_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
    ),
    "deepseek-r1": ModelArch(
        name="deepseek-r1",
        family="deepseek",
        num_layers=61,
        hidden_size=7168,
        intermediate_size=18432,
        num_attention_heads=128,
        num_key_value_heads=128,
        head_dim=128,
        vocab_size=129280,
        tie_word_embeddings=False,
        is_moe=True,
        num_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
    ),
    "qwen2.5-moe-14b": ModelArch(
        name="qwen2.5-moe-14b",
        family="qwen",
        num_layers=28,
        hidden_size=2048,
        intermediate_size=1408,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
        is_moe=True,
        num_experts=64,
        num_experts_per_tok=8,
        n_shared_experts=0,
    ),
    "mixtral-8x7b": ModelArch(
        name="mixtral-8x7b",
        family="mixtral",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=32000,
        tie_word_embeddings=False,
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
    ),
    # GPT-2 (for fast local CPU unit testing)
    "gpt2": ModelArch(
        name="gpt2",
        family="gpt2",
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        vocab_size=50257,
        tie_word_embeddings=True,
        mlp_type="standard",
    ),
}


def get_model_spec(model_name: str) -> ModelArch:
    """Retrieve model architectural specification by name or alias."""
    key = model_name.lower().strip()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]

    # Fuzzy prefix match (e.g. 'qwen-7b' -> 'qwen2.5-7b')
    for name, spec in MODEL_PRESETS.items():
        if key in name:
            return spec

    raise ValueError(
        f"Unknown model architecture '{model_name}'. Available presets: {list(MODEL_PRESETS.keys())}"
    )


def list_models() -> list[str]:
    """Return all available preset model names."""
    return list(MODEL_PRESETS.keys())
