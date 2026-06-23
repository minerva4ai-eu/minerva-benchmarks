"""
mfu_callback.py
~~~~~~~~~~~~~~~
A HuggingFace TrainerCallback that computes per-step MFU (Model FLOP Utilisation)
using the same arithmetic as Megatron-LM's num_floating_point_operations().

Supported architectures (auto-detected from config.json via AutoConfig):
  - LLaMA 3 / LLaMA 2 family            (LlamaConfig)
  - Mistral / Mixtral (MoE)              (MistralConfig / MixtralConfig)
  - Gemma 3 / Gemma 2 / Gemma 1         (Gemma3Config / Gemma2Config / GemmaConfig)
  - Qwen 2 / Qwen 2 MoE / Qwen 3 MoE   (Qwen2Config / Qwen2MoeConfig / Qwen3MoeConfig)
  - GPT-NeoX / Falcon / Phi              (GPTNeoXConfig / FalconConfig / PhiConfig)
  - Any model with standard HF config fields

Usage
-----
    from transformers import AutoConfig
    from mfu_callback import mfu_callback_from_hf_config

    cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    callback = mfu_callback_from_hf_config(cfg, gpu_peak_flops=989e12)
    trainer = SFTTrainer(..., callbacks=[callback])

The callback logs `mfu` (as a %) to the Trainer log dict every logging step.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# FLOP formula (ported from Megatron-LM num_floating_point_operations)
# ---------------------------------------------------------------------------


def num_floating_point_operations(
    num_layers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    *,
    num_query_groups: Optional[int] = None,
    kv_channels: Optional[int] = None,
    swiglu: bool = False,
    # MoE
    num_experts: Optional[int] = None,
    moe_ffn_hidden_size: Optional[int] = None,
    moe_router_topk: int = 1,
    moe_layer_freq: Union[int, List[int]] = 1,
    shared_expert_ffn_hidden_size: int = 0,
    # MTP (multi-token prediction)
    mtp_num_layers: int = 0,
    # Batch / sequence
    batch_size: int = 1,
    seq_length: int = 2048,
    # Packed-sequence overrides
    total_real_tokens: Optional[float] = None,
    seqlen_squared_sum: Optional[float] = None,
) -> float:
    """Total FLOPs for one global batch (fwd + bwd, ×3 Megatron convention)."""

    T = total_real_tokens if total_real_tokens is not None else batch_size * seq_length
    S2 = (
        seqlen_squared_sum
        if seqlen_squared_sum is not None
        else batch_size * seq_length**2
    )

    gqa_groups = (
        num_query_groups if num_query_groups is not None else num_attention_heads
    )

    # ── layer counts ──────────────────────────────────────────────────────
    if num_experts is None:
        num_dense_layers = num_layers
        num_moe_layers = 0
    else:
        if isinstance(moe_layer_freq, int):
            pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        else:
            pattern = list(moe_layer_freq)
            assert len(pattern) == num_layers
        num_moe_layers = sum(pattern)
        num_dense_layers = num_layers - num_moe_layers

    total_layers = num_layers + mtp_num_layers
    _moe_ffn = (
        moe_ffn_hidden_size if moe_ffn_hidden_size is not None else ffn_hidden_size
    )

    FWD_BWD = 3
    FMA = 2
    ffn_exp = 3 if swiglu else 2  # SwiGLU needs gate+up (×2 width) + down

    # ── MLP ──────────────────────────────────────────────────────────────
    mlp_flops = (
        FWD_BWD * FMA * hidden_size * (ffn_hidden_size * ffn_exp) * num_dense_layers * T
    )
    moe_routed = (
        FWD_BWD
        * FMA
        * hidden_size
        * (_moe_ffn * moe_router_topk * ffn_exp)
        * num_moe_layers
        * T
    )
    moe_shared = (
        FWD_BWD
        * FMA
        * hidden_size
        * (shared_expert_ffn_hidden_size * ffn_exp)
        * num_moe_layers
        * T
    )

    # ── Attention ─────────────────────────────────────────────────────────
    # kv_channels lets you override head_dim (e.g. Gemma uses head_dim ≠ hidden/heads)
    # p scales the QKV projection width; if kv_channels is None, p=1 (standard)
    p = (kv_channels * num_attention_heads / hidden_size) if kv_channels else 1.0
    g = gqa_groups

    attn_token_linear = (
        FWD_BWD
        * FMA
        * hidden_size
        * p
        * (hidden_size + hidden_size * (g / num_attention_heads))
        * total_layers
        * T
    )
    attn_core = (
        FWD_BWD
        * FMA
        * hidden_size
        * p  # /2 causal × 2 ops cancel
        * total_layers
        * S2
    )

    # ── MTP extra heads ───────────────────────────────────────────────────
    mtp_flops = (
        FWD_BWD
        * FMA
        * mtp_num_layers
        * (3 * hidden_size + 2 * hidden_size * hidden_size)
        * T
    )

    # ── Logit projection ──────────────────────────────────────────────────
    logit_flops = FWD_BWD * FMA * hidden_size * vocab_size * (mtp_num_layers + 1) * T

    return (
        mlp_flops
        + moe_routed
        + moe_shared
        + attn_token_linear
        + attn_core
        + mtp_flops
        + logit_flops
    )


# ---------------------------------------------------------------------------
# ModelFLOPConfig
# ---------------------------------------------------------------------------


@dataclass
class ModelFLOPConfig:
    """Architecture parameters needed to compute FLOPs."""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    vocab_size: int
    seq_length: int

    num_query_groups: Optional[int] = None  # None → MHA
    kv_channels: Optional[int] = None  # head_dim override
    swiglu: bool = False

    # MoE
    num_experts: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_router_topk: int = 1
    moe_layer_freq: Union[int, List[int]] = 1
    shared_expert_ffn_hidden_size: int = 0

    # MTP
    mtp_num_layers: int = 0


# ---------------------------------------------------------------------------
# MFUCallback
# ---------------------------------------------------------------------------


class MFUCallback(TrainerCallback):
    """
    Computes and logs MFU (%) after every logging step.

    Parameters
    ----------
    model_config : ModelFLOPConfig
    gpu_peak_flops : float
        Peak FLOP/s of ONE GPU (e.g. 989e12 for H100 BF16).
    log_key : str
        Key written into the Trainer log dict. Default "mfu".
    """

    @dataclass
    class State:
        tflops_this_gpu: list[float]
        mfu_this_gpu: list[float]

    def __init__(
        self,
        model_config: ModelFLOPConfig,
        gpu_peak_flops: float,
        log_key: str = "mfu",
    ):
        self.state = self.State([], [])
        self.cfg = model_config
        self.gpu_peak_flops = gpu_peak_flops
        self.log_key = log_key
        self._step_start_time: Optional[float] = None
        self._last_logged_step: int = 0

    def _tflops_per_batch(self, batch_size: int) -> float:
        c = self.cfg
        return (
            num_floating_point_operations(
                num_layers=c.num_layers,
                hidden_size=c.hidden_size,
                ffn_hidden_size=c.ffn_hidden_size,
                num_attention_heads=c.num_attention_heads,
                vocab_size=c.vocab_size,
                num_query_groups=c.num_query_groups,
                kv_channels=c.kv_channels,
                swiglu=c.swiglu,
                num_experts=c.num_experts,
                moe_ffn_hidden_size=c.moe_ffn_hidden_size,
                moe_router_topk=c.moe_router_topk,
                moe_layer_freq=c.moe_layer_freq,
                shared_expert_ffn_hidden_size=c.shared_expert_ffn_hidden_size,
                mtp_num_layers=c.mtp_num_layers,
                batch_size=batch_size,
                seq_length=c.seq_length,
            )
            / 1e12
        )

    def _num_gpus(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        elapsed = time.perf_counter() - self._step_start_time
        if elapsed <= 0:
            return

        steps = max(state.global_step - self._last_logged_step, 1)
        self._last_logged_step = state.global_step

        world_size = max(args.world_size, self._num_gpus())
        global_bs = (
            args.per_device_train_batch_size
            * world_size
            * args.gradient_accumulation_steps
        )
        total_flops = self._tflops_per_batch(global_bs) * steps
        achieved_flops = total_flops / elapsed / world_size
        mfu = achieved_flops / self.gpu_peak_flops * 100
        self.state.tflops_this_gpu.append(round(achieved_flops, 2))
        self.state.mfu_this_gpu.append(round(mfu, 2))
        return super().on_step_end(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._step_start_time is None:
            return

        logs[f"D{os.environ['RANK']}:TFLOPs/sec/GPU"] = (
            f"{self.state.tflops_this_gpu[-1]:.2f}"
        )
        logs[f"D{os.environ['RANK']}:mfu"] = f"{self.state.mfu_this_gpu[-1]:.2f}%"


# ---------------------------------------------------------------------------
# Factory — reads AutoConfig and maps every field correctly
# ---------------------------------------------------------------------------


def _unwrap_text_config(cfg):
    """
    Gemma 3 (and other VLMs) store text arch inside cfg.text_config.
    All other models return cfg unchanged.
    """
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return text_cfg
    return cfg


def mfu_callback_from_hf_config(
    model_or_config,
    tokenizer_or_config,
    gpu_peak_flops: float,
    seq_length: Optional[int] = None,
    log_key: str = "mfu",
    **kwargs,
) -> MFUCallback:
    """
    Build an MFUCallback from a HuggingFace PretrainedConfig (or a model).

    Handles: LLaMA 3/2, Mistral, Mixtral (MoE), Gemma 3/2/1 (incl. VLM wrapper),
             Qwen2, Qwen2-MoE, Qwen3-MoE, GPT-NeoX, Phi, Falcon, and any model
             that follows the standard HF attribute naming.

    Extra keyword arguments override auto-detected values and are forwarded to
    ModelFLOPConfig (e.g. swiglu=True, seq_length=8192).
    """
    raw_cfg = getattr(model_or_config, "config", model_or_config)
    cfg = _unwrap_text_config(raw_cfg)

    raw_tkn_cfg = getattr(tokenizer_or_config, "config", tokenizer_or_config)
    tkn_cfg = _unwrap_text_config(raw_tkn_cfg)

    def _get(*names, default=None):
        for name in names:
            v = getattr(cfg, name, None)
            if v is not None:
                return v
        return default

    def _get_tkn(*names, default=None):
        for name in names:
            v = getattr(tkn_cfg, name, None)
            if v is not None:
                return v
        return default

    # ── Core architecture fields ──────────────────────────────────────────
    # Field names verified against HF config docs for each model family:
    #
    #  LlamaConfig      : num_hidden_layers, hidden_size, intermediate_size,
    #                     num_attention_heads, num_key_value_heads, vocab_size
    #  MistralConfig    : same as Llama
    #  MixtralConfig    : same as Llama + num_local_experts, num_experts_per_tok
    #  GemmaConfig      : num_hidden_layers, hidden_size, intermediate_size,
    #                     num_attention_heads, num_key_value_heads, head_dim, vocab_size
    #  Gemma2Config     : same as Gemma
    #  Gemma3Config     : (text_config unwrapped above) same as Gemma
    #  Qwen2Config      : num_hidden_layers, hidden_size, intermediate_size,
    #                     num_attention_heads, num_key_value_heads, vocab_size
    #  Qwen2MoeConfig   : + num_experts, num_experts_per_tok, moe_intermediate_size,
    #                       shared_expert_intermediate_size
    #  GPTNeoXConfig    : num_hidden_layers, hidden_size, intermediate_size,
    #                     num_attention_heads  (no GQA)
    #  FalconConfig     : num_hidden_layers, hidden_size, num_attention_heads,
    #                     num_kv_heads (GQA variant)

    num_layers = _get("num_hidden_layers", "n_layer", "num_layers")
    if num_layers is None:
        raise ValueError("Cannot detect num_layers from config")

    hidden_size = _get("hidden_size", "n_embd", "d_model")
    if hidden_size is None:
        raise ValueError("Cannot detect hidden_size from config")

    num_heads = _get("num_attention_heads", "n_head", "num_heads")
    if num_heads is None:
        raise ValueError("Cannot detect num_attention_heads from config")

    vocab_size = _get_tkn("vocab_size")
    if vocab_size is None:
        raise ValueError("Cannot detect vocab_size from config")

    # intermediate_size (ffn_hidden_size in ModelFLOPConfig)
    # GPT-NeoX uses "intermediate_size"; older GPT-2-style uses 4*hidden
    ffn_hidden_size = _get(
        "intermediate_size",  # Llama, Mistral, Mixtral, Gemma, Qwen, NeoX
        "ffn_dim",  # some older models
        "n_inner",  # GPT-2
        default=4 * hidden_size,
    )

    # GQA: num_key_value_heads
    # Falcon uses num_kv_heads; most others use num_key_value_heads
    num_kv_heads = _get(
        "num_key_value_heads",  # Llama, Mistral, Mixtral, Gemma, Qwen
        "num_kv_heads",  # Falcon
        "num_query_groups",  # some NeMo-style configs
    )
    # None means MHA (num_kv_heads == num_heads) — leave as None, handled in FLOP fn

    # head_dim override (Gemma uses a fixed head_dim=256 independent of hidden/heads)
    # Only set kv_channels when the config explicitly specifies head_dim AND it differs
    # from the default (hidden_size // num_heads).
    explicit_head_dim = _get("head_dim")
    default_head_dim = hidden_size // num_heads
    kv_channels = (
        explicit_head_dim
        if (explicit_head_dim and explicit_head_dim != default_head_dim)
        else None
    )

    # Sequence length
    _seq = seq_length or _get("max_position_embeddings", default=2048)

    # ── Activation / SwiGLU detection ────────────────────────────────────
    # LLaMA/Mistral/Mixtral/Qwen all use hidden_act="silu" with SwiGLU gate
    # (gate_proj + up_proj → down_proj).  Gemma uses "gelu" without a gate.
    # We check hidden_act AND the presence of a gate projection in the config.
    hidden_act = str(
        _get("hidden_act", "hidden_activation", "activation_function") or ""
    )
    # Models with SwiGLU (gated MLP): silu or swish + NOT gelu
    has_swiglu_act = any(k in hidden_act.lower() for k in ("silu", "swish", "swiglu"))
    # Some configs expose mlp_bias or gate_proj explicitly; use act as proxy
    _swiglu = has_swiglu_act

    # ── MoE fields ────────────────────────────────────────────────────────
    # Mixtral  : num_local_experts, num_experts_per_tok, intermediate_size (per expert)
    # Qwen2Moe : num_experts, num_experts_per_tok, moe_intermediate_size,
    #            shared_expert_intermediate_size, shared_expert_num (=1 usually)

    num_experts = _get("num_local_experts", "num_experts")  # total expert count
    moe_topk = _get("num_experts_per_tok", "top_k")  # routed experts per token
    # Mixtral reuses intermediate_size for each expert's FFN width;
    # Qwen2Moe has a separate moe_intermediate_size
    moe_ffn = _get("moe_intermediate_size")  # Qwen2Moe only; else None
    # If None, moe_ffn falls back to ffn_hidden_size inside num_floating_point_operations
    shared_expert_ffn = _get("shared_expert_intermediate_size", default=0) or 0

    # ── Assemble (kwargs can override anything) ───────────────────────────
    defaults = dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
        seq_length=_seq,
        num_query_groups=num_kv_heads,
        kv_channels=kv_channels,
        swiglu=_swiglu,
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=moe_topk or 1,
        shared_expert_ffn_hidden_size=shared_expert_ffn,
    )
    defaults.update(kwargs)  # user overrides win

    flop_cfg = ModelFLOPConfig(**defaults)
    return MFUCallback(
        model_config=flop_cfg, gpu_peak_flops=gpu_peak_flops, log_key=log_key
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Simulate AutoConfig objects with the exact field names HF uses ────
    class FakeConfig:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    model_configs = [
        (
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/gemma-3-1b-it/config.json",
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/gemma-3-1b-it",  # tokenizer
        ),
        (
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/gemma-3-12b-it/config.json",
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/gemma-3-12b-it",  # tokenizer
        ),
        (
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Mistral-7B-Instruct-v0.3/config.json",
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Mistral-7B-Instruct-v0.3",  # tokenizer
        ),
        (
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.1-8B-Instruct/config.json",
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.1-8B-Instruct",  # tokenizer
        ),
        (
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.3-70B-Instruct/config.json",
            "/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.3-70B-Instruct",  # tokenizer
        ),
    ]

    """configs = {
        "llama3-8b": FakeConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            hidden_act="silu",
            max_position_embeddings=8192,
        ),
        "llama3-70b": FakeConfig(
            num_hidden_layers=80,
            hidden_size=8192,
            intermediate_size=28672,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=128256,
            hidden_act="silu",
            max_position_embeddings=8192,
        ),
        "mistral-7b": FakeConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            hidden_act="silu",
            max_position_embeddings=32768,
        ),
        "mixtral-8x7b": FakeConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            hidden_act="silu",
            max_position_embeddings=32768,
            num_local_experts=8,
            num_experts_per_tok=2,
        ),
        "gemma3-12b": FakeConfig(
            num_hidden_layers=48,
            hidden_size=3840,
            intermediate_size=15360,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocab_size=262208,
            hidden_act="gelu_pytorch_tanh",
            head_dim=256,
            max_position_embeddings=131072,
        ),
        "gemma3-1b": FakeConfig(
            num_hidden_layers=26,
            hidden_size=1152,
            intermediate_size=6912,
            num_attention_heads=4,
            num_key_value_heads=1,
            vocab_size=262208,
            hidden_act="gelu_pytorch_tanh",
            head_dim=256,
            max_position_embeddings=32768,
        ),
    }"""

    H100 = 989e12
    print(
        f"{'Model':<18} {'layers':>6} {'hidden':>6} {'ffn':>6} {'heads':>5} "
        f"{'kv':>4} {'experts':>7} {'topk':>4} {'swiglu':>6} "
        f"{'FLOPs/batch (T)':>16}  {'MFU% (8xH100, 0.5s)':>20}"
    )
    print("-" * 110)

    for path, tkn_path in model_configs:
        hfcfg = AutoConfig.from_pretrained(path)

        tokenizer = AutoTokenizer.from_pretrained(tkn_path, trust_remote_code=True)
        cb = mfu_callback_from_hf_config(
            hfcfg, tokenizer, gpu_peak_flops=H100, seq_length=4096
        )
        c = cb.cfg
        fl = cb._flops_per_batch(batch_size=4)
        mfu = fl / 0.5 / (H100 * 8) * 100

        name = path.split("/")[-2]
        print(
            f"{name:<18} {c.num_layers:>6} {c.hidden_size:>6} {c.ffn_hidden_size:>6} "
            f"{c.num_attention_heads:>5} {str(c.num_query_groups):>4} "
            f"{str(c.num_experts):>7} {c.moe_router_topk:>4} {str(c.swiglu):>6} "
            f"{fl / 1e12:>16.1f}  {mfu:>20.1f}%"
        )
