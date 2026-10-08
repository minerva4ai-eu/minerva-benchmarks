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
    callback = mfu_callback_from_hf_config(cfg, tokenizer, gpu_peak_flops=989, trainer_callback="pytorch")
    trainer = SFTTrainer(..., callbacks=[callback])

The callback logs `mfu` (as a %) to the Trainer log dict every logging step.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

# scripts.shared.data import load_and_prepare_raw_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")
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
    num_query_groups: int | None = None,
    kv_channels: int | None = None,
    swiglu: bool = False,
    # MoE
    num_experts: int | None = None,
    moe_ffn_hidden_size: int | None = None,
    moe_router_topk: int = 1,
    moe_layer_freq: int | list[int] = 1,
    shared_expert_ffn_hidden_size: int = 0,
    # MTP (multi-token prediction)
    mtp_num_layers: int = 0,
    # Sliding-window attention: how many layers use a local window
    num_local_attn_layers: int = 0,
    sliding_window: int | None = None,
    # 3 = full fine-tuning/pre-training; ~2 when weights are frozen (e.g. LoRA)
    fwd_bwd_factor: float = 3,
    # Batch / sequence
    batch_size: int = 1,
    seq_length: int = 2048,
    # Packed-sequence overrides
    total_real_tokens: float | None = None,
    seqlen_squared_sum: float | None = None,
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

    FWD_BWD = fwd_bwd_factor
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
        # Q + O projections (h each) and K + V projections (h*g/n each)
        * (2 * hidden_size + 2 * hidden_size * (g / num_attention_heads))
        * total_layers
        * T
    )
    # Sliding-window layers only attend to ~L*W - W^2/2 pairs instead of L^2/2
    n_local = min(max(num_local_attn_layers, 0), total_layers)
    local_ratio = 1.0
    if n_local and sliding_window and sliding_window < seq_length:
        w = sliding_window
        local_ratio = (seq_length * w - w * w / 2) / (seq_length * seq_length / 2)
    effective_attn_layers = (total_layers - n_local) + n_local * local_ratio
    attn_core = (
        FWD_BWD
        * FMA
        * hidden_size
        * p  # /2 causal × 2 ops cancel
        * effective_attn_layers
        * S2
    )

    # ── MoE router ────────────────────────────────────────────────────────
    router_flops = FWD_BWD * FMA * hidden_size * (num_experts or 0) * num_moe_layers * T

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
        + router_flops
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

    num_query_groups: int | None = None  # None → MHA
    kv_channels: int | None = None  # head_dim override
    swiglu: bool = False

    # MoE
    num_experts: int | None = None
    moe_ffn_hidden_size: int | None = None
    moe_router_topk: int = 1
    moe_layer_freq: int | list[int] = 1
    shared_expert_ffn_hidden_size: int = 0

    # MTP
    mtp_num_layers: int = 0

    # Sliding-window attention
    num_local_attn_layers: int = 0
    sliding_window: int | None = None

    # 3 for full training; ~2 if weights are frozen (LoRA / PEFT)
    fwd_bwd_factor: float = 3

    def flops_per_batch(self, batch_size: int) -> float:
        """Total FLOPs (not TFLOPs) for one global batch of `batch_size` sequences."""
        return num_floating_point_operations(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            vocab_size=self.vocab_size,
            num_query_groups=self.num_query_groups,
            kv_channels=self.kv_channels,
            swiglu=self.swiglu,
            num_experts=self.num_experts,
            moe_ffn_hidden_size=self.moe_ffn_hidden_size,
            moe_router_topk=self.moe_router_topk,
            moe_layer_freq=self.moe_layer_freq,
            shared_expert_ffn_hidden_size=self.shared_expert_ffn_hidden_size,
            mtp_num_layers=self.mtp_num_layers,
            num_local_attn_layers=self.num_local_attn_layers,
            sliding_window=self.sliding_window,
            fwd_bwd_factor=self.fwd_bwd_factor,
            batch_size=batch_size,
            seq_length=self.seq_length,
        )


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
        Peak TFLOP/s of ONE GPU (e.g. 989 for H100 BF16). Not FLOP/s.
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
        self._step_start_time: float | None = None
        self._last_logged_step: int = 0

    def _tflops_per_batch(self, batch_size: int) -> float:
        return self.cfg.flops_per_batch(batch_size) / 1e12

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

        # elapsed covers exactly one optimizer step (since on_step_begin)
        self._last_logged_step = state.global_step

        world_size = max(args.world_size, 1)
        global_bs = (
            args.per_device_train_batch_size
            * world_size
            * args.gradient_accumulation_steps
        )
        total_flops = self._tflops_per_batch(global_bs)
        achieved_flops = total_flops / elapsed / world_size
        mfu = achieved_flops / self.gpu_peak_flops * 100
        self.state.tflops_this_gpu.append(round(achieved_flops, 2))
        self.state.mfu_this_gpu.append(round(mfu, 2))
        return super().on_step_end(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._step_start_time is None:
            return

        logs[f"D{os.environ.get('RANK')}:TFLOPs/sec/GPU"] = (
            f"{self.state.tflops_this_gpu[-1]:.2f}"
        )
        logs[f"D{os.environ.get('RANK')}:mfu"] = f"{self.state.mfu_this_gpu[-1]:.2f}%"


class FlopCounterCallback(pl.Callback):
    def __init__(
        self,
        model_config: ModelFLOPConfig,
        gpu_peak_flops: float,
        global_batch_size: int,
        enabled: bool = True,
        warmup_steps: int = 0,
    ):
        self.enabled = enabled
        self.cfg = model_config
        self.gpu_peak_flops = gpu_peak_flops
        self.global_batch_size = global_batch_size
        self.warmup_steps = max(warmup_steps, 0)
        self.flops_list: list[float] = []
        self.mfu_list: list[float] = []
        self._step_start_time: float | None = None
        self._step_start_event = None
        self._step_time_events = []
        self._step_number = 0
        self._world_size = 1
        self._average_metrics: tuple[float, float] | None = None
        self.logger = logging.getLogger("MINERVA_BENCH.scripts.shared.flops")

    def _flops_per_batch(self) -> float:
        return self.cfg.flops_per_batch(self.global_batch_size)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.enabled:
            self._step_start_time = time.perf_counter()
            self._step_start_event = torch.cuda.Event(enable_timing=True)
            self._step_start_event.record()
            self._step_number += 1

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (
            not self.enabled
            or self._step_start_time is None
            or self._step_start_event is None
        ):
            return

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._step_time_events.append((self._step_start_event, end_event))
        self._step_start_event = None
        self._world_size = max(trainer.world_size, 1)

        elapsed = time.perf_counter() - self._step_start_time
        if elapsed <= 0:
            return

        flops_per_gpu = self._flops_per_batch() / self._world_size
        tflops_per_second = flops_per_gpu / elapsed / 1e12
        mfu = tflops_per_second / self.gpu_peak_flops * 100
        self.flops_list.append(tflops_per_second)
        self.mfu_list.append(mfu)

        metrics = {
            "FLOPs/GPU/step": flops_per_gpu,
            "TFLOPs/sec/GPU": tflops_per_second,
            "MFU (%)": mfu,
        }
        self.logger.info(
            "step=%d FLOPs/GPU/step=%.3e TFLOPs/sec/GPU=%.2f MFU=%.2f%%",
            self._step_number,
            flops_per_gpu,
            tflops_per_second,
            mfu,
        )
        if trainer.logger:
            trainer.logger.log_metrics(metrics, step=self._step_number)

    def on_train_end(self, trainer, pl_module):
        averages = self._get_average_metrics()
        if averages is not None:
            self.logger.info(
                "Average FLOPs/sec/GPU excluding %d warmup steps: %.1f TFLOPs/sec, MFU: %.2f%%",
                self.warmup_steps,
                averages[0],
                averages[1],
            )

    def _get_average_metrics(self) -> tuple[float, float] | None:
        measured_events = self._step_time_events[self.warmup_steps :]
        if not self.enabled or not measured_events:
            return None
        if self._average_metrics is None:
            torch.cuda.synchronize()
            elapsed_seconds = sum(
                start.elapsed_time(end) / 1000 for start, end in measured_events
            )
            if elapsed_seconds <= 0:
                return None
            flops_per_gpu_per_step = self._flops_per_batch() / self._world_size
            average_tflops = (
                flops_per_gpu_per_step * len(measured_events) / elapsed_seconds / 1e12
            )
            self._average_metrics = (
                average_tflops,
                average_tflops / self.gpu_peak_flops * 100,
            )
        return self._average_metrics

    def get_avg_flops(self) -> float | None:
        """Return average TFLOPs/s per GPU, excluding configured warmup steps."""
        averages = self._get_average_metrics()
        return averages[0] if averages is not None else None

    def get_avg_mfu(self) -> float | None:
        """Return average MFU, excluding configured warmup steps."""
        averages = self._get_average_metrics()
        return averages[1] if averages is not None else None


class FLOPsMFUCalculator:
    """
    Computes and logs MFU (%) after every logging step.

    Parameters
    ----------
    model_config : ModelFLOPConfig
    gpu_peak_flops : float
        Peak TFLOP/s of ONE GPU (e.g. 989 for H100 BF16). Not FLOP/s.
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
        self._step_start_time: float | None = None
        self._last_logged_step: int = 0

    def _tflops_per_batch(self, batch_size: int) -> float:
        return self.cfg.flops_per_batch(batch_size) / 1e12

    def _num_gpus(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def on_step_begin(
        self,
    ):
        self._step_start_time = time.perf_counter()

    def on_step_end(
        self,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        world_size: int,
        global_step: int,
    ):
        elapsed = time.perf_counter() - self._step_start_time
        if elapsed <= 0:
            return

        # elapsed covers exactly one optimizer step (since on_step_begin)
        self._last_logged_step = global_step

        global_bs = micro_batch_size * world_size * gradient_accumulation_steps
        total_flops = self._tflops_per_batch(global_bs)
        achieved_flops = total_flops / elapsed / world_size
        mfu = achieved_flops / self.gpu_peak_flops * 100
        self.state.tflops_this_gpu.append(round(achieved_flops, 2))
        self.state.mfu_this_gpu.append(round(mfu, 2))
        return

    def on_log(self, logs: dict[str, str]) -> dict[str, str]:
        if logs is None or self._step_start_time is None:
            return

        logs[f"D{os.environ['RANK']}:TFLOPs/sec/GPU"] = (
            f"{self.state.tflops_this_gpu[-1]:.2f}"
        )
        logs[f"D{os.environ['RANK']}:mfu"] = f"{self.state.mfu_this_gpu[-1]:.2f}%"

        return logs


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
    trainer_callback: Literal["lightning", "pytorch", "custom"],
    seq_length: int | None = None,
    log_key: str = "mfu",
    global_batch_size: int | None = None,
    warmup_steps: int = 0,
    **kwargs,
) -> MFUCallback | FLOPsMFUCalculator | FlopCounterCallback:
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

    # Prefer the model config: the LM head / embedding size is what is multiplied,
    # and it is usually larger than the tokenizer's vocab_size (padding, added tokens).
    vocab_size = _get("vocab_size") or _get_tkn("vocab_size")
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

    # ── MLP gating (SwiGLU / GeGLU = 3 matrices; plain MLP = 2) ──────────
    # Nearly all modern decoder families (Llama, Mistral, Qwen, Gemma, DeepSeek...)
    # use a gated MLP regardless of activation. Only a few legacy families don't.
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    _NON_GATED = {
        "gpt2", "gpt_neox", "gptj", "falcon", "phi", "opt", "bloom", "mpt",
        "gpt_bigcode", "starcoder2", "persimmon", "stablelm_epoch",
    }  # fmt: skip
    _swiglu = model_type not in _NON_GATED

    # ── MoE fields ────────────────────────────────────────────────────────
    # Mixtral  : num_local_experts, num_experts_per_tok (expert width = intermediate_size)
    # Qwen2/3Moe: num_experts, num_experts_per_tok, moe_intermediate_size,
    #            decoder_sparse_step, mlp_only_layers, [shared_expert_intermediate_size]
    # DeepSeek : n_routed_experts, n_shared_experts, moe_intermediate_size,
    #            first_k_dense_replace, moe_layer_freq
    # Llama4   : num_local_experts, intermediate_size (expert + shared expert),
    #            intermediate_size_mlp (dense layers), interleave_moe_layer_step
    num_experts = _get("num_local_experts", "num_experts", "n_routed_experts")
    moe_topk = _get("num_experts_per_tok", "top_k")  # routed experts per token
    moe_ffn = _get("moe_intermediate_size")  # else falls back to ffn_hidden_size
    shared_expert_ffn = _get("shared_expert_intermediate_size", default=0) or 0
    n_shared = _get("n_shared_experts", default=0) or 0
    if n_shared and moe_ffn:
        shared_expert_ffn = n_shared * moe_ffn

    moe_layer_pattern: int | list[int] = 1
    if model_type.startswith("llama4") and num_experts:
        dense_ffn = _get("intermediate_size_mlp")
        moe_ffn = ffn_hidden_size
        shared_expert_ffn = ffn_hidden_size  # one shared expert per MoE layer
        if dense_ffn:
            ffn_hidden_size = dense_ffn
        step = _get("interleave_moe_layer_step", default=1) or 1
        moe_layer_pattern = [1 if (i + 1) % step == 0 else 0 for i in range(num_layers)]
    elif num_experts and hasattr(cfg, "first_k_dense_replace"):  # DeepSeek family
        first_dense = _get("first_k_dense_replace", default=0) or 0
        freq = _get("moe_layer_freq", default=1) or 1
        if isinstance(freq, int):
            moe_layer_pattern = [
                1 if (i >= first_dense and i % freq == 0) else 0
                for i in range(num_layers)
            ]
        else:
            moe_layer_pattern = [int(x) for x in freq]
        logger.warning(
            "DeepSeek-style MLA attention is approximated as standard GQA attention."
        )
    elif num_experts and hasattr(cfg, "decoder_sparse_step"):  # Qwen2/3-MoE
        step = _get("decoder_sparse_step", default=1) or 1
        dense_only = set(_get("mlp_only_layers", default=[]) or [])
        moe_layer_pattern = [
            1 if (i not in dense_only and (i + 1) % step == 0) else 0
            for i in range(num_layers)
        ]

    # ── Sliding-window attention ─────────────────────────────────────────
    sliding_window = _get("sliding_window")
    layer_types = _get("layer_types")
    n_local = 0
    if sliding_window:
        if layer_types:
            n_local = sum("sliding" in str(t) for t in layer_types)
        elif model_type.startswith("gemma3"):
            pattern = _get("sliding_window_pattern", default=6) or 6
            n_local = sum(1 for i in range(num_layers) if (i + 1) % pattern != 0)
        elif model_type == "gemma2":
            n_local = (num_layers + 1) // 2
        elif model_type in ("mistral", "mixtral"):
            n_local = num_layers

    # Sequence length: max_position_embeddings can be 128k+ and makes the
    # quadratic attention term meaningless, so callers should pass seq_length.
    if seq_length is None:
        logger.warning(
            "seq_length not provided; falling back to max_position_embeddings."
        )

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
        moe_layer_freq=moe_layer_pattern,
        shared_expert_ffn_hidden_size=shared_expert_ffn,
        num_local_attn_layers=n_local,
        sliding_window=sliding_window if n_local else None,
    )
    defaults.update(kwargs)  # user overrides win

    flop_cfg = ModelFLOPConfig(**defaults)

    if trainer_callback == "lightning":
        if global_batch_size is None:
            raise ValueError("global_batch_size is required for the Lightning callback")
        return FlopCounterCallback(
            model_config=flop_cfg,
            gpu_peak_flops=gpu_peak_flops,
            global_batch_size=global_batch_size,
            warmup_steps=warmup_steps,
        )
    if trainer_callback == "pytorch":
        return MFUCallback(
            model_config=flop_cfg, gpu_peak_flops=gpu_peak_flops, log_key=log_key
        )
    return FLOPsMFUCalculator(
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
            "<path to model HF>/config.json",
            "<path to model HF>",  # tokenizer
        ),
    ]

    H100 = 989  # peak TFLOP/s
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
        fl = cb._tflops_per_batch(batch_size=4)
        mfu = fl / 0.5 / (H100 * 8) * 100

        name = path.split("/")[-2]
        print(
            f"{name:<18} {c.num_layers:>6} {c.hidden_size:>6} {c.ffn_hidden_size:>6} "
            f"{c.num_attention_heads:>5} {c.num_query_groups!s:>4} "
            f"{c.num_experts!s:>7} {c.moe_router_topk:>4} {c.swiglu!s:>6} "
            f"{fl:>16.1f}  {mfu:>20.1f}%"
        )
