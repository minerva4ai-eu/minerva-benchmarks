#!/usr/bin/python3
"""SFT training script using a Megatron-Core native model + nl.MegatronStrategy (DDP+TP+PP+CP).

Why NeMo (PyTorch Lightning + MegatronStrategy) rather than Megatron-Bridge:
    Both back ends run on Megatron-Core; they differ in who owns the training loop.
    In Megatron-Bridge examples, a plain config object drives the loop.
    NeMo instead runs the loop through the PyTorch Lightning Trainer, whose callback
    hooks (on_train_batch_start/end, on_train_start/end) host our instrumentation:
    the FLOP counter, the Torch profiler and the throughput MegatronBenchmarkCallback are
    plain pl.Callback objects plugged into those hooks.

IMPORTANT: MegatronStrategy runs the full forward-backward over all micro-batches
    + the optimizer step inside a single training_step. So batch hooks fire once per GLOBAL
    step (reason in global steps, incl. warmup counts), and the fine-grained hooks
    (on_before/after_backward) never fire.

To Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py

Differences vs the AutoModel/FSDP2 script:
    * Model: HFAutoModelForCausalLM -> a NeMo model (llm.LlamaModel, llm.Qwen2Model, ...).
        The architecture must match your HF checkpoint. Local HF weights are loaded via nl.AutoResume
        (the model's "hf" importer converts them on the fly), not by the model constructor.
    * Strategy: nl.FSDP2Strategy -> nl.MegatronStrategy (DDP + TP/PP/CP).
    * Optimizer: pytorch_adam_with_cosine_annealing -> nl.MegatronOptimizerModule + scheduler.
        Gradient clipping is set on OptimizerConfig.clip_grad.
    * Precision: Trainer(precision=...) -> plugins=nl.MegatronMixedPrecision(...).
    * Data: HFDatasetDataModule + custom collate -> llm.FineTuningDataModule (JSONL input/output).
        A one-time HF->JSONL prep step is included.

Build the dataset before training:
    python nemo_megatron.py \
        --prepare \
        --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
        --dataset-root ./sft_megatron_data \
        --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
        --nemo-ckpt-path ./nemo_ckpt

Source: https://docs.nvidia.com/nemo-framework/user-guide/25.11/nemo-2.0/index.html
"""

import os

# Filter NCCL debug output to rank 0 only — we are using torchrun to launch distributed training
if int(os.environ.get("RANK", 0)) != 0:
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.pop("NCCL_DEBUG_FILE", None)

import json
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import ceil
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from datasets import Dataset
from lightning.pytorch.utilities.types import STEP_OUTPUT
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.flop_counter import FlopCounterMode
from utils import MegatronBenchmarkCallback

# Add the (Model, Config) pair that matches your HF checkpoint here.
MODEL_ARCHS = {
    "Qwen2.5-7B-Instruct": (llm.Qwen2Model, llm.Qwen25Config7B),
    "Qwen2.5-72B-Instruct": (llm.Qwen2Model, llm.Qwen25Config72B),
}


class FlopCounterCallback(pl.Callback):
    def __init__(self, enabled):
        self.enabled = enabled
        self.flops_list = []
        self.ctx = None

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.enabled:
            self.ctx = FlopCounterMode(display=False)
            self.ctx.__enter__()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.enabled and self.ctx:
            import torch

            self.ctx.__exit__(None, None, None)
            self.flops_list.append(self.ctx.get_total_flops())
            self.ctx = None
            torch.cuda.empty_cache()  # free unused vRAM to reduce risks of CUDA OOM

    def on_train_end(self, trainer, pl_module):
        if self.enabled and self.flops_list:
            import numpy as np

            print(f"Median FLOPs/step: {np.median(self.flops_list) / 1e12:.1f} TFLOPs")


class TorchProfilerCallback(pl.Callback):
    """PyTorch Profiler as a Lightning Callback."""

    def __init__(self, rank: int):
        if rank == 0:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=16, warmup=1, active=8, repeat=1),
                on_trace_ready=tensorboard_trace_handler("./profile/"),
                profile_memory=True,
                record_shapes=True,
            )
        else:
            self.profiler = profile(activities=[])

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.profiler.start()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.profiler.step()

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.profiler.stop()


def parse_args() -> Namespace:
    """Process command-line arguments."""
    parser = ArgumentParser()

    # Mode
    parser.add_argument(
        "--prepare",
        action=BooleanOptionalAction,
        default=False,
        help="One-time prep (single process): build JSONL from the HF dataset. No training.",
    )

    # Training related arguments
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Number of examples seen for one model update.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length of each sample per GPU.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")

    # Benchmarking / debugging arguments
    parser.add_argument(
        "--test",
        action=BooleanOptionalAction,
        default=False,
        help="Run in test mode for a limited number of steps.",
    )
    parser.add_argument(
        "--test-nsteps",
        type=int,
        default=100,
        help="Number of steps to run in test mode.",
    )
    parser.add_argument(
        "--enable-flop-counter",
        action=BooleanOptionalAction,
        default=False,
        help="Compute FLOPs per step.",
    )
    parser.add_argument(
        "--pytorch-profiler",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to use pytorch profiler.",
    )

    # DataLoader related arguments
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="HuggingFace dataset path. Used only in --prepare.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory with training.jsonl / validation.jsonl.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers spawned by the dataloader.",
    )

    # Optimizer related arguments
    parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps for linear LR warmup (0.1 = 10%).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for Adam."
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="Weight decay for Adam."
    )

    # Model related arguments
    parser.add_argument(
        "--model-path", type=Path, help="Local HuggingFace model directory."
    )
    parser.add_argument(
        "--nemo-ckpt-path",
        type=Path,
        default=Path("./nemo_ckpt"),
        help="Where the converted NeMo checkpoint is written.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=BooleanOptionalAction,
        default=False,
        help="Enable full activation recomputation.",
    )
    parser.add_argument(
        "--fp8",
        action=BooleanOptionalAction,
        default=False,
        help="Enable FP8 training (via MegatronMixedPrecision).",
    )

    # Distributed training arguments
    parser.add_argument(
        "--devices-per-node", type=int, default=1, help="Number of GPUs per node."
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument(
        "--pp-size", type=int, default=1, help="Pipeline parallel size."
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size.")
    parser.add_argument(
        "--sequence-parallel",
        action=BooleanOptionalAction,
        default=False,
        help="Enable sequence parallelism (requires TP>1).",
    )

    # Logging / Checkpointing arguments
    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="Logging frequency."
    )
    parser.add_argument("--wandb-project", help="Wandb project name.")

    return parser.parse_args()


def prepare(args: Namespace) -> None:
    """One-time, single-process prep: build training.jsonl / validation.jsonl from the HF dataset."""
    import sys

    sys.stdout.flush()

    print(f"[prepare] model_path={args.model_path}", flush=True)
    print(f"[prepare] model_path.exists()={args.model_path.exists()}", flush=True)
    if args.model_path.exists():
        print(
            f"[prepare] model files: {list(args.model_path.iterdir())[:10]}", flush=True
        )

    # Convert HF checkpoint into NeMo checkpoint
    # https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.2/nemo/collections/llm/api.py#L577
    from nemo.collections.llm import import_ckpt

    print(f"[prepare] looking up MODEL_ARCHS for '{args.model_path.name}'", flush=True)
    model_cls, config_cls = MODEL_ARCHS[args.model_path.name]
    print(f"[prepare] creating tokenizer for {args.model_path}", flush=True)
    tokenizer = AutoTokenizer(args.model_path, use_fast=True)
    print("[prepare] tokenizer created, calling import_ckpt", flush=True)
    print(f"[prepare]   model_cls={model_cls}", flush=True)
    print(f"[prepare]   config_cls={config_cls}", flush=True)
    print(f"[prepare]   source=hf://{args.model_path}", flush=True)
    print(f"[prepare]   output_path={args.nemo_ckpt_path}", flush=True)
    import_ckpt(
        model=model_cls(config_cls(seq_length=args.seq_length), tokenizer=tokenizer),
        source=f"hf://{args.model_path}",  # local dir -> "hf:///abs/path", géré par l'importeur hf
        output_path=args.nemo_ckpt_path,
        overwrite=True,
    )
    print(f"[prepare] Converted HF -> NeMo at {args.nemo_ckpt_path}", flush=True)

    # Prepare dataset
    from datasets import load_dataset

    print(f"[prepare] loading dataset from {args.dataset_path}", flush=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(str(args.dataset_path))
    print(f"[prepare] dataset loaded: {list(dataset.keys())}", flush=True)

    def dump(split_name: str, hf_split: Dataset) -> None:
        out_file = args.dataset_root / f"{split_name}.jsonl"

        with open(out_file, "w") as f:
            f.writelines(
                json.dumps({"messages": ex["messages"]}) + "\n" for ex in hf_split
            )
        print(f"Wrote {out_file} ({len(hf_split)} examples)", flush=True)

    # Create training.jsonl
    dump("training", dataset["train"])
    # Reuse train as validation for warmup/sanity if no val split, mirroring the original script.
    dump(
        "validation",
        dataset["validation"] if "validation" in dataset else dataset["train"],
    )


def main() -> None:
    """Run SFT with a Megatron-Core model using Megatron strategy."""
    # 1. Get command-line arguments
    args = parse_args()

    if args.prepare:
        prepare(args)
        return

    # 2. Distributed Training Setup
    world = args.devices_per_node * args.num_nodes
    rank = int(os.environ.get("RANK", 0))  # Set by torchrun

    assert args.dp_size * args.pp_size * args.tp_size * args.cp_size == world, (
        f"4D mismatch: DP*PP*TP*CP={args.dp_size * args.pp_size * args.tp_size * args.cp_size} != world={world}"
    )

    n_train = sum(1 for _ in open(args.dataset_root / "training.jsonl"))
    total_steps = args.epochs * ceil(n_train / args.global_batch_size)
    max_steps = args.test_nsteps if args.test else total_steps
    lr_warmup_steps = int(args.lr_warmup_ratio * total_steps)

    # NOTE: with MegatronStrategy you do not pass accumulate_grad_batches to the Trainer
    # Micro-batching to reach the global batch size is handled internally
    # We compute it ourselves for throughput math.
    grad_acc = args.global_batch_size // (args.batch_size * args.dp_size)

    if rank == 0:
        print(f"World size                : {world}")
        print(f"Global batch size         : {args.global_batch_size}")
        print(f"Gradient accumulation     : {grad_acc}")
        print(f"Micro batch size (per GPU): {args.batch_size}")
        print(f"Sequence length           : {args.seq_length}")
        print(f"Activation checkpointing  : {args.activation_checkpointing}")
        print(f"FP8 training              : {args.fp8}")

    # 3. Model (Megatron-Core native). Weights are restored later via AutoResume.
    tokenizer = AutoTokenizer(args.model_path)
    model_cls, config_cls = MODEL_ARCHS[args.model_path.name]
    model_config = config_cls(seq_length=args.seq_length)
    # With CP>1 the sequence is split across ranks, so one rank's slice may contain
    # zero answer tokens. The default loss averages per rank, which then divides by
    # that local zero -> NaN. This flag instead sums the loss and divides by the total
    # token count over the whole CP group, so no rank divides by its own zero.
    model_config.calculate_per_token_loss = True
    if args.activation_checkpointing:
        model_config.recompute_granularity = "full"
        model_config.recompute_method = "uniform"
        model_config.recompute_num_layers = 1
    model = model_cls(model_config, tokenizer=tokenizer)

    # Distributed training strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=args.cp_size,
        sequence_parallel=args.sequence_parallel,
        pipeline_dtype=torch.bfloat16,
    )

    if rank == 0:
        print(f"DP size                  : {args.dp_size}")
        print(f"PP size                  : {args.pp_size}")
        print(f"TP size                  : {args.tp_size}")
        print(f"CP size                  : {args.cp_size}")
        print(f"Sequence parallel        : {args.sequence_parallel}")

    # 4. Data processing (file-based SFT).
    data = llm.FineTuningDataModule(
        dataset_root=str(args.dataset_root),
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        micro_batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        num_workers=args.num_workers,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
        },
    )  # https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.2/nemo/collections/llm/gpt/data/fine_tuning.py#L35

    # 5. Training preparation
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad=0.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    scheduler = CosineAnnealingScheduler(
        max_steps=max_steps,
        warmup_steps=lr_warmup_steps,
        min_lr=args.min_lr,
    )
    optim = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=scheduler)

    # 6. Training loop

    # Wandb logging
    wandb = None
    if args.wandb_project is not None:
        from lightning.pytorch.loggers import WandbLogger

        wandb = WandbLogger(
            project=args.wandb_project,
            name=(
                f"{args.model_path.name}"
                f"_nodes{args.num_nodes}"
                f"_devices{args.devices_per_node}"
                f"_strat_MegatronStrategy"
                f"_dp{args.dp_size}"
                f"_pp{args.pp_size}"
                f"_tp{args.tp_size}"
                f"_cp{args.cp_size}"
                f"_sp{args.sequence_parallel}"
                f"_gbs{args.global_batch_size}"
                f"_mbs{args.batch_size}"
                f"_seqlen{args.seq_length}"
            ),
        )

    callbacks = []

    # Benchmark logging
    callbacks.append(
        MegatronBenchmarkCallback(
            rank,
            max_steps,  # total number of weight updates
            args.global_batch_size,  # number of samples per weight update
            args.seq_length,  # number of tokens per sample
            ceil(n_train / args.global_batch_size),  # number of steps per epoch
            1,  # 1 warmup step
        )
    )

    # FLOPs counting
    callbacks.append(
        FlopCounterCallback(enabled=args.enable_flop_counter and rank == 0)
    )

    # Pytorch Profiler
    if args.pytorch_profiler:
        callbacks.append(TorchProfilerCallback(rank))

    trainer = nl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=args.devices_per_node,
        num_nodes=args.num_nodes,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed", fp8="hybrid" if args.fp8 else None
        ),
        logger=wandb,
        callbacks=callbacks,
        max_epochs=args.epochs,
        max_steps=max_steps,
        limit_val_batches=0,  # for warmup in sanity check
        num_sanity_val_steps=0,  # useless since torch.compile re-compile at each epoch
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=False,
        enable_model_summary=False,
        use_distributed_sampler=False,  # MegatronStrategy uses its own (Megatron) data sampler
        # No accumulate_grad_batches / gradient_clip_val here: micro-batching is managed by
        # Megatron's microbatch calculator, and grad clipping by OptimizerConfig.clip_grad (TP-safe).
    )

    # Restore local HF weights (the model's "hf" importer converts them on the fly).
    resume = nl.AutoResume(
        restore_config=nl.RestoreConfig(path=str(args.nemo_ckpt_path))
    )

    # The loss (masked cross-entropy) lives in the model, not here.
    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
        resume=resume,
        peft=None,
    )


if __name__ == "__main__":
    main()
