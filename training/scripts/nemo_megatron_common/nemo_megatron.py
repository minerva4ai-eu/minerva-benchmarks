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
import logging
import os
from math import ceil
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import torch
from datasets import Dataset
from lightning.pytorch.utilities.types import STEP_OUTPUT
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from scripts.nemo_megatron_common.utils import MegatronBenchmarkCallback
from scripts.shared.args import (
    get_megatron_parser,
    megatron_construct_config,
)

# scripts.shared.data import load_and_prepare_raw_dataset
from scripts.shared.logger import RankAdapter, setup_logging
from scripts.slurm.utils import load_config
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from scripts.shared.args import MegatronConfiguration

# Add the (Model, Config) pair that matches your HF checkpoint here.
MODEL_ARCHS = {
    "Qwen2.5-7B-Instruct": (llm.Qwen2Model, llm.Qwen25Config7B),
    "Qwen2.5-72B-Instruct": (llm.Qwen2Model, llm.Qwen25Config72B),
}

_args = get_megatron_parser().parse_args()
config = load_config(_args.yaml_file)
setup_logging(level=logging.INFO, cfg=config)
logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")
logger_rank = RankAdapter(logger, {})

args = megatron_construct_config(_args)


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

            logger.info(
                f"Median FLOPs/step: {np.median(self.flops_list) / 1e12:.1f} TFLOPs"
            )


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


def prepare(cfg: "MegatronConfiguration") -> None:
    """One-time, single-process prep: build training.jsonl / validation.jsonl from the HF dataset."""
    import sys
    from pathlib import Path

    # Convert HF checkpoint into NeMo checkpoint
    # https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.2/nemo/collections/llm/api.py#L577
    from nemo.collections.llm import import_ckpt

    sys.stdout.flush()
    model_path = Path(cfg.model_path)
    logger.info(f"[prepare] model_path={cfg.model_path}")
    logger.info(f"[prepare] model_path.exists()={model_path.exists()}")
    if os.path.exists(cfg.nemo_ckpt_path):
        logger.info(
            f"[prepare] Model checkpoint already exists '{cfg.nemo_ckpt_path}'. Skipping conversion..."
        )
    else:
        if model_path.exists():
            logger.info(f"[prepare] model files: {list(model_path.iterdir())[:10]}")

        logger.info(f"[prepare] looking up MODEL_ARCHS for '{model_path.name}'")
        model_cls, config_cls = MODEL_ARCHS[model_path.name]
        logger.info(f"[prepare] creating tokenizer for {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        logger.info("[prepare] tokenizer created, calling import_ckpt")
        logger.info(f"[prepare]   model_cls={model_cls}")
        logger.info(f"[prepare]   config_cls={config_cls}")
        logger.info(f"[prepare]   source=hf://{model_path}")
        logger.info(f"[prepare]   output_path={cfg.nemo_ckpt_path}")
        import_ckpt(
            model=model_cls(config_cls(seq_length=cfg.max_length), tokenizer=tokenizer),
            source=f"hf://{model_path}",  # local dir -> "hf:///abs/path", géré par l'importeur hf
            output_path=cfg.nemo_ckpt_path,
            overwrite=False,
        )
        logger.info(f"[prepare] Converted HF -> NeMo at {cfg.nemo_ckpt_path}")

    # Prepare dataset
    from datasets import load_dataset

    train_split_name = "training"
    valid_split_name = "validation"

    train_file = os.path.join(cfg.dataset_root, f"{train_split_name}.jsonl")

    if os.path.exists(train_file):
        logger.info(
            f"[prepare] File already exists '{train_file}'. Skipping conversion..."
        )
        return
    logger.info(f"[prepare] loading dataset from {cfg.dataset_path}")
    os.makedirs(cfg.dataset_root, exist_ok=True)
    dataset = load_dataset(str(cfg.dataset_path))
    logger.info(f"[prepare] dataset loaded: {list(dataset.keys())}")

    def dump(out_file: str, hf_split: Dataset) -> None:

        with open(out_file, "w") as f:
            f.writelines(
                json.dumps({"messages": ex["messages"]}) + "\n" for ex in hf_split
            )
        print(f"Wrote {out_file} ({len(hf_split)} examples)")

    # Create training.jsonl
    dump(train_split_name, dataset["train"])
    # Reuse train as validation for warmup/sanity if no val split, mirroring the original script.
    dump(
        valid_split_name,
        dataset["validation"] if "validation" in dataset else dataset["train"],
    )


def main(repeatid: int):
    """Run SFT with a Megatron-Core model using Megatron strategy."""

    # 2. Distributed Training Setup
    world = args.devices_per_node * args.num_nodes
    rank = int(os.environ.get("RANK", 0))  # Set by torchrun

    assert args.dp_size * args.pp_size * args.tp_size * args.cp_size == world, (
        f"4D mismatch: DP*PP*TP*CP={args.dp_size * args.pp_size * args.tp_size * args.cp_size} != world={world}"
    )

    n_train = sum(1 for _ in open(args.dataset_root.joinpath("training.jsonl")))
    epochs = args.epochs if args.epochs else 1
    total_steps = (
        args.max_steps
        if args.max_steps
        else epochs * ceil(n_train / args.global_batch_size)
    )
    max_steps = args.test_nsteps if args.test else total_steps
    lr_warmup_steps = int(args.warmup_ratio * total_steps)

    # NOTE: with MegatronStrategy you do not pass accumulate_grad_batches to the Trainer
    # Micro-batching to reach the global batch size is handled internally
    # We compute it ourselves for throughput math.
    grad_acc = args.global_batch_size // (args.batch_size * args.dp_size)

    logger.info(f"World size                : {world}")
    logger.info(f"Global batch size         : {args.global_batch_size}")
    logger.info(f"Gradient accumulation     : {grad_acc}")
    logger.info(f"Micro batch size (per GPU): {args.batch_size}")
    logger.info(f"Sequence length           : {args.max_length}")
    logger.info(f"Activation checkpointing  : {args.gradient_checkpointing}")
    logger.info(f"FP8 training              : {args.fp8}")

    # 3. Model (Megatron-Core native). Weights are restored later via AutoResume.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_cls, config_cls = MODEL_ARCHS[args.model_name]
    model_config = config_cls(seq_length=args.max_length)
    # With CP>1 the sequence is split across ranks, so one rank's slice may contain
    # zero answer tokens. The default loss averages per rank, which then divides by
    # that local zero -> NaN. This flag instead sums the loss and divides by the total
    # token count over the whole CP group, so no rank divides by its own zero.
    model_config.calculate_per_token_loss = True
    if args.gradient_checkpointing:
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
        # pipeline_dtype=torch.bfloat16,
    )

    logger.info(f"DP size                  : {args.dp_size}")
    logger.info(f"PP size                  : {args.pp_size}")
    logger.info(f"TP size                  : {args.tp_size}")
    logger.info(f"CP size                  : {args.cp_size}")
    logger.info(f"EP size                  : {args.ep_size}")
    logger.info(f"Sequence parallel        : {args.sequence_parallel}")

    # 4. Data processing (file-based SFT).
    data = llm.FineTuningDataModule(
        dataset_root=args.dataset_root,
        seq_length=args.max_length,
        tokenizer=tokenizer,
        micro_batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        num_workers=args.dataloader_num_workers,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
        },
    )  # https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.2/nemo/collections/llm/gpt/data/fine_tuning.py#L35

    # 5. Training preparation

    callbacks = []
    plugins = []
    # Benchmark logging
    callbacks.append(
        MegatronBenchmarkCallback(
            rank,
            max_steps,  # total number of weight updates
            args.global_batch_size,  # number of samples per weight update
            args.max_length,  # number of tokens per sample
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

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=0.0,
        use_distributed_optimizer=True,
    )
    if args.precision == "bf16_fp8":
        plugins.append(
            nl.MegatronMixedPrecision(
                precision="bf16-mixed", fp8="hybrid" if args.fp8 else None
            )
        )
        opt_config.bf16 = True
        strategy.pipeline_dtype = torch.bfloat16
    elif args.precision == "bf16":
        plugins.append(nl.MegatronMixedPrecision(precision="bf16-mixed"))
        opt_config.bf16 = True
        strategy.pipeline_dtype = torch.bfloat16
    elif args.precision == "fp16":
        plugins.append(nl.MegatronMixedPrecision(precision="fp16_mixed"))
        opt_config.fp16 = True
        strategy.pipeline_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported Megatron precision: {args.precision}")
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
                f"{args.model_name}"
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
                f"_seqlen{args.max_length}"
            ),
        )
    trainer = nl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=args.devices_per_node,
        num_nodes=args.num_nodes,
        plugins=plugins,
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

    jobid = os.environ.get("SLURM_JOB_ID", "unknown")
    jobstepid = os.environ.get("SLURM_STEP_ID", "0")
    jobsteprocid = os.environ.get("SLURM_PROCID", "0")
    summary_file = os.path.join(
        args.output_dir,
        f"nemo_experiments-{jobid}-step{jobstepid}-task{jobsteprocid}",
        f"repeatid-{repeatid}",
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    nemo_logger = nl.NeMoLogger(
        log_dir=os.path.abspath(os.path.dirname(summary_file)),
        name="MINERVA-Bench-Nemo-Megatron",
    )
    nemo_logger.setup(trainer, resume_if_exists=True)

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
    if args.prepare:
        logger.info("_________________________________")
        logger.info("Running preparation ")
        logger.info("_________________________________")

        prepare(args)
    else:
        try:
            for repeatid in range(config.experiment.repeat):
                logger.info("_________________________________")
                logger.info(f"Running repeatid: {repeatid}")
                logger.info("_________________________________")
                main(repeatid)
        except Exception as e:
            logger.exception("Error occured during execution of main")
            raise e
