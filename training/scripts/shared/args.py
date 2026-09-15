import argparse
import logging
import os
from argparse import BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import OmegaConf
from scripts.slurm.utils import load_config

logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")


@dataclass
class Configuration:
    # MODEL
    model_path: str
    model_name: str

    # DATA
    dataset_path: str
    dataset_name: str
    dataloader_num_workers: int
    dataset_train_files: list[str]
    dataset_validation_files: list[str]

    # TRAINING PARAMS
    global_batch_size: int
    batch_size: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_length: int
    gradient_accumulation_steps: int
    epochs: int | None
    max_steps: int | None
    precision: Literal["fp32", "fp16", "bf16"]
    gradient_checkpointing: int
    enable_compile: bool

    # LOGGING
    logging_steps: int

    # OUTPUTS
    output_dir: str

    # GPU SPECS
    gpu_name: str
    peak_flops: float

    # FSDP
    fsdp_max_comm_comp_overlap: bool


@dataclass
class MegatronConfiguration(Configuration):
    # MEGATRON
    ## Mode
    prepare: bool
    ## Benchmarking / debugging arguments
    test: bool
    test_nsteps: bool
    enable_flop_counter: bool
    pytorch_profiler: bool
    ## DataLoader related arguments
    dataset_root: Path
    ## Optimizer related arguments
    min_lr: str
    ## Model related arguments
    nemo_ckpt_path: str
    fp8: bool
    ## Distributed training arguments
    devices_per_node: int
    num_nodes: int
    dp_size: int
    pp_size: int
    tp_size: int
    cp_size: int
    ep_size: int
    sequence_parallel: bool
    log_every_n_steps: int
    wandb_project: str


# --- Argument Parsing ---#
def get_parser():
    parser = argparse.ArgumentParser(
        # TODO: Fill in argparse description
        description="ToDo"
    )
    parser.add_argument(
        "--yaml_file", type=str, required=True, help="Path to yaml configuration"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )

    parser.add_argument("--logging_steps", type=float, default=1, help="Logging Steps")

    # Dataloader related arguments
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )

    # Optimizer related arguments
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for Adam."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="Weight decay for Adam."
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay."
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps for linear LR warmup (0.1 = 10%).",
    )
    return parser


def get_fsdp_parser():
    parser = get_parser()
    parser.add_argument(
        "--max_comm_comp_overlap",
        default=False,
        action="store_true",
        help=(
            "Whether to enable maximum communication-computation overlap in FSDP. "
            + "Sets 'forward_prefetch' to True and 'limit_all_gathers' to False in FSDP config."
            + "Note: This may increase GPU memory usage, so use with caution on memory-constrained setups."
        ),
    )
    return parser


def get_deepspeed_parser():
    parser = get_parser()
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    return parser


MEGATRON_DEFAULT_CKPT_PATH = "nemo_ckpt"


def get_megatron_parser():
    """Process command-line arguments."""
    parser = get_parser()

    # Mode
    parser.add_argument(
        "--prepare",
        action=BooleanOptionalAction,
        default=False,
        help="One-time prep (single process): build JSONL from the HF dataset. No training.",
    )

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
    parser.add_argument(
        "--nemo-ckpt-path",
        type=Path,
        default=Path(f"./{MEGATRON_DEFAULT_CKPT_PATH}"),
        help="Where the converted NeMo checkpoint is written.",
    )
    # parser.add_argument(
    #    "--fp8",
    #    action=BooleanOptionalAction,
    #    default=False,
    #    help="Enable FP8 training (via MegatronMixedPrecision).",
    # )

    # Logging / Checkpointing arguments
    parser.add_argument(
        "--log-every-n-steps", type=int, default=10, help="Logging frequency."
    )
    parser.add_argument("--wandb-project", help="Wandb project name.")

    return parser


def get_peak_gpu_flops(config: BenchmarkConfig) -> float:

    # FIXME: get theoretical flops
    key = "theoretical_peak_fp32_tflops"
    peak_gpu_tflops = float(config.arch.gpu.theoretical_peak_fp32_tflops)
    if config.model.training.precision == "bf16":
        key = "theoretical_peak_bf16_tensor_tflops"
        peak_gpu_tflops = float(config.arch.gpu.theoretical_peak_bf16_tensor_tflops)
    elif config.model.training.precision == "fp16":
        key = "theoretical_peak_fp16_tensor_tflops"
        peak_gpu_tflops = float(config.arch.gpu.theoretical_peak_fp16_tensor_tflops)

    msg = f"Found {key} = {peak_gpu_tflops}"
    logger.info(msg)
    return peak_gpu_tflops


def construct_config(args: argparse.Namespace) -> Configuration:
    config = load_config(args.yaml_file)

    RUNID = os.environ["SLURM_JOB_ID"]
    RUNJD = os.environ.get("SLURM_STEP_ID", "0")
    OUTPUT_DIR = os.path.join(
        os.environ.get("LOG_DIR", os.path.join("outputs", "logs", "pyft")),
        RUNID,
        f"outputs/training-results/step-{RUNJD}",
    )

    train_args = Configuration(
        model_path=config.model.path,
        model_name=config.model.path.split("/")[-1],
        dataset_path=config.dataset.path,
        dataset_name=config.dataset.name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_train_files=list(config.dataset.train) if config.dataset.train else [],
        dataset_validation_files=list(config.dataset.validation)
        if config.dataset.validation
        else [],
        precision=config.model.training.precision,
        global_batch_size=config.model.training.global_batch_size,
        batch_size=config.model.training.batch_size,
        gradient_accumulation_steps=config.model.training.grad_accum,
        lr=config.model.training.lr,
        enable_compile=config.model.training.enable_compile,
        max_steps=config.model.training.steps,
        epochs=config.model.training.epochs,
        max_length=config.model.training.max_model_length
        if not OmegaConf.is_missing(config.model.training, "max_model_length")
        else config.dataset.max_seq_len,
        output_dir=OUTPUT_DIR,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=config.model.training.gradient_checkpointing,
        gpu_name=config.arch.gpu.name,
        peak_flops=get_peak_gpu_flops(config),
        fsdp_max_comm_comp_overlap=args.max_comm_comp_overlap
        if "max_comm_comp_overlap" in args
        else False,
    )

    logger.info("yaml_file_path = %s", args.yaml_file)
    logger.info("model_path = %s", train_args.model_path)
    logger.info("model_name = %s", train_args.model_name)
    logger.info("dataset_path = %s", train_args.dataset_path)
    logger.info("dataset_name = %s", train_args.dataset_name)
    logger.info("dataset_train_files = %s", train_args.dataset_train_files)
    logger.info("dataset_validation_files = %s", train_args.dataset_validation_files)
    logger.info("dataset = %s", train_args.dataset_name)
    logger.info("precision = %s", train_args.precision)
    logger.info("batch_size = %s", train_args.batch_size)
    logger.info(
        "gradient_accumulation_steps = %s", train_args.gradient_accumulation_steps
    )
    logger.info("lr = %s", train_args.lr)
    logger.info("max_steps = %s", train_args.max_steps)
    logger.info("epochs = %s", train_args.epochs)
    logger.info("max_length = %s", train_args.max_length)
    logger.info("enable_compile = %s", train_args.enable_compile)
    logger.info("output_dir = %s", train_args.output_dir)
    return train_args


def megatron_construct_config(args: argparse.Namespace) -> MegatronConfiguration:
    cfg = construct_config(args)
    config = load_config(args.yaml_file)

    from configs_hydra.dataclasses_hydra.arch import PrecisionType

    train_args = MegatronConfiguration(
        prepare=args.prepare,
        ## Benchmarking / debugging arguments
        test=args.test,
        test_nsteps=args.test_nsteps,
        enable_flop_counter=args.enable_flop_counter,
        pytorch_profiler=args.pytorch_profiler,
        ## DataLoader related arguments
        dataset_root=Path(cfg.dataset_path),
        ## Optimizer related arguments
        min_lr=args.min_lr,
        ## Model related arguments
        nemo_ckpt_path=os.path.join(cfg.model_path, MEGATRON_DEFAULT_CKPT_PATH),
        fp8=config.model.training.precision == PrecisionType.bf16_fp8.value,
        ## Distributed training arguments
        devices_per_node=config.slurm.sbatch.gpus_per_node,
        num_nodes=config.slurm.sbatch.nodes,
        dp_size=config.framework.megatron_parallelism.dp,
        pp_size=config.framework.megatron_parallelism.pp,
        tp_size=config.framework.megatron_parallelism.tp,
        cp_size=config.framework.megatron_parallelism.cp,
        ep_size=config.framework.megatron_parallelism.cp,
        sequence_parallel=config.framework.megatron_parallelism.sp,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        **cfg.__dict__,
    )
    return train_args
