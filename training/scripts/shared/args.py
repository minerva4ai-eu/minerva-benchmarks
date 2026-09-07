import argparse
import logging
import os
from dataclasses import dataclass
from typing import Literal

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


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
    run_dir: str

    # GPU SPECS
    gpu_name: str
    peak_flops: float

    # FSDP
    fsdp_max_comm_comp_overlap: bool = False


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

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )

    parser.add_argument("--lr", type=float, default=1e-04, help="Learning rate")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate",
    )
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight Decay")
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
    # parser.add_argument(
    #    "--gradient_checkpointing",
    #    action="store_true",
    #    help="Enable gradient checkpointing to save memory",
    # )
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


def load_config(path: str) -> BenchmarkConfig:
    return OmegaConf.load(path)


def construct_args(args: argparse.Namespace) -> Configuration:
    config = load_config(args.yaml_file)
    train_args = Configuration(
        model_path=config.model.path,
        model_name=config.model.path.split("/")[-1],
        dataset_path=config.dataset.path,
        dataset_name=config.dataset.name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_train_files=config.dataset.train if config.dataset.train else [],
        dataset_validation_files=config.dataset.validation
        if config.dataset.validation
        else [],
        precision=config.model.training.precision,
        batch_size=config.model.training.batch_size,
        gradient_accumulation_steps=config.model.training.grad_accum,
        lr=config.model.training.lr,
        enable_compile=config.model.training.enable_compile,
        max_steps=config.model.training.steps,
        epochs=config.model.training.epochs,
        max_length=config.model.training.max_model_length
        if not OmegaConf.is_missing(config.model.training, "max_model_length")
        else config.dataset.max_seq_len,
        run_dir=config.run_dir,
        output_dir=os.path.join(config.run_dir, args.output_dir),
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=config.model.training.gradient_checkpointing,
        gpu_name=config.arch.gpu.name,
        peak_flops=get_peak_gpu_flops(config),
    )

    if "max_comm_comp_overlap" in args:
        train_args.fsdp_max_comm_comp_overlap = args.max_comm_comp_overlap

    logger.info("model_path = %s", train_args.model_path)
    logger.info("model_name = %s", train_args.model_name)
    logger.info("dataset_path = %s", train_args.dataset_path)
    logger.info("dataset_name = %s", train_args.dataset_name)
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
    logger.info("run_dir = %s", train_args.run_dir)
    logger.info("output_dir = %s", train_args.output_dir)
    return train_args


"""def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # TODO: Fill in argparse description
        description="ToDo"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-04, help="Learning rate")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate",
    )
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight Decay")
    parser.add_argument("--logging_steps", type=float, default=1, help="Logging Steps")

    parser.add_argument("--max_steps", type=float, default=None, help="Maximum steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length")
    parser.add_argument("--epochs_save_every", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision type for model weights (fp32, fp16, bf16)",
    )
    parser.add_argument(
        "--enable_compile",
        default=False,
        action="store_true",
        help="Disable torch.compile() in the custom trainer to avoid compilation-related device/runtime issues.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="",
    )

    return parser"""
