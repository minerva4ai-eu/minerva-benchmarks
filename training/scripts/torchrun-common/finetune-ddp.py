import gc
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# # sys.path.append("../..")

import logging
from datetime import datetime

import torch
import torch.distributed as dist
from shared.args import construct_args, get_parser
from shared.custom_train import PerformanceTrackingSFTTrainer
from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import (
    print_rank,
)
from transformers import AutoTokenizer
from trl.trainer.sft_config import (
    SFTConfig,
)

RUNID = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d%H%M%S"))
RUNJD = os.environ.get("SLURM_STEP_ID")
LOG_DIR = os.path.join("outputs", "logs", "pyft", RUNID)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# FIXME: logging level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s |  %(levelname)s | %(name)s : %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"minerva-step{RUNJD}.log"))],
)

logger = logging.getLogger(__name__)


def is_main_process():
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


# --- Main ---
def main():

    logger.info(
        "localrankvar = %s, rankvar = %s, grouprankvar = %s, rolerankvar = %s, localworldsizevar = %s, worldsizevar = %s",
        os.environ.get("LOCAL_RANK", None),
        os.environ.get("RANK", None),
        os.environ.get("GROUP_RANK", None),
        os.environ.get("ROLE_RANK", None),
        os.environ.get("LOCAL_WORLD_SIZE", None),
        os.environ.get("WORLD_SIZE", None),
    )

    # Get main id
    jobid = os.environ["SLURM_JOB_ID"]
    jobstepid = os.environ["SLURM_STEP_ID"]
    jobsteprocid = os.environ["SLURM_PROCID"]

    # Get rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ["RANK"])
    torch.cuda.empty_cache()

    args = construct_args(get_parser().parse_args())

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    print_rank(rank, f"Loading tokenizer... {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank(rank, "Tokenizer Loaded")

    # ---------------------------------------------------------------------
    # Handle dataset path (string or dict)
    # ---------------------------------------------------------------------

    train_dataset, eval_dataset = load_and_prepare_raw_dataset(
        dataset_name=args.dataset_name, dataset_path=args.dataset_path, test_size=0.1
    )

    # --- Precision selection ---
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    try:
        compilation_args = {}
        if args.enable_compile:
            torch_compile_backend = "inductor"
            torch_compile_mode = "default"
            compilation_args = {
                "torch_compile": True,
                "torch_compile_backend": torch_compile_backend,
                "torch_compile_mode": torch_compile_mode,
            }
            print_rank(0, f"Compilation arguments: {compilation_args}")

        training_args = SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_strategy="no",
            save_total_limit=1,
            fp16=args.precision == "fp16",
            bf16=args.precision == "bf16",
            optim="adamw_torch",
            logging_dir=f"{args.output_dir}/logs",
            report_to="none",
            eval_steps=None,
            ddp_timeout=1800,
            # Dataloader is created automatically from trainer
            dataloader_drop_last=True,
            dataloader_num_workers=args.dataloader_num_workers,
            data_seed=32,
            dataloader_persistent_workers=args.dataloader_num_workers > 1,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=8,
            # --- SFT-specific args ---
            max_length=args.max_length,  # replaces manual truncation in collator
            dataset_text_field="text",  # TODO: set to your dataset's text column name
            # OR remove and use formatting_func below
            packing=True,  # set True to pack short sequences for efficiency
            dataset_kwargs={"skip_prepare_dataset": False},
            # TODO: If your dataset is already tokenized (input_ids present), set:
            #   dataset_kwargs={"skip_prepare_dataset": True}
            #   and remove dataset_text_field above.
            # torch model compilation
            **compilation_args,
        )
        training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
        if args.max_steps is not None:
            training_args.max_steps = int(args.max_steps)

        print_rank(
            0,
            f"GPU_NAME: {args.gpu_name} | Using peak GPU TFLOPS for MFU calculation: {args.peak_flops} TFLOPS",
        )

        print_rank(f"Loading Model... dtype: {dtype}")

        model_config = AutoConfig.from_pretrained(model_name)
        flops_callback = mfu_callback_from_hf_config(
            model_config,
            tokenizer,
            gpu_peak_flops=args.peak_flops,
            seq_length=args.max_length,
        )
        trainer = PerformanceTrackingSFTTrainer(
            model=args.model_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=collate_fn,
            processing_class=tokenizer,
            callbacks=[
                flops_callback,
            ],
            peak_gpu_tflops=args.peak_flops,
        )

        # TODO: Check behavior (ddp measures CPU RAM and disables cache)

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )
        print_rank(rank, f"Distributed type: {trainer.accelerator.distributed_type}")

        trainer.train()

        stop_flag["stop"] = True
        time.sleep(2)

        trainer.write_summary(
            output_file=os.path.join(
                args.output_dir,
                f"training_summary_job{jobid}-step{jobstepid}-task{jobsteprocid}-{rank}.json",
            ),
            gpu_stats=gpu_stats_during,
        )

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        print_rank(rank, "Fine-tuning completed successfully.")

    except Exception as e:
        print_rank(rank, "Fine-tuning completed with error.")
        raise e


if __name__ == "__main__":
    main()
