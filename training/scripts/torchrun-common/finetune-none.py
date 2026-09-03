import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# # sys.path.append("../..")

import logging
from datetime import datetime

import torch
from shared.args import construct_args, get_parser
from shared.custom_train import PerformanceTrackingSFTTrainer
from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import print_rank
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
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
    # Get main id
    jobid = os.environ["SLURM_JOB_ID"]
    jobstepid = os.environ["SLURM_STEP_ID"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = construct_args(get_parser().parse_args())

    logger.info("configs = %s", configs)
    model_name = configs["model"]["path"]
    logger.info("model_name = %s", model_name)
    dataset_path = configs["dataset"]["path"]
    logger.info("data = %s", dataset_path)
    dataset_name = configs["dataset"]["name"]
    logger.info("dataset = %s", dataset_name)
    precision = configs["model"]["training"]["precision"]
    logger.info("precision = %s", precision)
    batch_size = configs["model"]["training"]["batch_size"]
    logger.info("batch_size = %s", batch_size)
    gradient_accumulation_steps = configs["model"]["training"]["grad_accum"]
    logger.info("gradient_accumulation_steps = %s", gradient_accumulation_steps)
    lr = configs["model"]["training"]["lr"]
    logger.info("lr = %s", lr)
    enable_compile = configs["model"]["training"]["enable_compile"]
    logger.info("enable_compile = %s", enable_compile)
    max_steps = configs["model"]["training"]["steps"]
    logger.info("max_steps = %s", max_steps)
    epochs = configs["model"]["training"]["epochs"]
    logger.info("epochs = %s", epochs)
    max_length = configs["model"]["training"]["max_model_length"]
    logger.info("epochs = %s", max_length)
    # TODO: fix
    max_length = 1024
    run_dir = configs["run_dir"]
    logger.info("run_dir = %s", run_dir)
    output_dir = os.path.join(run_dir, args.output_dir)
    logger.info("output_dir = %s", output_dir)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer Loaded")

    # --------------------#
    # Handle dataset path #
    # --------------------#
    train_dataset, eval_dataset = load_and_prepare_raw_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, test_size=0.1
    )

    # --- Precision selection ---
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
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
            # overwrite_output_dir=True,
            # num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # effective batch size
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_strategy="no",
            save_total_limit=1,
            fp16=True if args.precision == "fp16" else False,
            bf16=True if args.precision == "bf16" else False,
            optim="adamw_torch",
            logging_dir=f"{output_dir}/logs",
            report_to="none",
            eval_strategy="no",
            eval_steps=None,
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
            pad_to_multiple_of=args.max_length,
            # TODO: If your dataset is already tokenized (input_ids present), set:
            #   dataset_kwargs={"skip_prepare_dataset": True}
            #   and remove dataset_text_field above.
            # torch model compilation
            **compilation_args,
        )

        logger.info(f"Loading Model... dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
        )
        model.to(device)
        logger.info("Model Loaded")

        # Conditionally add either epochs or max_steps
        training_args.num_train_epochs = epochs if epochs is not None else 1
        if max_steps is not None:
            training_args.max_steps = int(max_steps)

        print(
            f"GPU_NAME: {args.gpu['name']} | Using peak GPU TFLOPS for MFU calculation: {args.peak_flops} TFLOPS",
        )

        flopsCallback_megatronLM = mfu_callback_from_hf_config(
            model,
            tokenizer,
            gpu_peak_flops=args.peak_flops,
            seq_length=max_length,
        )
        trainer = PerformanceTrackingSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=collate_fn,
            processing_class=tokenizer,
            callbacks=[
                flopsCallback_megatronLM,
            ],
            peak_gpu_tflops=args.peak_flops,
        )

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )

        trainer.train()

        stop_flag["stop"] = True
        time.sleep(2)

        trainer.write_summary(output_dir=output_dir, gpu_stats=gpu_stats_during)

        del trainer
        torch.cuda.empty_cache()
        print("Fine-tuning completed successfully.")

    except Exception as e:
        print("Fine-tuning completed with error.")
        raise e


if __name__ == "__main__":
    main()
