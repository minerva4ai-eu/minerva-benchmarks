import logging
import os
import time

import torch
from scripts.shared.args import construct_args, get_parser
from scripts.shared.args import get_fsdp_parser as get_parser
from scripts.shared.custom_train import PerformanceTrackingSFTTrainer
from scripts.shared.data import (
    load_and_prepare_raw_dataset,
)
from scripts.shared.flops import mfu_callback_from_hf_config
from scripts.shared.gpu_monitor import start_gpu_monitor
from scripts.shared.logger import RankAdapter, setup_logging
from scripts.shared.utils import (
    print_rank,
    save_summary_stats_json,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.trainer.sft_config import (
    SFTConfig,
)

setup_logging(level=logging.INFO, yaml=get_parser().parse_args().yaml_file)

logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")
logger_rank = RankAdapter(logger, {})


# --- Main ---
def main():
    # Get main id
    rank = os.environ["RANK"]
    jobid = os.environ["SLURM_JOB_ID"]
    jobstepid = os.environ["SLURM_STEP_ID"]
    jobsteprocid = os.environ["SLURM_PROCID"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = construct_args(get_parser().parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading tokenizer... {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer Loaded")

    # --------------------#
    # Handle dataset path #
    # --------------------#
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
            logging_dir=f"{args.output_dir}/logs",
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
        training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
        if args.max_steps is not None:
            training_args.max_steps = int(args.max_steps)

        print(
            f"GPU_NAME: {args.gpu_name} | Using peak GPU TFLOPS for MFU calculation: {args.peak_flops} TFLOPS",
        )

        flopsCallback_megatronLM = mfu_callback_from_hf_config(
            model,
            tokenizer,
            gpu_peak_flops=args.peak_flops,
            seq_length=args.max_length,
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

        try:
            # Start GPU monitor
            gpu_stats_during, stop_flag = start_gpu_monitor(
                interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
            )

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

            print("Fine-tuning completed successfully.")

        except Exception as e:
            save_summary_stats_json(
                summary={
                    "error": str(e),
                },
                output_file=os.path.join(
                    args.output_dir,
                    f"training_summary_job{jobid}-step{jobstepid}-task{jobsteprocid}-{rank}.json",
                ),
            )
            print_rank("Fine-tuning failed to complete!")
            raise e
        finally:
            del trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
