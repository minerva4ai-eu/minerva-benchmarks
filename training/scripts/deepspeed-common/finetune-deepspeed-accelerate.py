import gc
import os
import time

import torch
import torch.distributed as dist
from shared.args import get_deepspeed_parser
from shared.custom_train import PerformanceTrackingSFTTrainer
from shared.data import collate_fn, get_train_eval_path, load_prepared_packed_dataset

# from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import print_rank
from transformers import AutoConfig, AutoTokenizer
from trl.trainer.sft_config import (
    SFTConfig,
)

args = get_deepspeed_parser().parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size

from_pretrained = True


def is_main_process():
    # HF/torchrun sets LOCAL_RANK env var; fallback to RANK
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


# --- Main ---
def main():
    # Get rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ["RANK"])
    torch.cuda.empty_cache()

    model_name = args.model
    output_dir = args.output_dir
    train_path, eval_path = get_train_eval_path(args)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    print_rank(rank, f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank(rank, "Tokenizer Loaded")

    # ---------------------------------------------------------------------
    # Handle dataset path (string or dict)
    # ---------------------------------------------------------------------

    # train_dataset, eval_dataset = load_and_prepare_raw_dataset(
    #    dataset_name=args.dataset, dataset_path=args.data, test_size=0.1
    # )
    train_dataset = load_prepared_packed_dataset(train_path)

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
        # --- 1. LOAD MODEL EFFICIENTLY ON META DEVICE / LOW MEMORY ---
        model_init_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            # "device_map": "auto",
        }
        if from_pretrained:
            print_rank(rank, f"Loading model architecture for {model_name}...")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_init_kwargs
            )
            model_init_kwargs = {}
        else:
            model = model_name
        training_args = SFTConfig(
            output_dir=output_dir,
            model_init_kwargs=model_init_kwargs,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_strategy="no",
            save_total_limit=1,
            fp16=args.precision == "fp16",
            bf16=args.precision == "bf16",
            optim="adamw_torch",
            logging_dir=f"{output_dir}/logs",
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
            # Deepspeed Config
            deepspeed=args.deepspeed_config_file,
            # --- SFT-specific args ---
            # max_length=MAX_LENGTH,  # replaces manual truncation in collator
            # dataset_text_field="text",  # TODO: set to your dataset's text column name
            # OR remove and use formatting_func below
            # packing=True,  # set True to pack short sequences for efficiency
            # dataset_kwargs={"skip_prepare_dataset": False},
            # pad_to_multiple_of=MAX_LENGTH,
            # TODO: If your dataset is already tokenized (input_ids present), set:
            #   dataset_kwargs={"skip_prepare_dataset": True}
            #   and remove dataset_text_field above.
            # torch model compilation
            **compilation_args,
        )

        training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
        if args.max_steps is not None:
            training_args.max_steps = int(args.max_steps)

        _peak_gpu_tflops = os.environ.get("GPU_PEAK_TFLOPS")
        peak_gpu_tflops = float(_peak_gpu_tflops) if _peak_gpu_tflops else None
        gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
        print_rank(
            0,
            f"GPU_NAME: {gpu_name} | Using peak GPU TFLOPS for MFU calculation: {peak_gpu_tflops} TFLOPS",
        )

        print_rank(f"Loading Model... dtype: {dtype}")

        model_config = AutoConfig.from_pretrained(model_name)
        flops_callback = mfu_callback_from_hf_config(
            model_config,
            tokenizer,
            gpu_peak_flops=peak_gpu_tflops,
            seq_length=args.max_length,
        )
        trainer = PerformanceTrackingSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            data_collator=collate_fn,
            processing_class=tokenizer,
            callbacks=[
                flops_callback,
            ],
            peak_gpu_tflops=peak_gpu_tflops,
        )

        print_rank(rank, "Trainer initialized and model has been wrapped!")

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )
        print_rank(rank, f"Distributed type: {trainer.accelerator.distributed_type}")

        trainer.train()

        stop_flag["stop"] = True
        time.sleep(2)

        trainer.write_summary(output_dir=output_dir, gpu_stats=gpu_stats_during)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        print_rank(rank, "Fine-tuning completed successfully.")

    except Exception as e:
        print_rank(rank, "Fine-tuning completed with error.")
        raise e


if __name__ == "__main__":
    main()
