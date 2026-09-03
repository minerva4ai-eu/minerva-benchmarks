import gc
import os
import time

import torch
import torch.distributed as dist
from shared.custom_train import (
    PerformanceTrackingSFTTrainer,  # Must subclass SFTTrainer now
)
from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import (
    get_fsdp_layer_to_wrap,
    print_rank,
)
from transformers import AutoTokenizer
from trl.trainer.sft_config import (
    SFTConfig,  # CHANGED: replaces TrainingArguments + Trainer
)
from utils import parse_args

# parse CLI args
args = parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size


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

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    print_rank(rank, f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank(rank, "Tokenizer Loaded")

    # NOTE: SFTTrainer can handle tokenization internally.
    # load_dataset should return raw (un-tokenized) datasets.
    # The collate_fn is no longer passed to the trainer — SFTTrainer manages collation.
    # If your load_dataset already returns pre-tokenized datasets, set
    # dataset_kwargs={"skip_prepare_dataset": True} in SFTConfig below.
    train_dataset, eval_dataset = load_and_prepare_raw_dataset(
        dataset_name=args.dataset, dataset_path=args.data, test_size=0.1
    )

    # Setup FSDP configuration
    layer = get_fsdp_layer_to_wrap(model_name)
    fsdp_config = {
        "transformer_layer_cls_to_wrap": ",".join(layer)
        if isinstance(layer, list)
        else layer,
        "use_orig_params": True,  # needed for torch.compile + param groups
        "sharding_strategy": "FULL_SHARD",
        "activation_checkpointing": True,
        "activation_checkpointing_kwargs": {"use_reentrant": False},
        "cpu_ram_efficient_loading": True,  # only rank 0 loads from disk to CPU
        "sync_module_states": True,  # rank 0 broadcasts shards directly to GPU
        "offload_params": False,
        "forward_prefetch": False,
        "limit_all_gathers": True,
        "backward_prefetch": "backward_post",
    }

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if args.max_comm_comp_overlap:
        fsdp_config["forward_prefetch"] = True
        fsdp_config["limit_all_gathers"] = False
        fsdp_config["backward_prefetch"] = "backward_pre"
        print_rank(
            rank,
            "Enabled maximum communication-computation overlap in FSDP config: "
            + "forward_prefetch=True, limit_all_gathers=False. "
            + "Note: Monitor GPU memory usage as this may increase it.",
        )

    try:
        compilation_args = {}
        if args.enable_compile:
            torch_compile_backend = "inductor"
            torch_compile_mode = "max-autotune-no-cudagraphs"
            compilation_args = {
                "torch_compile": True,
                "torch_compile_backend": torch_compile_backend,
                "torch_compile_mode": torch_compile_mode,
            }
            print_rank(0, f"Compilation arguments: {compilation_args}")

        # CHANGED: TrainingArguments → SFTConfig
        # SFTConfig is a drop-in superset of TrainingArguments with SFT-specific fields.
        training_args = SFTConfig(
            output_dir=output_dir,
            model_init_kwargs={
                "torch_dtype": dtype,
                "attn_implementation": "flash_attention_2",
                "low_cpu_mem_usage": True,
            },
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=False,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
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
            dataloader_drop_last=True,
            dataloader_num_workers=args.dataloader_num_workers,
            data_seed=32,
            dataloader_persistent_workers=args.dataloader_num_workers > 1,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=8,
            # FSDP Config (unchanged)
            fsdp="full_shard auto_wrap",
            fsdp_config=fsdp_config,
            # --- SFT-specific args ---
            max_length=MAX_LENGTH,  # replaces manual truncation in collator
            dataset_text_field="text",
            packing=True,  # set True to pack short sequences for efficiency
            dataset_kwargs={
                "skip_prepare_dataset": False,
            },
            pad_to_multiple_of=MAX_LENGTH,
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

        flopsCallback_megatronLM = mfu_callback_from_hf_config(
            model_config,
            tokenizer,
            gpu_peak_flops=peak_gpu_tflops,
            seq_length=args.max_length,
        )
        # NOTE: data_collator removed — SFTTrainer handles collation via DataCollatorForLanguageModeling.
        # If you need a custom formatting function instead of dataset_text_field, pass:
        #   formatting_func=lambda x: [f"### Input: {x['input']}\n### Output: {x['output']}"]
        trainer = PerformanceTrackingSFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[flopsCallback_megatronLM],
            peak_gpu_tflops=peak_gpu_tflops,
            # TODO: Uncomment if using a formatting function instead of dataset_text_field:
            # formatting_func=your_formatting_func,
        )

        print_rank(rank, "Trainer initialized and model has been wrapped!")
        # print_rank(rank, ":::::::::")
        # for i in model.named_parameters():
        #    print_rank(rank, f"{i[0]} -> {i[1].device}")
        # print_rank(rank, ":::::::::")

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )
        print_rank(
            rank, f"Accelerator FSDP plugin: {trainer.accelerator.state.fsdp_plugin}"
        )
        print_rank(rank, f"Distributed type: {trainer.accelerator.distributed_type}")

        trainer.train()

        # Stop GPU monitor
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
