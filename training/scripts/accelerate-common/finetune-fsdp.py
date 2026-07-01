import gc
import os
import time

import psutil
import torch
import torch.distributed as dist
from gpu_monitor import GPUMonitorCallback, start_gpu_monitor
from shared.custom_train import (
    PerformanceTrackingSFTTrainer,  # Must subclass SFTTrainer now
)
from shared.data import load_and_prepare_raw_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.utils import (
    count_parameters,
    get_fsdp_layer_to_wrap,
    print_rank,
    save_summary_stats_json,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import (
    SFTConfig,  # CHANGED: replaces TrainingArguments + Trainer
)
from utils import parse_args

# parse CLI args
args = parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size


def is_main_process():
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


def main():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.empty_cache()

    model_name = args.model
    data = args.data
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
        "use_orig_params": True,
        "sharding_strategy": "FULL_SHARD",
        "activation_checkpointing": True,
        "activation_checkpointing_kwargs": {"use_reentrant": False},
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
        "offload_params": False,
        "forward_prefetch": False,
        "limit_all_gathers": True,
        "backward_prefetch": "backward_post",
    }

    # Check if we're running on CPU and adjust precision accordingly
    if not torch.cuda.is_available() and args.precision in ["bf16", "fp16"]:
        print(f"⚠️  WARNING: {args.precision} not supported on CPU, switching to fp32")
        dtype = torch.float32
    elif args.precision == "fp16":
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

    trainable_params, total_params, trainable_pct = 0, 0, 0
    # CHANGED: TrainingArguments → SFTConfig
    # SFTConfig is a drop-in superset of TrainingArguments with SFT-specific fields.
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
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
        dataset_text_field="text",  # TODO: set to your dataset's text column name
        #                                   # OR remove and use formatting_func below
        packing=True,  # set True to pack short sequences for efficiency
        dataset_kwargs={"skip_prepare_dataset": False},
        # TODO: If your dataset is already tokenized (input_ids present), set:
        #   dataset_kwargs={"skip_prepare_dataset": True}
        #   and remove dataset_text_field above.
        # torch model compilation
        **(
            {
                "torch_compile": True,
                "torch_compile_backend": "inductor",
                "torch_compile_mode": "max-autotune-no-cudagraphs",
            }
            if bool(args.enable_compile)
            else {}
        ),
    )
    try:
        training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
        if args.max_steps is not None:
            training_args.max_steps = int(args.max_steps)

        monitor = GPUMonitorCallback(n_gpus=int(os.environ.get("GPUS_PER_NODE", 1)))

        _peak_gpu_tflops = os.environ.get("GPU_PEAK_TFLOPS")
        peak_gpu_tflops = float(_peak_gpu_tflops) if _peak_gpu_tflops else None
        gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
        print_rank(
            0,
            f"GPU_NAME: {gpu_name} | Using peak GPU TFLOPS for MFU calculation: {peak_gpu_tflops} TFLOPS",
        )

        print_rank(f"Loading Model... dtype: {dtype}")

        if args.enable_compile:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
        print_rank("Model Loaded")

        ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print_rank(rank, f"CPU RAM after model load: {ram_gb:.1f} GB")

        if hasattr(model.config, "use_cache"):
            print_rank(0, "Disabling model's cache")
            model.config.use_cache = False

        print_rank("Model Loaded")
        # print_rank(0, ":::::::::")
        # for i in model.named_parameters():
        #    print_rank(0, f"{i[0]} -> {i[1].device}")
        # print_rank(0, ":::::::::")

        # flop_counter = FlopCounter(model)
        flopsCallback_megatronLM = mfu_callback_from_hf_config(
            model,
            tokenizer,
            gpu_peak_flops=peak_gpu_tflops,
            seq_length=args.max_length,
        )
        # CHANGED: data_collator removed — SFTTrainer handles collation via DataCollatorForLanguageModeling.
        # CHANGED: If you need a custom formatting function instead of dataset_text_field, pass:
        #   formatting_func=lambda x: [f"### Input: {x['input']}\n### Output: {x['output']}"]
        trainer = PerformanceTrackingSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=collate_fn,  # REMOVED: SFTTrainer manages this internally
            processing_class=tokenizer,  # CHANGED: 'tokenizer' param renamed in TRL ≥0.12
            callbacks=[monitor, flopsCallback_megatronLM],
            peak_gpu_tflops=peak_gpu_tflops,
            # TODO: Uncomment if using a formatting function instead of dataset_text_field:
            # formatting_func=your_formatting_func,
        )

        print_rank(rank, "Trainer initialized and model has been wrapped!")
        # print_rank(rank, ":::::::::")
        # for i in model.named_parameters():
        #    print_rank(rank, f"{i[0]} -> {i[1].device}")
        # print_rank(rank, ":::::::::")

        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )

        start_time = time.time()
        trainer.train()
        total_finetune_time = time.time() - start_time

        stop_flag["stop"] = True
        time.sleep(2)

        trainable_params, total_params, trainable_pct = count_parameters(model)

        log_history = trainer.state.log_history
        avg_training_loss = avg_validation_loss = None
        avg_epoch_time_sec = avg_epoch_time_hours = None
        avg_step_time_sec = avg_step_time_hours = None

        total_training_time_secs = getattr(
            trainer.state, "total_training_seconds_custom", total_finetune_time
        )
        tokens_per_gpu_all_epochs = getattr(
            trainer.state, "total_tokens_per_gpu_custom", trainer.total_tokens_this_gpu
        )
        tokens_global_all_epochs = getattr(
            trainer.state, "total_tokens_global_custom", trainer.total_tokens_global
        )

        avg_gpu_flops = getattr(trainer.state, "average_flops_custom")
        avg_gpu_mfu = getattr(trainer.state, "average_mfu_custom")

        if training_args.max_steps:
            avg_step_time_sec = total_training_time_secs / training_args.max_steps
            avg_step_time_hours = avg_step_time_sec / 3600
        else:
            avg_epoch_time_sec = (
                total_training_time_secs / training_args.num_train_epochs
            )
            avg_epoch_time_hours = avg_epoch_time_sec / 3600
            avg_step_time_sec = avg_epoch_time_sec / len(train_dataset)
            avg_step_time_hours = avg_step_time_sec / 3600

        effective_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )

        samples_per_sec = (
            effective_batch_size / avg_step_time_sec if avg_step_time_sec else None
        )
        training_throughput_tokens_per_sec_per_gpu = (
            tokens_per_gpu_all_epochs / total_training_time_secs
            if total_training_time_secs
            else None
        )
        training_throughput_tokens_per_sec_global = (
            tokens_global_all_epochs / total_training_time_secs
            if total_training_time_secs
            else None
        )
        avg_gpu_power_watts = (
            sum(gpu_stats_during["power"]) / len(gpu_stats_during["power"])
            if gpu_stats_during["power"]
            else None
        )
        tokens_per_sec_per_watt_global = (
            training_throughput_tokens_per_sec_global / avg_gpu_power_watts
            if training_throughput_tokens_per_sec_global and avg_gpu_power_watts
            else None
        )

        save_summary_stats_json(
            summary={
                "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
                "total_gpus": world_size,
                "model": model_name,
                "dataset": data,
                "framework": "accelerate",
                "parallelism_type": "fsdp",
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "trainable_parameters": trainable_params,
                "total_trainable_parameters": total_params,
                "trainable_parameters_percentage": trainable_pct,
                "learning_rate": training_args.learning_rate,
                "avg_gpu_memory_gb": sum(gpu_stats_during["mem"])
                / len(gpu_stats_during["mem"])
                if gpu_stats_during["mem"]
                else None,
                "peak_gpu_memory_gb": max(gpu_stats_during["mem"])
                if gpu_stats_during["mem"]
                else None,
                "avg_gpu_utilization_percent": sum(gpu_stats_during["util"])
                / len(gpu_stats_during["util"])
                if gpu_stats_during["util"]
                else None,
                "peak_gpu_utilization_percent": max(gpu_stats_during["util"])
                if gpu_stats_during["util"]
                else None,
                "avg_gpu_power_watts": avg_gpu_power_watts,
                "peak_gpu_power_watts": max(gpu_stats_during["power"])
                if gpu_stats_during["power"]
                else None,
                "total_execution_time_hours": total_training_time_secs / 3600,
                "training_throughput_tokens_per_sec": training_throughput_tokens_per_sec_global,
                "training_throughput_tokens_per_sec_global": training_throughput_tokens_per_sec_global,
                "training_throughput_tokens_per_sec_per_gpu": training_throughput_tokens_per_sec_per_gpu,
                "tokens_per_sec_per_watt_global": tokens_per_sec_per_watt_global,
                "samples_per_sec": samples_per_sec,
                "total_tokens_per_gpu_all_epochs": tokens_per_gpu_all_epochs,
                "total_tokens_global_all_epochs": tokens_global_all_epochs,
                "avg_training_loss": avg_training_loss,
                "avg_validation_loss": avg_validation_loss,
                "total_training_time_hours": total_training_time_secs / 3600,
                "avg_epoch_training_time_sec": avg_epoch_time_sec,
                "avg_epoch_training_time_hours": avg_epoch_time_hours,
                "avg_step_training_time_sec": avg_step_time_sec,
                "avg_step_training_time_hours": avg_step_time_hours,
                "avg_gpu_flops": avg_gpu_flops,
                "avg_gpu_mfu": avg_gpu_mfu,
            },
            output_file=os.path.join(output_dir, f"training_summary_{rank}.json"),
        )

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        print_rank(rank, "Fine-tuning completed successfully.")

    except Exception as e:
        save_summary_stats_json(
            summary={
                "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
                "total_gpus": world_size,
                "model": model_name,
                "dataset": data,
                "framework": "accelerate",
                "parallelism_type": "fsdp",
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "trainable_parameters": trainable_params,
                "total_trainable_parameters": total_params,
                "trainable_parameters_percentage": trainable_pct,
                "learning_rate": training_args.learning_rate,
                "error": str(e),
            },
            output_file=os.path.join(output_dir, f"training_summary_{rank}.json"),
        )
        print_rank(rank, "Fine-tuning failed to complete!")
        raise e


if __name__ == "__main__":
    main()
