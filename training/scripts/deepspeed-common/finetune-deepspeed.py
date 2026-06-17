import os
import time

import psutil
import torch
import torch.distributed as dist
from gpu_monitor import GPUMonitorCallback, start_gpu_monitor
from shared.custom_train import PerformanceTrackingTrainer
from shared.data import load_dataset
from shared.utils import count_parameters, print_rank, save_summary_stats_json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from utils import parse_args

args = parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size


def is_main_process():
    # HF/torchrun sets LOCAL_RANK env var; fallback to RANK
    rank = int(os.environ["RANK"])
    return rank == 0


# --- Main ---
def main():
    # Get rank
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # world_size = int(os.environ.get("GPU_NODE", 1)) * int(
        #    os.environ.get("SLURM_NNODES", 1)
        # )
    print(f"Rank {rank}/{world_size} started...")
    zero_stage = os.environ["PARALLELISM"]

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

    # ---------------------------------------------------------------------
    # Handle dataset path (string or dict)
    # ---------------------------------------------------------------------
    train_dataset, eval_dataset, collate_fn, _ = load_dataset(
        dataset_name=args.dataset,
        dataset_path=args.data,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Model
    # --- Precision selection ---
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    trainable_params, total_params, trainable_pct = 0, 0, 0
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # effective batch size
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        # warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_total_limit=1,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        optim="adamw_torch",
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        eval_steps=None,
        # Dataloader is created automatically from trainer
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        data_seed=32,
        dataloader_persistent_workers=args.dataloader_num_workers > 1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        # Deepspeed Config
        deepspeed=args.deepspeed_config_file,
        # torch model compilation
    )
    if bool(args.enable_compile):
        training_args.torch_compile = True
        training_args.torch_compile_backend = "inductor"
        training_args.torch_compile_mode = "max-autotune-no-cudagraphs"
    try:
        # train_dataloader = DataLoader(
        #    train_dataset,
        #    batch_size=BATCH_SIZE,
        #    shuffle=True,
        #    num_workers=args.dataloader_num_workers,
        #    pin_memory=True,
        #    collate_fn=collate_fn_train,
        #    persistent_workers=True,
        # )
        # eval_dataloader = DataLoader(
        #    eval_dataset,
        #    batch_size=BATCH_SIZE,
        #    shuffle=False,
        #    num_workers=args.dataloader_num_workers,
        #    pin_memory=True,
        #    collate_fn=collate_fn_eval,
        #    persistent_workers=True,
        # )

        training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
        if args.max_steps is not None:
            training_args.max_steps = int(args.max_steps)

        monitor = GPUMonitorCallback(n_gpus=int(os.environ.get("GPU_NODE", 1)))

        # Peak GPU TFLOPs for MFU (bf16/fp16 tensor core peak).
        # Set PEAK_GPU_TFLOPS env var for your hardware, e.g.:
        #   A100 SXM4 80GB = 312, H100 SXM5 = 989, MI250X = 383, MI300X = 1307

        _peak_gpu_tflops = os.environ.get("GPU_PEAK_TFLOPS")
        peak_gpu_tflops = float(_peak_gpu_tflops) if _peak_gpu_tflops else None
        gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
        print_rank(
            0,
            f"GPU_NAME: {gpu_name} | Using peak GPU TFLOPS for MFU calculation: {peak_gpu_tflops} TFLOPS",
        )
        # model_config = AutoConfig.from_pretrained(model_name)

        print_rank(f"Loading Model... dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print_rank(
            int(os.environ["RANK"]), f"CPU RAM after model load: {ram_gb:.1f} GB"
        )

        if hasattr(model.config, "use_cache"):
            print_rank(0, "Disabling model's cache")
            model.config.use_cache = False
        print_rank("Model Loaded")

        print_rank(0, ":::::::::")
        for i in model.named_parameters():
            print_rank(0, f"{i[0]} -> {i[1].device}")
        print_rank(0, ":::::::::")

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print_rank(rank, "Gradient checkpointing enabled!")

        trainer = PerformanceTrackingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            tokenizer=tokenizer,
            callbacks=[monitor],
            peak_gpu_tflops=peak_gpu_tflops,
        )

        print_rank(rank, "Trainer initialized and model has been wrapped!")
        print_rank(rank, ":::::::::")
        for i in model.named_parameters():
            print_rank(rank, f"{i[0]} -> {i[1].device}")
        print_rank(rank, ":::::::::")

        # Start GPU monitor
        gpu_stats_during, stop_flag = start_gpu_monitor(
            interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
        )

        # Train Model
        start_time = time.time()
        trainer.train()
        total_finetune_time = time.time() - start_time

        # Stop GPU monitor
        stop_flag["stop"] = True
        time.sleep(2)  # give it a moment to exit cleanly

        trainable_params, total_params, trainable_pct = count_parameters(model)

        # ---- Compute metrics ----
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
        # global_avg_gpu_flops = getattr(trainer.state, "global_average_flops_custom")
        avg_gpu_mfu = getattr(trainer.state, "average_mfu_custom")
        # global_avg_gpu_mfu = getattr(trainer.state, "global_average_mfu_custom")

        if training_args.max_steps:
            avg_step_time_sec = total_training_time_secs / training_args.max_steps
            avg_step_time_hours = avg_step_time_sec / 3600
        else:
            avg_epoch_time_sec = (
                total_training_time_secs / training_args.num_train_epochs
            )
            avg_epoch_time_hours = avg_epoch_time_sec / 3600
            avg_step_time_sec = avg_epoch_time_sec / len(train_dataset)  # approximate
            avg_step_time_hours = avg_step_time_sec / 3600

        # ---- Compute derived metrics ----
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
                "framework": "deepspeed",
                "parallelism_type": zero_stage,
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
        print("Fine-tuning completed successfully.")
    except Exception as e:
        save_summary_stats_json(
            summary={
                "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
                "model": model_name,
                "dataset": data,
                "framework": "deepspeed",
                "parallelism_type": zero_stage,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "trainable_parameters": trainable_params,
                "total_trainable_parameters": total_params,
                "trainable_parameters_percentage": trainable_pct,
                "learning_rate": training_args.learning_rate,
                "error": str(e),
            },
            output_file=os.path.join(output_dir, "training_summary_0.json"),
        )

        print("Fine-tuning failed to complete!")
        raise e


if __name__ == "__main__":
    main()
