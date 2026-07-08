# finetune_llama8b.py
import os
import sys
import time

import torch
from custom_train import CustomTrainer
from gpu_monitor import GPUMonitorCallback, start_gpu_monitor
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from utils import (
    count_parameters,
    parse_args,
    parse_dataset_paths,
    save_summary_stats_json,
)

args = parse_args()
sys.path.append(os.path.join(args.minerva_dir, "..", ".."))
from training.training_MN5.configs.config_datasets_handlers_map import (
    DATASET_HANDLER_MAP,
)

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size

def is_main_process():
    # HF/torchrun sets LOCAL_RANK env var; fallback to RANK
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset not in DATASET_HANDLER_MAP:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # Get Dataset Handler
    DatasetHandlerClass = DATASET_HANDLER_MAP[args.dataset]

    model_name = args.model
    data = args.data
    output_dir = args.output_dir

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer Loaded")

    # ---------------------------------------------------------------------
    # Handle dataset path (string or dict)
    # ---------------------------------------------------------------------
    train_path, val_path, is_split = parse_dataset_paths(data)

    print(
        f"📂 Dataset input type: {'train/val split' if is_split else 'single dataset'}"
    )
    print(f"  Train path: {train_path}")
    if val_path:
        print(f"  Validation path: {val_path}")

    # If we have explicit train/validation files
    if is_split and val_path:
        print("Loading train and validation datasets separately...")
        train_dataset = DatasetHandlerClass(
            path=train_path, tokenizer=tokenizer, max_length=MAX_LENGTH
        )
        dataset = train_dataset
        eval_dataset = DatasetHandlerClass(
            path=val_path, tokenizer=tokenizer, max_length=MAX_LENGTH
        )
        dataset_for_collate = train_dataset  # use train dataset for collate_fn lookup
        print(
            f"Loaded {len(train_dataset)} training and {len(eval_dataset)} validation samples."
        )

    else:
        # Single dataset — perform 90/10 random split
        print("Single dataset detected — applying 90/10 train/val split.")
        dataset = DatasetHandlerClass(
            path=train_path, tokenizer=tokenizer, max_length=MAX_LENGTH
        )
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        dataset_for_collate = dataset  # use full dataset for collate_fn lookup
        print(
            f"Split dataset into {len(train_dataset)} train and {len(eval_dataset)} eval samples."
        )

    def resolve_collate(ds_obj, fallback):
        if hasattr(ds_obj, "collate_fn"):
            return getattr(ds_obj, "collate_fn")
        if hasattr(ds_obj, "dataset") and hasattr(ds_obj.dataset, "collate_fn"):
            return getattr(ds_obj.dataset, "collate_fn")
        if fallback is not None and hasattr(fallback, "collate_fn"):
            return getattr(fallback, "collate_fn")
        return None

    collate_fn_train = resolve_collate(train_dataset, dataset_for_collate)
    collate_fn_eval = resolve_collate(eval_dataset, dataset_for_collate)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.dataloader_num_workers,  # parallel data loading
        pin_memory=True,  # faster CPU→GPU transfer
        collate_fn=collate_fn_train,  # your custom padding function
        persistent_workers=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn_eval,
        persistent_workers=True,
    )

    # # Dataset
    # dataset = DatasetHandlerClass(path=data, tokenizer=tokenizer, max_length=MAX_LENGTH)

    # # Split into 90% train / 10% eval
    # train_size = int(0.9 * len(dataset))
    # eval_size = len(dataset) - train_size
    # train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     collate_fn=dataset.collate_fn,
    # )
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=dataset.collate_fn,
    # )

    # Model
    # --- Precision selection ---
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Loading Model... dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    print("Model Loaded")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
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
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=None,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    # Conditionally add either epochs or max_steps
    training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
    if args.max_steps is not None:
        training_args.max_steps = int(args.max_steps)

    monitor = GPUMonitorCallback(n_gpus=int(os.environ.get("GPU_NODE", 1)))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[monitor],
    )

    # Start GPU monitor
    gpu_stats_during, stop_flag = start_gpu_monitor(
        interval_sec=5, n_gpus=int(os.environ.get("GPU_NODE", 1))
    )

    # Train Model
    start_time = time.time()
    trainer.train()
    total_finetune_time = time.time() - start_time

    # Stop GPU monitor
    stop_flag["stop"] = True
    time.sleep(2)  # give it a moment to exit cleanly

    # Get params
    trainable_params, total_params, trainable_pct = count_parameters(model)

    # Get Average Training and Validation Loss
    log_history = trainer.state.log_history
    avg_training_loss = None
    avg_validation_loss = None
    avg_epoch_time_sec, avg_epoch_time_hours = None, None
    avg_step_time_sec, avg_step_time_hours = None, None

    if log_history:
        # Average training loss over last logged steps
        train_losses = [x["loss"] for x in log_history if "loss" in x]
        avg_training_loss = (
            sum(train_losses) / len(train_losses) if train_losses else None
        )

        eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
        avg_validation_loss = (
            sum(eval_losses) / len(eval_losses) if eval_losses else None
        )
        # Extract all train_runtime entries (only 1, that is the total training time).
        epoch_times = [
            log["train_runtime"] for log in log_history if "train_runtime" in log
        ]
        total_training_time_secs = sum(epoch_times) if epoch_times else None

        # Compute average runtime per epoch (seconds)
        # Determine if training is step-based or epoch-based
        if args.max_steps is not None and args.max_steps > 0:
            avg_step_time_sec = total_training_time_secs / args.max_steps
            avg_step_time_hours = avg_step_time_sec / 3600
        else:
            avg_epoch_time_sec = total_training_time_secs / args.epochs
            avg_epoch_time_hours = avg_epoch_time_sec / 3600

    save_summary_stats_json(
        summary={
            "nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
            "total_gpus": int(os.environ.get("SLURM_NNODES", 1))
            * int(os.environ.get("GPU_NODE", 1)),
            "model": model_name,
            "dataset": data,
            "framework": "torchrun",
            "parallelism_type": "none",
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "trainable_parameters": trainable_params,
            "total_trainable_parameters": total_params,
            "trainable_parameters_percentage": trainable_pct,
            "learning_rate": training_args.learning_rate,
            "avg_gpu_memory_gb": sum(gpu_stats_during["mem"])
            / len(gpu_stats_during["mem"]),
            "peak_gpu_memory_gb": max(gpu_stats_during["mem"]),
            "avg_gpu_utilization_percent": sum(gpu_stats_during["util"])
            / len(gpu_stats_during["util"]),
            "peak_gpu_utilization_percent": max(gpu_stats_during["util"]),
            "avg_gpu_power_watts": sum(gpu_stats_during["power"])
            / len(gpu_stats_during["power"]),
            "peak_gpu_power_watts": max(gpu_stats_during["power"]),
            "total_execution_time_hours": total_training_time_secs / 3600,
            "training_throughput_tokens_per_sec": (len(dataset) * MAX_LENGTH)
            / total_training_time_secs,
            "avg_training_loss": avg_training_loss,
            "avg_validation_loss": avg_validation_loss,
            "total_training_time_hours": total_training_time_secs / 3600,
            "avg_epoch_training_time_sec": avg_epoch_time_sec,
            "avg_epoch_training_time_hours": avg_epoch_time_hours,
            "avg_step_training_time_sec": avg_step_time_sec,
            "avg_step_training_time_hours": avg_step_time_hours,
        },
        output_file=os.path.join(output_dir, "training_summary_0.json"),
    )

    print("Single-GPU Fine-tuning complete.")


if __name__ == "__main__":
    main()
