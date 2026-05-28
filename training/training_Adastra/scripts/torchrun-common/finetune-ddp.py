import os
import sys
import time
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, random_split
from custom_train import CustomTrainer
from gpu_monitor import GPUMonitorCallback, start_gpu_monitor
from nic_monitor import NICMonitorCallback, start_nic_monitor
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


class TokenTrackingTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens_this_gpu = 0
        self.total_tokens_global = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            if "input_ids" in inputs and inputs["input_ids"] is not None:
                tokens_in_batch = int(inputs["input_ids"].numel())
            elif "labels" in inputs and inputs["labels"] is not None:
                tokens_in_batch = int(inputs["labels"].numel())
            else:
                tokens_in_batch = 0
            tokens_in_batch *= max(1, self.args.gradient_accumulation_steps)
            self.total_tokens_this_gpu += tokens_in_batch
        except Exception:
            pass
        return super().training_step(model, inputs)

    def _finalize_token_counts(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local = torch.tensor(self.total_tokens_this_gpu, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
            self.total_tokens_global = int(local.item())
        except Exception:
            self.total_tokens_global = int(self.total_tokens_this_gpu)

    def train(self, *args, **kwargs):
        self.total_tokens_this_gpu = 0
        self.total_tokens_global = None

        start_time = time.time()
        output = super().train(*args, **kwargs)
        end_time = time.time()

        self._finalize_token_counts()
        elapsed = end_time - start_time

        try:
            setattr(self.state, "total_training_seconds_custom", float(elapsed))
            setattr(self.state, "total_tokens_per_gpu_custom", int(self.total_tokens_this_gpu))
            setattr(self.state, "total_tokens_global_custom", int(self.total_tokens_global))
        except Exception:
            pass

        if self.is_world_process_zero():
            print("\n=== TOKEN / THROUGHPUT SUMMARY (TokenTrackingTrainer) ===")
            print(f"Total tokens per GPU (ALL epochs): {self.total_tokens_this_gpu:,}")
            print(f"Total tokens GLOBAL (ALL epochs): {self.total_tokens_global:,}")
            if elapsed > 0:
                print(f"Total training time (s): {elapsed:.2f}")
                print(f"Tokens/sec per GPU: {self.total_tokens_this_gpu / elapsed:,.2f}")
                print(f"Tokens/sec GLOBAL: {self.total_tokens_global / elapsed:,.2f}")
            print("==========================================================\n")

        return output


def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def main():
    if args.dataset not in DATASET_HANDLER_MAP:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    DatasetHandlerClass = DATASET_HANDLER_MAP[args.dataset]

    model_name = args.model
    data = args.data
    output_dir = args.output_dir

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading tokenizer... {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer Loaded")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("GPU_NODE", 1)) * int(os.environ.get("SLURM_NNODES", 1))

    train_path, val_path, is_split = parse_dataset_paths(data)

    print(f"📂 Dataset input type: {'train/val split' if is_split else 'single dataset'}")
    print(f"  Train path: {train_path}")
    if val_path:
        print(f"  Validation path: {val_path}")

    if is_split and val_path:
        train_dataset = DatasetHandlerClass(path=train_path, tokenizer=tokenizer, max_length=MAX_LENGTH)
        eval_dataset = DatasetHandlerClass(path=val_path, tokenizer=tokenizer, max_length=MAX_LENGTH)
        dataset_for_collate = train_dataset
    else:
        dataset = DatasetHandlerClass(path=train_path, tokenizer=tokenizer, max_length=MAX_LENGTH)
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        dataset_for_collate = dataset

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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=args.dataloader_num_workers, pin_memory=True,
                                  collate_fn=collate_fn_train, persistent_workers=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=args.dataloader_num_workers, pin_memory=True,
                                 collate_fn=collate_fn_eval, persistent_workers=True)

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Loading Model... dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map=None, attn_implementation="flash_attention_2", use_cache=False,
    )
    print("Model Loaded")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        eval_strategy="epoch",
        eval_steps=None,
        ddp_timeout=3600,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    training_args.num_train_epochs = args.epochs if args.epochs is not None else 1
    if args.max_steps is not None:
        training_args.max_steps = int(args.max_steps)
        # train_dataset = torch.utils.data.Subset(train_dataset, range(min(450, len(train_dataset))))
        # eval_dataset  = torch.utils.data.Subset(eval_dataset,  range(min(50, len(eval_dataset))))

    n_gpus = int(os.environ.get("GPU_NODE", 1))
    monitor     = GPUMonitorCallback(n_gpus=n_gpus)
    nic_monitor = NICMonitorCallback()

    trainer = TokenTrackingTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[monitor, nic_monitor],
    )

    gpu_stats_during, gpu_stop = start_gpu_monitor(interval_sec=5, n_gpus=n_gpus)
    nic_stats_during, nic_stop = start_nic_monitor(interval_sec=1)

    start_time = time.time()
    trainer.train()
    total_finetune_time = time.time() - start_time

    gpu_stop["stop"] = True
    nic_stop["stop"] = True
    time.sleep(2)

    trainable_params, total_params, trainable_pct = count_parameters(model)

    log_history = trainer.state.log_history
    avg_training_loss = avg_validation_loss = None
    avg_epoch_time_sec = avg_epoch_time_hours = None
    avg_step_time_sec = avg_step_time_hours = None

    total_training_time_secs = getattr(trainer.state, "total_training_seconds_custom", total_finetune_time)
    tokens_per_gpu_all_epochs = getattr(trainer.state, "total_tokens_per_gpu_custom", trainer.total_tokens_this_gpu)
    tokens_global_all_epochs = getattr(trainer.state, "total_tokens_global_custom", trainer.total_tokens_global)

    if training_args.max_steps:
        avg_step_time_sec = total_training_time_secs / training_args.max_steps
        avg_step_time_hours = avg_step_time_sec / 3600
    else:
        avg_epoch_time_sec = total_training_time_secs / training_args.num_train_epochs
        avg_epoch_time_hours = avg_epoch_time_sec / 3600
        avg_step_time_sec = avg_epoch_time_sec / len(train_dataloader)
        avg_step_time_hours = avg_step_time_sec / 3600

    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    samples_per_sec = effective_batch_size / avg_step_time_sec if avg_step_time_sec else None
    training_throughput_tokens_per_sec_per_gpu = tokens_per_gpu_all_epochs / total_training_time_secs if total_training_time_secs else None
    training_throughput_tokens_per_sec_global = tokens_global_all_epochs / total_training_time_secs if total_training_time_secs else None
    avg_gpu_power_watts = sum(gpu_stats_during["power"]) / len(gpu_stats_during["power"]) if gpu_stats_during["power"] else None
    tokens_per_sec_per_watt_global = training_throughput_tokens_per_sec_global / avg_gpu_power_watts if training_throughput_tokens_per_sec_global and avg_gpu_power_watts else None

    def safe_avg(lst): return sum(lst) / len(lst) if lst else None
    def safe_max(lst): return max(lst) if lst else None

    save_summary_stats_json(
        summary={
            "nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "num_gpus_per_node": n_gpus,
            "total_gpus": world_size,
            "model": model_name,
            "dataset": data,
            "framework": "torchrun",
            "parallelism_type": "ddp",
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "trainable_parameters": trainable_params,
            "total_trainable_parameters": total_params,
            "trainable_parameters_percentage": trainable_pct,
            "learning_rate": training_args.learning_rate,
            # GPU
            "avg_gpu_memory_gb":            safe_avg(gpu_stats_during["mem"]),
            "peak_gpu_memory_gb":           safe_max(gpu_stats_during["mem"]),
            "avg_gpu_utilization_percent":  safe_avg(gpu_stats_during["util"]),
            "peak_gpu_utilization_percent": safe_max(gpu_stats_during["util"]),
            "avg_gpu_power_watts":          avg_gpu_power_watts,
            "peak_gpu_power_watts":         safe_max(gpu_stats_during["power"]),
            # NIC
            "avg_nic_rx_MBps":  safe_avg(nic_stats_during["rx_MBps"]),
            "peak_nic_rx_MBps": safe_max(nic_stats_during["rx_MBps"]),
            "avg_nic_tx_MBps":  safe_avg(nic_stats_during["tx_MBps"]),
            "peak_nic_tx_MBps": safe_max(nic_stats_during["tx_MBps"]),
            "avg_nic_rx_Mpps":  safe_avg(nic_stats_during["rx_Mpps"]),
            "peak_nic_rx_Mpps": safe_max(nic_stats_during["rx_Mpps"]),
            "avg_nic_tx_Mpps":  safe_avg(nic_stats_during["tx_Mpps"]),
            "peak_nic_tx_Mpps": safe_max(nic_stats_during["tx_Mpps"]),
            # Timing & throughput
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
        },
        output_file=os.path.join(output_dir, f"training_summary_{rank}.json"),
    )

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()