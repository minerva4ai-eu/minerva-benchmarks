import os
import time

import deepspeed
import psutil
import torch
import torch.distributed as dist
from gpu_monitor import start_gpu_monitor
from shared.data import load_dataset
from shared.utils import count_parameters, print_rank, save_summary_stats_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_args

args = parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_main_process():
    return int(os.environ["RANK"]) == 0


def get_dist_info():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def reduce_tensor(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar tensor and return the global sum."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ---------------------------------------------------------------------------
# MFU helpers
# ---------------------------------------------------------------------------


def estimate_flops_per_token(model: torch.nn.Module) -> float:
    """
    Roughly 6 * num_params FLOPs per token for a dense transformer
    (forward + backward = 2× forward; forward ≈ 2 * P multiply-adds → 6P total).
    Returns FLOPs as a float.
    """
    num_params = sum(p.numel() for p in model.parameters())
    return 6.0 * num_params


def compute_mfu(
    flops_per_token: float,
    tokens_per_step: int,
    step_time_sec: float,
    peak_tflops: float,
) -> float:
    """Model FLOPs Utilisation in [0, 1]."""
    if step_time_sec <= 0 or peak_tflops is None:
        return 0.0
    achieved_tflops = (flops_per_token * tokens_per_step) / step_time_sec / 1e12
    return achieved_tflops / peak_tflops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rank, world_size = get_dist_info()
    print(f"Rank {rank}/{world_size} started...")

    zero_stage = os.environ["PARALLELISM"]
    model_name = args.model
    data = args.data
    output_dir = args.output_dir

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print_rank(rank, f"Loading tokenizer... {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank(rank, "Tokenizer loaded")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset, eval_dataset, collate_fn, _ = load_dataset(
        dataset_name=args.dataset,
        dataset_path=args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # ------------------------------------------------------------------
    # Distributed sampler + DataLoader
    # ------------------------------------------------------------------
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=32,
        drop_last=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.dataloader_num_workers > 1,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print_rank(rank, f"Loading model... dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print_rank(rank, f"CPU RAM after model load: {ram_gb:.1f} GB")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        print_rank(rank, "Disabled model cache")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank(rank, "Gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # DeepSpeed initialisation
    # The DS config file owns: optimizer, scheduler, ZeRO stage, precision.
    # We do NOT build an optimizer here — DS reads it from the JSON.
    # Works identically for ZeRO-1, ZeRO-2, and ZeRO-3.
    # ------------------------------------------------------------------

    with open(args.deepspeed_config_file, "r") as f:
        import json

        ds_config = json.load(f)
    ds_config["bf16"] = {"enabled": args.precision == "bf16"}
    ds_config["fp16"] = {"enabled": args.precision == "fp16"}
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    
    print(f"DeepSpeed Config: \n{ds_config}")

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    print_rank(rank, "DeepSpeed engine initialised")

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    num_epochs: int = args.epochs if args.epochs is not None else 1
    max_steps: int = int(args.max_steps) if args.max_steps is not None else None
    grad_accum_steps: int = args.gradient_accumulation_steps
    logging_steps: int = args.logging_steps

    # FLOPs / MFU setup
    _peak_tflops_env = os.environ.get("GPU_PEAK_TFLOPS")
    peak_gpu_tflops = float(_peak_tflops_env) if _peak_tflops_env else None
    gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
    flops_per_token = estimate_flops_per_token(model)
    print_rank(
        rank,
        f"GPU_NAME: {gpu_name} | Peak TFLOPS for MFU: {peak_gpu_tflops} | "
        f"FLOPs/token: {flops_per_token:.3e}",
    )

    # ------------------------------------------------------------------
    # GPU background monitor
    # ------------------------------------------------------------------
    gpu_stats_during, stop_flag = start_gpu_monitor(
        interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
    )

    # ------------------------------------------------------------------
    # Metric accumulators
    # ------------------------------------------------------------------
    total_loss_sum: float = 0.0
    total_loss_steps: int = 0

    tokens_per_gpu_all_epochs: int = 0  # tokens seen by this rank
    tokens_global_all_epochs: int = 0  # sum across all ranks

    step_flops_list: list[float] = []  # per-step achieved TFLOPs
    step_mfu_list: list[float] = []  # per-step MFU

    global_step: int = 0  # optimizer steps taken
    total_training_time_secs: float = 0.0

    trainable_params, total_params, trainable_pct = count_parameters(model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print_rank(rank, "Starting training...")
    train_start = time.time()

    training_done = False
    for epoch in range(num_epochs):
        if training_done:
            break

        train_sampler.set_epoch(epoch)
        engine.train()

        micro_step = 0  # counts every forward/backward call
        accum_loss: float = 0.0
        accum_tokens_local: int = 0

        step_start = time.time()

        for batch in train_loader:
            # ---------- move batch to device ----------
            batch = {
                k: v.to(engine.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # ---------- token counting ---------------
            # input_ids shape: (B, T); drop padding tokens
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                local_tokens = int(attention_mask.sum().item())
            else:
                local_tokens = input_ids.numel()
            accum_tokens_local += local_tokens

            # ---------- forward + backward -----------
            # engine.backward() handles ZeRO gradient sharding correctly
            # for all stages; never call loss.backward() directly.
            outputs = engine(**batch)
            loss = outputs.loss / grad_accum_steps
            engine.backward(loss)
            accum_loss += loss.item()

            micro_step += 1

            # ---------- optimizer step ---------------
            if micro_step % grad_accum_steps == 0:
                step_time = time.time() - step_start

                # All-reduce token count to get global tokens this step
                tokens_tensor = torch.tensor(
                    accum_tokens_local, dtype=torch.long, device=engine.device
                )
                reduce_tensor(tokens_tensor, world_size)
                global_tokens_this_step = int(tokens_tensor.item())

                # engine.step() also clips gradients (configured in DS JSON)
                engine.step()

                # ------ MFU / FLOPs ------------------
                achieved_flops = (
                    flops_per_token * global_tokens_this_step / step_time / 1e12
                    if step_time > 0
                    else 0.0
                )
                mfu = (
                    compute_mfu(
                        flops_per_token,
                        global_tokens_this_step,
                        step_time,
                        peak_gpu_tflops,
                    )
                    if peak_gpu_tflops
                    else 0.0
                )
                step_flops_list.append(achieved_flops)
                step_mfu_list.append(mfu)

                # ------ accumulate global metrics ----
                tokens_per_gpu_all_epochs += accum_tokens_local
                tokens_global_all_epochs += global_tokens_this_step

                # ------ loss logging -----------------
                # accum_loss was already divided by grad_accum_steps each step;
                # multiply back so we log the mean of the original micro-losses.
                step_loss = accum_loss * grad_accum_steps
                total_loss_sum += step_loss
                total_loss_steps += 1

                if is_main_process() and global_step % logging_steps == 0:
                    print(
                        f"[Epoch {epoch + 1} | Step {global_step}] "
                        f"loss={step_loss:.4f}  "
                        f"tokens_global={global_tokens_this_step}  "
                        f"TFLOPs={achieved_flops:.2f}  "
                        f"MFU={mfu * 100:.1f}%  "
                        f"step_time={step_time:.2f}s"
                    )

                # ------ reset micro accumulators -----
                accum_loss = 0.0
                accum_tokens_local = 0
                step_start = time.time()

                global_step += 1

                # ------ max_steps guard --------------
                if max_steps is not None and global_step >= max_steps:
                    training_done = True
                    break

        # end of epoch — handle leftover micro-steps (partial accumulation)
        # If micro_step % grad_accum_steps != 0 there are un-stepped gradients;
        # we flush them so the last partial batch isn't silently dropped.
        if not training_done and micro_step % grad_accum_steps != 0:
            step_time = time.time() - step_start
            tokens_tensor = torch.tensor(
                accum_tokens_local, dtype=torch.long, device=engine.device
            )
            reduce_tensor(tokens_tensor, world_size)
            global_tokens_this_step = int(tokens_tensor.item())

            engine.step()

            tokens_per_gpu_all_epochs += accum_tokens_local
            tokens_global_all_epochs += global_tokens_this_step

            step_loss = accum_loss * grad_accum_steps
            total_loss_sum += step_loss
            total_loss_steps += 1
            global_step += 1

    total_training_time_secs = time.time() - train_start

    # ------------------------------------------------------------------
    # Stop GPU monitor
    # ------------------------------------------------------------------
    stop_flag["stop"] = True
    time.sleep(2)

    # ------------------------------------------------------------------
    # Aggregate final metrics
    # ------------------------------------------------------------------
    avg_training_loss = total_loss_sum / total_loss_steps if total_loss_steps else None

    avg_gpu_flops = (
        sum(step_flops_list) / len(step_flops_list) if step_flops_list else None
    )
    avg_gpu_mfu = sum(step_mfu_list) / len(step_mfu_list) if step_mfu_list else None

    effective_batch_size = args.batch_size * grad_accum_steps * world_size

    avg_step_time_sec = total_training_time_secs / global_step if global_step else None
    avg_step_time_hours = avg_step_time_sec / 3600 if avg_step_time_sec else None

    if max_steps is None:
        avg_epoch_time_sec = total_training_time_secs / num_epochs
        avg_epoch_time_hours = avg_epoch_time_sec / 3600
    else:
        avg_epoch_time_sec = None
        avg_epoch_time_hours = None

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

    # ------------------------------------------------------------------
    # Save summary — identical schema to the original
    # ------------------------------------------------------------------
    save_summary_stats_json(
        summary={
            "nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "num_gpus_per_node": int(os.environ.get("GPU_NODE", 1)),
            "total_gpus": world_size,
            "model": model_name,
            "dataset": data,
            "framework": "deepspeed",
            "parallelism_type": zero_stage,
            "batch_size": args.batch_size,
            "gradient_accumulation": grad_accum_steps,
            "trainable_parameters": trainable_params,
            "total_trainable_parameters": total_params,
            "trainable_parameters_percentage": trainable_pct,
            "learning_rate": args.lr,
            "avg_gpu_memory_gb": (
                sum(gpu_stats_during["mem"]) / len(gpu_stats_during["mem"])
                if gpu_stats_during["mem"]
                else None
            ),
            "peak_gpu_memory_gb": (
                max(gpu_stats_during["mem"]) if gpu_stats_during["mem"] else None
            ),
            "avg_gpu_utilization_percent": (
                sum(gpu_stats_during["util"]) / len(gpu_stats_during["util"])
                if gpu_stats_during["util"]
                else None
            ),
            "peak_gpu_utilization_percent": (
                max(gpu_stats_during["util"]) if gpu_stats_during["util"] else None
            ),
            "avg_gpu_power_watts": avg_gpu_power_watts,
            "peak_gpu_power_watts": (
                max(gpu_stats_during["power"]) if gpu_stats_during["power"] else None
            ),
            "total_execution_time_hours": total_training_time_secs / 3600,
            "training_throughput_tokens_per_sec": training_throughput_tokens_per_sec_global,
            "training_throughput_tokens_per_sec_global": training_throughput_tokens_per_sec_global,
            "training_throughput_tokens_per_sec_per_gpu": training_throughput_tokens_per_sec_per_gpu,
            "tokens_per_sec_per_watt_global": tokens_per_sec_per_watt_global,
            "samples_per_sec": samples_per_sec,
            "total_tokens_per_gpu_all_epochs": tokens_per_gpu_all_epochs,
            "total_tokens_global_all_epochs": tokens_global_all_epochs,
            "avg_training_loss": avg_training_loss,
            "avg_validation_loss": None,  # no eval loop; add if needed
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

    print_rank(rank, "Fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
