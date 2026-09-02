import math
import os
import time

import deepspeed
import psutil
import torch
import torch.distributed as dist
from shared.args import get_deepspeed_parser
from shared.data import collate_fn, get_train_eval_path, load_prepared_packed_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import (
    is_main_process,
    print_rank,
    save_training_summary,
    setup_distributed,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

args = get_deepspeed_parser().parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_dist_info():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def reduce_tensor(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar tensor and return the global sum."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def load_model(model_path, dtype, ds_config):
    # from deepspeed.runtime.zero.stage3 import (
    #    estimate_zero3_model_states_mem_needs_all_live,
    # )

    ds_hf_config = HfDeepSpeedConfig(ds_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    # with deepspeed.zero.Init(
    #    config_dict_or_path=ds_config,
    #    remote_device="cpu",
    # ):
    #    model = AutoModelForCausalLM.from_pretrained(
    #        model_path,
    #        torch_dtype=dtype,
    #        ignore_mismatched_sizes=True,
    #        low_cpu_mem_usage=True,
    #    )
    # estimate_zero3_model_states_mem_needs_all_live(
    #    model=model,
    #    num_gpus_per_node=dist.get_world_size(),
    #    num_nodes=int(os.environ["SLURM_NNODES"]),
    # )
    return model


# ------#
# Main  #
# ------#


def main():
    model_path = args.model
    model_name = args.model.split("/")[-1]
    train_path, eval_path = get_train_eval_path(args)

    rank, world_size, local_rank = setup_distributed()
    torch.cuda.empty_cache()

    output_dir = args.output_dir
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.precision, torch.float32
    )
    print_rank(0, f"Training dtype: {dtype}")

    # -----------------------#
    # Load DeepSpeed Config  #
    # -----------------------#
    with open(args.deepspeed_config_file, "r") as f:
        import json

        ds_config = json.load(f)
    ds_config["bf16"] = {"enabled": args.precision == "bf16"}
    ds_config["fp16"] = {"enabled": args.precision == "fp16"}
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # -----------#
    # Tokenizer  #
    # -----------#
    print_rank(rank, f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------#
    # Dataset  #
    # ---------#
    train_dataset = load_prepared_packed_dataset(train_path)
    # eval_dataset = load_prepared_packed_dataset(eval_path)

    print_rank(
        rank,
        f"Packed train dataset size: {len(train_dataset)} blocks of {args.max_length} tokens",
    )

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=32
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 1,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
    )

    # -------#
    # Model  #
    # -------#

    print_rank(0, f"Loading model '{model_path}' dtype={dtype}")
    model = load_model(model_path, dtype, ds_config)

    ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print_rank(rank, f"CPU RAM after model load: {ram_gb:.1f} GB")

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        print_rank(rank, "Disabled model cache")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank(rank, "Gradient checkpointing enabled")

    if args.enable_compile:
        print_rank(0, "torch.compile enabled.")
        model = torch.compile(model, backend="inductor", mode="default")
        print_rank(0, "Model compilation finished.")

    # -----------------#
    # Training config  #
    # -----------------#
    grad_accum_steps: int = args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)

    logging_steps: int = args.logging_steps
    num_epochs = args.epochs if args.epochs is not None and args.epochs > 0 else 1
    total_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else steps_per_epoch * num_epochs
    )

    peak_gpu_tflops = (
        float(os.environ["GPU_PEAK_TFLOPS"])
        if os.environ.get("GPU_PEAK_TFLOPS")
        else None
    )
    gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
    print_rank(0, f"GPU: {gpu_name} | peak TFLOPs for MFU: {peak_gpu_tflops}")

    # ########################################################################
    # DeepSpeed initialisation                                               #
    # The DS config file owns: optimizer, scheduler, ZeRO stage, precision.  #
    # We do NOT build an optimizer here — DS reads it from the JSON.         #
    # Works identically for ZeRO-1, ZeRO-2, and ZeRO-3.                      #
    ##########################################################################

    print(f"DeepSpeed Config: \n{ds_config}")

    ########################
    # Get DeepSpeed Engine #
    ########################
    engine, _, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    print_rank(rank, "DeepSpeed engine initialised")

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

    global_step: int = 0  # optimizer steps taken
    total_training_time_secs: float = 0.0

    # trainable_params, total_params, trainable_pct = count_parameters(model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print_rank(0, "Starting training...")
    engine.train()
    train_start = time.time()

    flopsCallback_megatronLM = mfu_callback_from_hf_config(
        AutoConfig.from_pretrained(model_path),
        tokenizer,
        gpu_peak_flops=peak_gpu_tflops,
        seq_length=args.max_length,
        trainer_callback=False,
    )
    training_done = False
    step_loss = 0
    for epoch in range(num_epochs):
        if training_done:
            break

        train_sampler.set_epoch(epoch)

        micro_step = 0  # counts every forward/backward call
        accum_loss: float = 0.0
        accum_tokens_local: int = 0

        if global_step == 0:
            flopsCallback_megatronLM.on_step_begin()
        step_start = time.time()

        for micro_step, batch in enumerate(train_dataloader):
            # ---------- move batch to device ----------
            batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}

            # ---------- forward + backward -----------
            # engine.backward() handles ZeRO gradient sharding correctly
            # for all stages; never call loss.backward() directly.
            outputs = engine(**batch)
            engine.backward(outputs.loss)

            accum_tokens_local += batch["input_ids"].numel()
            accum_loss += outputs.loss.item()

            micro_step += 1

            # ---------- optimizer step ---------------
            if micro_step % grad_accum_steps == 0:
                step_time = time.time() - step_start

                # engine.step() also clips gradients (configured in DS JSON)
                engine.step()
                torch.cuda.empty_cache()

                flopsCallback_megatronLM.on_step_end(
                    micro_batch_size=args.batch_size,
                    world_size=world_size,
                    gradient_accumulation_steps=grad_accum_steps,
                    global_step=global_step,
                )

                if global_step % args.logging_steps == 0:
                    print_rank(
                        0,
                        f"epoch {epoch} step {global_step}/{total_steps} | "
                        f"loss {outputs.loss.item():.4f} | "
                        f"lr {lr_scheduler.get_last_lr()[0]:.5e} | "
                        f"TFLOPs/s/GPU {flopsCallback_megatronLM.state.tflops_this_gpu[-1]:.2f} | "
                        f"MFU {flopsCallback_megatronLM.state.mfu_this_gpu[-1]} | "
                        f"gpu_mem_alloc {torch.cuda.memory_allocated() / 1e9:.2f}GB",
                    )
                # All-reduce token count to get global tokens this step
                tokens_tensor = torch.tensor(
                    accum_tokens_local, dtype=torch.float32, device=engine.device
                )

                # ------ accumulate global metrics ----
                tokens_per_gpu_all_epochs += accum_tokens_local

                # ------ loss logging -----------------
                # accum_loss was already divided by grad_accum_steps each step;
                # multiply back so we log the mean of the original micro-losses.
                step_loss = accum_loss * grad_accum_steps
                total_loss_sum += step_loss
                total_loss_steps += 1

                # ------ reset micro accumulators -----
                accum_loss = 0.0
                accum_tokens_local = 0
                step_start = time.time()

                global_step += 1

                # ------ max_steps guard --------------
                if args.max_steps is not None and global_step >= args.max_steps:
                    training_done = True
                    break

                flopsCallback_megatronLM.on_step_begin()
        # end of epoch — handle leftover micro-steps (partial accumulation)
        # If micro_step % grad_accum_steps != 0 there are un-stepped gradients;
        # we flush them so the last partial batch isn't silently dropped.
        if not training_done and micro_step % grad_accum_steps != 0:
            step_time = time.time() - step_start

            engine.step()

            tokens_per_gpu_all_epochs += accum_tokens_local

            step_loss = accum_loss * grad_accum_steps
            total_loss_sum += step_loss
            total_loss_steps += 1
            global_step += 1

    elapsed_total = time.time() - train_start

    # ------------------------------------------------------------------
    # Stop GPU monitor
    # ------------------------------------------------------------------
    stop_flag["stop"] = True
    time.sleep(2)

    # ------------------------#
    # Aggregate final metrics #
    # ------------------------#

    # All-reduce token count to get global tokens this step
    tokens_tensor = torch.tensor(
        tokens_per_gpu_all_epochs, dtype=torch.long, device=engine.device
    )
    reduce_tensor(tokens_tensor, world_size)
    total_tokens_global = int(tokens_tensor.item())
    tokens_tensor = torch.tensor(
        tokens_per_gpu_all_epochs, dtype=torch.long, device=engine.device
    )
    step_loss_tensor = torch.tensor(step_loss, dtype=torch.long, device=engine.device)
    reduce_tensor(step_loss_tensor, world_size)
    final_step_loss = int(step_loss_tensor.item()) / dist.get_world_size()

    avg_mfu = (
        sum(flopsCallback_megatronLM.state.mfu_this_gpu)
        / len(flopsCallback_megatronLM.state.mfu_this_gpu)
        if flopsCallback_megatronLM.state.mfu_this_gpu
        else None
    )
    avg_tflops = (
        sum(flopsCallback_megatronLM.state.tflops_this_gpu)
        / len(flopsCallback_megatronLM.state.tflops_this_gpu)
        if flopsCallback_megatronLM.state.tflops_this_gpu
        else None
    )
    summary_log = [
        f"* Total training time (s): {elapsed_total:.2f}",
        f"* Tokens/sec global: {total_tokens_global / elapsed_total:.2f}",
        f"* Avg TFLOPs/s/GPU: {avg_tflops}",
        f"* Avg MFU: {avg_mfu}",
    ]

    for i in range(world_size):
        print_rank(i, "=== TRAINING SUMMARY ===")
        print_rank(i, "\n".join(summary_log))

    # ------------------------------------------------------------------
    # Save summary — identical schema to the original
    # ------------------------------------------------------------------
    save_training_summary(
        output_dir=output_dir,
        rank=rank,
        model_name=model_name,
        dataset_name=args.dataset,
        framework="accelerate",
        parallelism_type="fsdp",
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        total_training_time_secs=elapsed_total,
        total_tokens_this_gpu=tokens_per_gpu_all_epochs,
        total_tokens_global=total_tokens_global,
        avg_gpu_flops=avg_tflops,
        avg_gpu_mfu=avg_mfu,
        gpu_stats=gpu_stats_during,
        training_loss=final_step_loss,
    )
    print_rank(rank, "Fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
