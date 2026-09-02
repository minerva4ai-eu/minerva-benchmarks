import functools
import gc
import math
import os
import time

import torch
import torch.distributed as dist
from shared.args import get_fsdp_parser
from shared.data import collate_fn, get_train_eval_path, load_prepared_packed_dataset
from shared.flops import mfu_callback_from_hf_config
from shared.gpu_monitor import start_gpu_monitor
from shared.utils import (
    is_main_process,
    print_rank,
    save_training_summary,
    setup_distributed,
)
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

args = get_fsdp_parser().parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size


def load_model(model_path, dtype):
    """
    dist.is_initialized() is already True by the time this runs (see setup_distributed
    above), and ACCELERATE_USE_FSDP / FSDP_CPU_RAM_EFFICIENT_LOADING are exported in the
    SLURM launch script. That means transformers' internal is_fsdp_enabled() check now
    correctly evaluates True, and from_pretrained will put non-loading ranks on the meta
    device automatically -- no need to hand-roll that logic here.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    return model


def get_transformer_layer_classes(model):
    """Resolve the actual decoder-layer class(es) generically via _no_split_modules,
    rather than relying on a hardcoded name lookup."""
    no_split = getattr(model, "_no_split_modules", None)
    if not no_split:
        raise ValueError(
            "Model has no `_no_split_modules`; set the layer class explicitly instead."
        )
    classes = set()
    for module in model.modules():
        if type(module).__name__ in no_split:
            classes.add(type(module))
    if not classes:
        raise ValueError(
            f"Could not find any modules matching {no_split} in the model."
        )
    return classes


def wrap_fsdp(model, dtype, v2: bool = True, max_comm_comp_overlap: bool = False):

    layer_classes = get_transformer_layer_classes(model)
    print_rank(
        0, f"FSDP auto-wrap layer classes: {[c.__name__ for c in layer_classes]}"
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )

    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    # 2. Materialize meta tensors into empty CUDA memory on current rank
    current_device = torch.cuda.current_device()

    if is_main_process(dist.get_rank()):
        print_rank(
            f"BEFORE FSDP WRAP | sample weight sum: {next(model.parameters()).sum().item()}",
        )
    torch.cuda.synchronize()
    if v2:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            # Note: buffer_dtype is automatically handled in v2
        )
        layer_classes = tuple(get_transformer_layer_classes(model))
        # If initialized on meta device:
        current_device = torch.cuda.current_device()

        for param in model.parameters():
            if param.is_meta:
                param.data = torch.empty_like(param.data, device=current_device)

        for module in model.modules():
            if isinstance(module, layer_classes):
                fully_shard(
                    module,
                    mp_policy=mp_policy,
                )
        fully_shard(
            model,
            mp_policy=mp_policy,
        )

        if max_comm_comp_overlap:
            # Enable backward prefetch explicitly across the fully_shard mesh
            for module in model.modules():
                if isinstance(module, layer_classes):
                    module.set_modules_to_backward_prefetch([])
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        # DEPRECATED: Moved to FSDPv2, wrappign with fully_shard()
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=current_device,
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=not max_comm_comp_overlap,
            forward_prefetch=max_comm_comp_overlap,
            backward_prefetch=(
                BackwardPrefetch.BACKWARD_PRE
                if max_comm_comp_overlap
                else BackwardPrefetch.BACKWARD_POST
            ),
        )
        print_rank("Model FSDP wrapped...")

        # Activation checkpointing, equivalent to fsdp_config["activation_checkpointing"]=True
        check_fn = lambda submodule: isinstance(submodule, tuple(layer_classes))
        checkpoint_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            # checkpoint_impl=CheckpointImpl.REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_fn, check_fn=check_fn
        )

    print_rank(
        f"AFTER FSDP WRAP | sample weight sum: {next(model.parameters()).sum().item()}",
    )
    print_rank("FSDP Model settings set...")
    return model


def save_model(model, tokenizer, output_dir, rank):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        # unwrap: model is FSDP(model) with no other wrapper, so .module gives the raw model
        model.module.save_pretrained(output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(output_dir)
        print_rank(0, f"Saved model + tokenizer to {output_dir}")
    dist.barrier()


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

    # --- Tokenizer ---
    print_rank(rank, f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_prepared_packed_dataset(train_path)
    # eval_dataset = load_prepared_packed_dataset(eval_path)

    print_rank(
        rank,
        f"Packed train dataset size: {len(train_dataset)} blocks of {MAX_LENGTH} tokens",
    )

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=32
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 1,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
    )

    # --- Model + FSDP ---
    print_rank(0, f"Loading model {model_path}...")
    model = load_model(model_path, dtype)
    print_rank(
        0, f"Max communication-computation overlap: {args.max_comm_comp_overlap}"
    )
    model = wrap_fsdp(
        model,
        dtype,
        max_comm_comp_overlap=args.max_comm_comp_overlap,
    )
    print_rank(rank, "Model wrapped with FSDP.")

    if args.enable_compile:
        print_rank(0, "torch.compile enabled.")
        model = torch.compile(
            model, backend="inductor", mode="max-autotune-no-cudagraphs"
        )
        print_rank(0, "Model compilation finished.")

    # --- Optimizer / schedule ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    grad_accum_steps = args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    num_epochs = args.epochs if args.epochs is not None else 1
    num_epochs = 1
    total_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else steps_per_epoch * num_epochs
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.03 * total_steps)),
        num_training_steps=total_steps,
    )

    peak_gpu_tflops = (
        float(os.environ["GPU_PEAK_TFLOPS"])
        if os.environ.get("GPU_PEAK_TFLOPS")
        else None
    )
    gpu_name = os.environ.get("GPU_NAME", "Unknown GPU")
    print_rank(0, f"GPU: {gpu_name} | peak TFLOPs for MFU: {peak_gpu_tflops}")

    gpu_stats_during, stop_flag = start_gpu_monitor(
        interval_sec=5, n_gpus=int(os.environ.get("GPUS_PER_NODE", 1))
    )

    # --- Training loop ---
    print_rank("Starting trainining...")
    model.train()
    total_tokens_this_gpu = 0
    global_step = 0
    start_time = time.time()

    flopsCallback_megatronLM = mfu_callback_from_hf_config(
        AutoConfig.from_pretrained(model_path),
        tokenizer,
        gpu_peak_flops=peak_gpu_tflops,
        seq_length=args.max_length,
        trainer_callback=False,
    )

    print_rank(0, "Beginning of training...")
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        if global_step == 0:
            flopsCallback_megatronLM.on_step_begin()
        for micro_step, batch in enumerate(train_dataloader):
            print_rank(
                0,
                f"micro_step:{micro_step} - batch_keys:{batch.keys()} - batch_len:{len(batch[list(batch.keys())[0]])}",
            )
            batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            total_tokens_this_gpu += batch["input_ids"].numel()

            if (micro_step + 1) % grad_accum_steps == 0:
                print_rank(
                    0,
                    f"global_step:{global_step} ",
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                flopsCallback_megatronLM.on_step_end(
                    micro_batch_size=BATCH_SIZE,
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

                if args.max_steps is not None and global_step >= int(args.max_steps):
                    break

                flopsCallback_megatronLM.on_step_begin()

        if args.max_steps is not None and global_step >= int(args.max_steps):
            break

    elapsed_total = time.time() - start_time
    stop_flag["stop"] = True
    time.sleep(2)

    # --- Global token count + summary ---
    tokens_tensor = torch.tensor(total_tokens_this_gpu, device=local_rank)
    dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
    total_tokens_global = tokens_tensor.item()

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

    save_training_summary(
        output_dir=output_dir,
        rank=rank,
        model_name=model_name,
        dataset_name=args.dataset,
        framework="torchrun",
        parallelism_type="fsdp",
        batch_size=BATCH_SIZE,
        gradient_accumulation=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        total_training_time_secs=elapsed_total,
        total_tokens_this_gpu=total_tokens_this_gpu,
        total_tokens_global=total_tokens_global,
        avg_gpu_flops=avg_tflops,
        avg_gpu_mfu=avg_mfu,
        gpu_stats=gpu_stats_during,
        training_loss
    )

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print_rank(rank, "Fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
