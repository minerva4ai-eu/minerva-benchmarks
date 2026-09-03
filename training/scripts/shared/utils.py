import functools
import json
import logging
import os
import time

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    TrainerCallback,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0


def is_local_rank_zero(local_rank):
    return local_rank == 0


def print_rank(rank_or_msg: int | str | None, msg: str | None = None):
    """Prints the message with the rank number.
    Usage:
        print_rank("msg")       -> all ranks
        print_rank(0, "msg")    -> rank 0 only
    """
    if isinstance(rank_or_msg, str):
        rank = None
        msg = rank_or_msg
    else:
        rank = rank_or_msg

    local_rank = int(os.environ.get("RANK", 0))
    if rank is None or local_rank == rank:
        print(
            f"[ RANK {local_rank} ]: {msg}",
            flush=True,
        )


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params, trainable_params / total_params * 100


def save_summary_stats_json(summary, output_file):
    with open(os.path.join(output_file), "w") as f:
        json.dump(summary, f, indent=4)
    # print(f"Training summary saved to {output_file}")


def timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            self.__dict__.setdefault(attr, []).append(time.time() - start)
            return result

        return wrapper

    return decorator


def perf_timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @perf_timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            self.__dict__.setdefault(attr, []).append(time.perf_counter() - start)
            return result

        return wrapper

    return decorator


#####################################
#           FSDP felpers            #
#####################################
def get_fsdp_layer_to_wrap(model_name_or_path: str) -> list[str]:
    """
    Uses model config to determine FSDP wrap layers.
    More reliable than string matching on model name.
    """

    config = AutoConfig.from_pretrained(model_name_or_path)

    model_type = config.model_type.lower()  # e.g. "llama", "mistral", "mixtral"
    print(f"[FSDP] Detected model_type: {model_type}")

    mapping = {
        # model_type → wrap layers
        "llama": ["LlamaDecoderLayer"],
        "mistral": ["MistralDecoderLayer"],
        "mixtral": ["MixtralDecoderLayer", "MixtralBlockSparseTop2MLP"],
        "gemma_text": ["GemmaDecoderLayer"],
        "gemma2_text": ["Gemma2DecoderLayer"],
        "gemma3_text": ["Gemma3DecoderLayer"],
        "gemma3": ["Gemma3DecoderLayer", "SiglipEncoderLayer"],
        # "gemma3": ["Gemma3ForConditionalGeneration"],
        # "qwen2": ["Qwen2DecoderLayer"],
        # "qwen2_moe": ["Qwen2MoeDecoderLayer", "Qwen2MoeMLP"],
        # "falcon": ["FalconDecoderLayer"],
        # "phi": ["PhiDecoderLayer"],
        # "phi3": ["Phi3DecoderLayer"],
        # "gpt2": ["GPT2Block"],
        # "opt": ["OPTDecoderLayer"],
        # "bloom": ["BloomBlock"],
        # "bert": ["BertLayer"],
        # "roberta": ["RobertaLayer"],
    }

    if model_type not in mapping:
        raise ValueError(
            f"model_type '{model_type}' not in FSDP layer mapping.\n"
            f"Full config architectures: {config.architectures}\n"
            "Add it manually to the mapping dict."
        )

    layers = mapping[model_type]
    print(f"[FSDP] Using wrap layers: {layers}")
    return layers


def save_training_summary(
    *,
    output_dir,
    rank,
    model_name,
    dataset_name,
    framework="accelerate",
    parallelism_type="Unknown",
    batch_size,
    gradient_accumulation,
    learning_rate,
    total_training_time_secs,
    total_tokens_this_gpu,
    total_tokens_global,
    avg_gpu_flops,
    avg_gpu_mfu,
    gpu_stats,
    training_loss=None,
):
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    avg_gpu_power_watts = (
        sum(gpu_stats["power"]) / len(gpu_stats["power"])
        if gpu_stats and "power" in gpu_stats and gpu_stats["power"]
        else None
    )

    training_throughput_tokens_per_sec_per_gpu = (
        total_tokens_this_gpu / total_training_time_secs
        if total_training_time_secs and total_training_time_secs > 0
        else None
    )
    training_throughput_tokens_per_sec_global = (
        total_tokens_global / total_training_time_secs
        if total_training_time_secs and total_training_time_secs > 0
        else None
    )
    tokens_per_sec_per_watt_global = (
        training_throughput_tokens_per_sec_global / avg_gpu_power_watts
        if training_throughput_tokens_per_sec_global is not None and avg_gpu_power_watts
        else None
    )

    summary = {
        "nodes": int(os.environ.get("SLURM_NNODES", "1")),
        "num_gpus_per_node": int(os.environ.get("GPU_NODE", "1")),
        "total_gpus": world_size,
        "model": model_name,
        "dataset": dataset_name,
        "framework": framework,
        "parallelism_type": parallelism_type,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
    }

    metrics_summary = {
        "avg_gpu_memory_gb": (
            sum(gpu_stats["mem"]) / len(gpu_stats["mem"])
            if gpu_stats and "mem" in gpu_stats and gpu_stats["mem"]
            else None
        ),
        "peak_gpu_memory_gb": (
            max(gpu_stats["mem"])
            if gpu_stats and "mem" in gpu_stats and gpu_stats["mem"]
            else None
        ),
        "avg_gpu_utilization_percent": (
            sum(gpu_stats["util"]) / len(gpu_stats["util"])
            if gpu_stats and "util" in gpu_stats and gpu_stats["util"]
            else None
        ),
        "peak_gpu_utilization_percent": (
            max(gpu_stats["util"])
            if gpu_stats and "util" in gpu_stats and gpu_stats["util"]
            else None
        ),
        "avg_gpu_power_watts": avg_gpu_power_watts,
        "peak_gpu_power_watts": (
            max(gpu_stats["power"])
            if gpu_stats and "power" in gpu_stats and gpu_stats["power"]
            else None
        ),
        "total_execution_time_hours": total_training_time_secs / 3600
        if total_training_time_secs is not None
        else None,
        "training_throughput_tokens_per_sec_global": training_throughput_tokens_per_sec_global,
        "training_throughput_tokens_per_sec_per_gpu": training_throughput_tokens_per_sec_per_gpu,
        "tokens_per_sec_per_watt_global": tokens_per_sec_per_watt_global,
        "samples_per_sec": None,
        "total_tokens_per_gpu_all_epochs": total_tokens_this_gpu,
        "total_tokens_global_all_epochs": total_tokens_global,
        "total_training_time_hours": total_training_time_secs / 3600
        if total_training_time_secs is not None
        else None,
        "avg_epoch_training_time_sec": None,
        "avg_epoch_training_time_hours": None,
        "avg_step_training_time_sec": None,
        "avg_step_training_time_hours": None,
        "avg_gpu_flops": avg_gpu_flops,
        "avg_gpu_mfu": avg_gpu_mfu,
        "training_loss": training_loss,
        "validation_loss": None,
    }

    final_summary = {**summary, **metrics_summary}
    output_file = os.path.join(output_dir, f"training_summary_{rank}.json")
    save_summary_stats_json(final_summary, output_file)
