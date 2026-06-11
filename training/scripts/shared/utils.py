import functools
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


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


def get_fsdp_layer_to_wrap(model_name_or_path: str) -> list[str]:
    """
    Uses model config to determine FSDP wrap layers.
    More reliable than string matching on model name.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)

    model_type = config.model_type.lower()  # e.g. "llama", "mistral", "mixtral"
    print(f"[FSDP] Detected model_type: {model_type}")

    mapping = {
        # model_type → wrap layers
        "llama": ["LlamaDecoderLayer"],
        "mistral": ["MistralDecoderLayer"],
        "mixtral": ["MixtralDecoderLayer", "MixtralBlockSparseTop2MLP"],
        "qwen2": ["Qwen2DecoderLayer"],
        "qwen2_moe": ["Qwen2MoeDecoderLayer", "Qwen2MoeMLP"],
        "gemma_text": ["GemmaDecoderLayer"],
        "gemma2_text": ["Gemma2DecoderLayer"],
        "gemma3_text": ["Gemma3DecoderLayer"],
        "falcon": ["FalconDecoderLayer"],
        "phi": ["PhiDecoderLayer"],
        "phi3": ["Phi3DecoderLayer"],
        "gpt2": ["GPT2Block"],
        "opt": ["OPTDecoderLayer"],
        "bloom": ["BloomBlock"],
        "bert": ["BertLayer"],
        "roberta": ["RobertaLayer"],
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
