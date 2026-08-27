import argparse
import ast
import json
import os
import yaml


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

    local_rank = int(os.environ["RANK"])
    if rank is None or local_rank == rank:
        print(f"[ RANK {local_rank} ]: {msg}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params, trainable_params / total_params * 100


def save_summary_stats_json(summary, output_file):
    with open(os.path.join(output_file), "w") as f:
        json.dump(summary, f, indent=4)
    # print(f"Training summary saved to {output_file}")


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        # TODO: Fill in argparse description
        description="ToDo"
    )
    parser.add_argument(
        "--yaml_file", type=str, required=True, help="Path to yaml configuration"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight Decay")
    parser.add_argument("--logging_steps", type=float, default=1, help="Logging Steps")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--max_comm_comp_overlap",
        default=False,
        action="store_true",
        help=(
            "Whether to enable maximum communication-computation overlap in FSDP. "
            + "Sets 'forward_prefetch' to True and 'limit_all_gathers' to False in FSDP config."
            + "Note: This may increase GPU memory usage, so use with caution on memory-constrained setups."
        ),
    )
    return parser.parse_args()

def parse_config(yaml_file):
    config = {}
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    return config