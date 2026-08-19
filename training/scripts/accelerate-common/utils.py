import argparse
import ast
import json
import os


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
        # Todo: Fill in description
        description="ToDo"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=2e-5, help="Weight Decay")
    parser.add_argument("--logging_steps", type=float, default=1, help="Logging Steps")
    parser.add_argument(
        "--enable_steps",
        type=bool,
        default=False,
        help="Enable maximum steps instead of Epochs",
    )
    parser.add_argument("--max_steps", type=float, default=None, help="Maximum steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length")
    parser.add_argument("--epochs_save_every", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision type for model weights (fp32, fp16, bf16)",
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
    parser.add_argument(
        "--enable_compile",
        default=False,
        action="store_true",
        help="Disable torch.compile() in the custom trainer to avoid compilation-related device/runtime issues.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Disable torch.compile() in the custom trainer to avoid compilation-related device/runtime issues.",
    )

    return parser.parse_args()


def parse_dataset_paths(data_arg):
    """
    Parses dataset path argument which can be:
      - A single string path (→ do train/val split)
      - A JSON string like '{"train": "...", "validation": "..."}'
      - A Python dict string like "{'train': '...', 'validation': '...'}"

    Returns:
        (train_path, val_path, is_split)
        is_split = True if both train and val are provided
    """
    train_path, val_path = None, None

    # Try Python-style dict
    try:
        parsed = ast.literal_eval(data_arg)
        if isinstance(parsed, dict) and "train" in parsed:
            return parsed["train"], parsed.get("validation"), True
    except (ValueError, SyntaxError):
        pass

    # Try JSON-style dict
    try:
        parsed = json.loads(data_arg)
        if isinstance(parsed, dict) and "train" in parsed:
            return parsed["train"], parsed.get("validation"), True
    except json.JSONDecodeError:
        pass

    # Otherwise, single dataset path
    return data_arg, None, False
