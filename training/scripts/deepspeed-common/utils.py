import argparse


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        # TODO: Fill in description of argparser
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
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warmup ratio for learning rate"
    )
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
    # 🆕 Precision type argument
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision type for model weights (fp32, fp16, bf16)",
    )
    # 🆕 Deepspeed configuration file
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )
    # 🆕 Gradient checkpointing
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--disable_compile",
        default=True,
        action="store_true",
        help="Disable torch.compile() in the custom trainer to avoid compilation-related device/runtime issues.",
    )

    return parser.parse_args()
