import os
import sys

from shared.args import get_parser
from shared.data import load_and_prepare_raw_dataset, prepare_packed_dataset
from transformers import (
    AutoTokenizer,
)

args = get_parser().parse_args()

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size


def main():

    model_path = args.model
    model_name = args.model.split("/")[-1]
    data_dir = "/".join(args.data.split("/")[:-1])
    if os.path.isdir(args.data):
        data_dir = args.data
    prepared_data = os.environ.get(
        "PRETOKENIZED_DATA_PATH",
        os.path.join(data_dir, f"{model_name}"),
    )
    train_path = os.path.join(prepared_data, f"train-packed-{MAX_LENGTH}")
    eval_path = os.path.join(prepared_data, f"eval-packed-{MAX_LENGTH}")

    if os.path.exists(prepared_data):
        print(f"Dataset pre-tokenized on path '{prepared_data}'...")
        sys.exit()
    print("Starting raw data pre-tokenizationLoaded datasets...")
    print(f"Loading tokenizer for model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset_raw, eval_dataset_raw = load_and_prepare_raw_dataset(
        dataset_name=args.dataset,
        dataset_path=args.data,
        test_size=0.1,
        return_raw=True,
    )
    print("Loaded datasets...")

    _ = prepare_packed_dataset(
        train_dataset_raw, tokenizer, MAX_LENGTH, train_path, 0, True
    )
    _ = prepare_packed_dataset(
        eval_dataset_raw, tokenizer, MAX_LENGTH, eval_path, 0, True
    )


if __name__ == "__main__":
    main()
