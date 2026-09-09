import fcntl
import os
import sys
import time

from scripts.shared.args import construct_args, get_parser
from scripts.shared.data import load_and_prepare_raw_dataset, prepare_packed_dataset
from transformers import (
    AutoTokenizer,
)

cfg = construct_args(get_parser().parse_args())

MAX_LENGTH = cfg.max_length
BATCH_SIZE = cfg.batch_size


def _is_pretokenized(train_path: str, eval_path: str) -> bool:
    """Both packed splits must exist for the dataset to be considered done."""
    return os.path.exists(train_path) and os.path.exists(eval_path)


def main():

    data_dir = "/".join(cfg.dataset_path.split("/")[:-1])
    if os.path.isdir(cfg.dataset_path):
        data_dir = cfg.dataset_path
    prepared_data = os.environ.get(
        "PRETOKENIZED_DATA_PATH",
        os.path.join(data_dir, f"{cfg.model_name}"),
    )
    train_path = os.path.join(prepared_data, f"train-packed-{MAX_LENGTH}")
    eval_path = os.path.join(prepared_data, f"eval-packed-{MAX_LENGTH}")

    # Fast path: already pretokenized, no locking needed.
    if _is_pretokenized(train_path, eval_path):
        print(f"Dataset pre-tokenized on path '{prepared_data}'...")
        sys.exit()

    # Lock logic: serialize concurrent pretokenization attempts across
    # processes (possibly on different nodes) via an advisory file lock
    # placed on the shared (parallel) filesystem next to the output data.
    os.makedirs(prepared_data, exist_ok=True)
    lock_path = os.path.join(
        prepared_data, f".pretokenize-{cfg.model_name}-{MAX_LENGTH}.lock"
    )

    with open(lock_path, "w") as lock_file:
        print(f"Waiting for pretokenization lock '{lock_path}'...")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Re-check after acquiring the lock: another process may have
            # finished the work while we were waiting.
            if _is_pretokenized(train_path, eval_path):
                print(f"Dataset pre-tokenized on path '{prepared_data}'...")
                return

            print("Starting raw data pre-tokenization...")
            print(f"Loading tokenizer for model '{cfg.model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_dataset_raw, eval_dataset_raw = load_and_prepare_raw_dataset(
                dataset_name=cfg.dataset_name,
                dataset_path=cfg.dataset_path,
                train_files=cfg.dataset_train_files,
                validation_files=cfg.dataset_validation_files,
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

            # Ensure the written data is visible on the shared filesystem
            # before releasing the lock.
            sync_path = os.path.join(prepared_data, ".sync")
            with open(sync_path, "w") as sync_file:
                sync_file.write(str(time.time()))
                sync_file.flush()
                os.fsync(sync_file.fileno())
            os.remove(sync_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
