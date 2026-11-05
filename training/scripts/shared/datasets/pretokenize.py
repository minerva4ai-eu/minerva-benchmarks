import logging
import os
import signal
import sys
import time
from contextlib import contextmanager

from scripts.shared.args import construct_config, get_parser
from scripts.shared.data import load_and_prepare_raw_dataset, prepare_packed_dataset
from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(f"MINERVA_BENCH.{__name__}")

cfg = construct_config(get_parser().parse_args())

MAX_LENGTH = cfg.max_length
BATCH_SIZE = cfg.batch_size


@contextmanager
def exclusive_lock(lock_path: str, poll_secs: int = 5, stale_secs: int = 6 * 3600):
    """Cross-process/job lock; O_EXCL creation is atomic on shared filesystems."""
    waited = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                # Lock left behind by a crashed job
                if time.time() - os.path.getmtime(lock_path) > stale_secs:
                    logger.warning(f"Removing stale lock '{lock_path}'")
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            if not waited:
                logger.info(f"Waiting for lock '{lock_path}'...")
                waited = True
            time.sleep(poll_secs)

    with os.fdopen(fd, "w") as f:
        f.write(f"job={os.environ.get('SLURM_JOB_ID', 'unknown')} pid={os.getpid()}")
    logger.info(f"Lock acquired: '{lock_path}'")
    # scancel sends SIGTERM, which skips `finally` unless converted to an exception.
    prev_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, prev_handler)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        logger.info(f"Lock released: '{lock_path}'")


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

    os.makedirs(prepared_data, exist_ok=True)
    lock_path = os.path.join(
        prepared_data, f".pretokenize-{cfg.model_name}-{MAX_LENGTH}.lock"
    )
    # Marker is written only after a full success, so partial outputs are never trusted
    done_path = os.path.join(
        prepared_data, f".pretokenize-{cfg.model_name}-{MAX_LENGTH}.done"
    )

    if os.path.exists(done_path):
        logger.info(f"Dataset pre-tokenized on path '{prepared_data}'...")
        sys.exit()

    with exclusive_lock(lock_path):
        # Re-check: another job may have finished while we waited
        if os.path.exists(done_path):
            logger.info(f"Dataset pre-tokenized on path '{prepared_data}'...")
            return

        logger.info("Starting raw data pre-tokenization...")
        logger.info(f"Loading tokenizer for model '{cfg.model_name}'...")
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
        logger.info("Loaded datasets...")

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

        with open(done_path, "w") as done_file:
            done_file.write(str(time.time()))
            done_file.flush()
            os.fsync(done_file.fileno())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s : %(message)s",
    )
    main()
