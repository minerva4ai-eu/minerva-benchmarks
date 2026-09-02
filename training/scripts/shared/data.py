import ast
import os
from typing import TYPE_CHECKING

import torch
from pandas import DataFrame
from shared.datasets.config_datasets_handlers_map import (
    DATASET_MAP,
)
from shared.utils import is_local_rank_zero, print_rank
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from shared.datasets.handlers import RawTextDataset
    from torch.utils.data import Dataset


class DatasetsNotPreparedError(Exception):
    def __init__(
        self,
    ):
        super().__init__(
            "Datasets have not been prepared and saved into a pre-tokenized format! Must run '--prepare' first!"
        )


def _resolve_dataset_files(dataset_root: str, dataset_files: list[str]) -> list[str]:
    if dataset_root is None or dataset_files is None:
        return None
    assert isinstance(dataset_files, list), (
        f"_resolve_dataset_files() | dataset_files input arg must be 'list[str]', received {type(dataset_files)}"
    )
    dataset_full_paths = []
    for item in dataset_files:
        if os.path.isabs(str(item)):
            dataset_full_paths.append(str(item))
            continue
        full_path = os.path.join(str(dataset_root), str(item))
        assert os.path.exists(full_path), (
            "Train/validation files should be absolute or direct subpaths of the dataset root path! "
            + f"Could not join paths:\n\t- {dataset_root}"
            + f"Could not join paths:\n\t- {dataset_root}"
        )
        dataset_full_paths.append(str(full_path))
    return dataset_full_paths


def parse_dataset_paths(
    dataset_root: str, train_files: list[str] = [], validation_files: list[str] = []
):
    """
    Parses dataset path argument which can be:
      - A single string path (→ do train/val split)
      - A JSON string like '{"train": "...", "validation": "..."}'
      - A Python dict string like "{'train': '...', 'validation': '...'}"
      - A dataset root path plus explicit train/validation filenames/subpaths

    Returns:
        (train_path, val_path, is_split)
        is_split = True if both train and val are provided
    """

    if not train_files:
        train_files = ast.literal_eval(os.environ.get("DATASET_TRAIN", "[]"))
    if not validation_files:
        validation_files = ast.literal_eval(os.environ.get("DATASET_VALIDATION", "[]"))
    print(f"train_files: {train_files}")
    print(f"validation_files: {validation_files}")

    if train_files or validation_files:
        if not train_files or not validation_files:
            raise ValueError(
                "Both train and validation dataset files must be provided together."
            )
        return (
            _resolve_dataset_files(dataset_root, train_files),
            _resolve_dataset_files(dataset_root, validation_files),
            True,
        )

    # Otherwise, single dataset path
    dataset_path = dataset_root
    return dataset_path, None, False


def load_and_prepare_raw_dataset(
    dataset_name: str,
    dataset_path: str,
    test_size: float = 0,
    shuffle: bool = True,
    seed: int = 42,
    train_files: list[str] = [],
    validation_files: list[str] = [],
    return_raw: bool = False,
) -> tuple["Dataset", "Dataset"] | tuple["RawTextDataset", "RawTextDataset"]:

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Get Dataset Handler
    DatasetClass = DATASET_MAP[dataset_name]

    train_path, val_path, is_split = parse_dataset_paths(
        dataset_path,
        train_files=train_files,
        validation_files=validation_files,
    )

    print_rank(
        f"📂 Dataset input type: {'train/val split' if is_split else 'single dataset'}"
    )
    print_rank(f"  Train path: {train_path}")
    if val_path:
        print_rank(f"  Validation path: {val_path}")

    # Load datasets
    if is_split and val_path:
        train_dataset = DatasetClass(path=train_path)
        eval_dataset = DatasetClass(path=val_path)
        if not return_raw:
            train_dataset = DatasetClass(path=train_path).prepare_text_dataset()
            eval_dataset = DatasetClass(path=val_path).prepare_text_dataset()

        return train_dataset, eval_dataset

    full_dataset = DatasetClass(path=train_path)
    if not return_raw:
        full_dataset = DatasetClass(path=train_path).prepare_text_dataset()
        return full_dataset
    train_data, test_data = train_test_split(
        full_dataset.data.to_dict(orient="records"),
        test_size=test_size,
        shuffle=shuffle,
        random_state=seed,
    )
    train_dataset = DatasetClass.from_data(DataFrame.from_records(train_data))
    eval_dataset = DatasetClass.from_data(DataFrame.from_records(test_data))

    return train_dataset, eval_dataset


def load_prepared_packed_dataset(path: str):

    from datasets import load_from_disk

    packed = load_from_disk(path)
    packed.set_format(type="torch", columns=["input_ids", "labels"])
    return packed


def prepare_packed_dataset(
    raw_dataset: "RawTextDataset",
    tokenizer,
    max_length: int,
    cache_path: str,
    local_rank: int,
    prepare: bool = False,
):
    """
    Only local-rank-0 per node does the CPU-heavy tokenize+pack step.
    Everyone else waits, then loads the cached arrow dataset from disk
    (memory-mapped -- cheap even when opened by many processes).
    """

    if not prepare and not os.path.exists(cache_path):
        raise DatasetsNotPreparedError()

    if is_local_rank_zero(local_rank) and not os.path.exists(cache_path):
        _ = raw_dataset.prepare_packed_dataset(tokenizer, max_length, cache_path)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_train_eval_path(args) -> tuple[str, str]:

    model_name = args.model.split("/")[-1]
    data_dir = "/".join(args.data.split("/")[:-1])
    if os.path.isdir(args.data):
        data_dir = args.data
    prepared_data = os.environ.get(
        "PRETOKENIZED_DATA_PATH",
        os.path.join(data_dir, f"{model_name}"),
    )
    train_path = os.path.join(prepared_data, f"train-packed-{args.max_length}")
    eval_path = os.path.join(prepared_data, f"eval-packed-{args.max_length}")

    return train_path, eval_path
