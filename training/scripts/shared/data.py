import json
import os
import random
from typing import TYPE_CHECKING, Any, Callable, Tuple, cast

from shared.datasets.config_datasets_handlers_map import (
    DATASET_HANDLER_MAP,
    DATASET_MAP,
)
from shared.datasets.handlers import DatasetHandler
from shared.utils import print_rank
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer


class CollateFnError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


def _resolve_dataset_files(dataset_root: str, dataset_files: list[str]) -> list[str]:
    if dataset_root is None or dataset_files is None:
        return None
    if isinstance(dataset_files, list):
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

    if os.path.isabs(str(dataset_file)):
        return str(dataset_file)
    full_path = os.path.join(str(dataset_root), str(dataset_file))

    assert os.path.exists(full_path), (
        "Train/validation files should be absolute or direct subpaths of the dataset root path! "
        + f"Could not join paths:\n\t- {dataset_root}"
        + f"Could not join paths:\n\t- {dataset_root}"
    )

    return full_path


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
    print(f"DATASET_TRAIN: {os.environ.get('DATASET_TRAIN', '[]')}")
    print(f"DATASET_VALIDATION: {os.environ.get('DATASET_VALIDATION', '[]')}")
    train_files = train_files or json.loads(os.environ.get("DATASET_TRAIN", "[]"))
    validation_files = validation_files or json.loads(
        os.environ.get("DATASET_VALIDATION", "[]")
    )
    print(f"DATASET_TRAIN: {os.environ.get('DATASET_TRAIN', '[]')}")
    print(f"DATASET_VALIDATION: {os.environ.get('DATASET_VALIDATION', '[]')}")

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


def load_dataset(
    dataset_name: str,
    dataset_path: str,
    tokenizer: "PreTrainedTokenizer",
    max_length: int,
    train_files: list[str] | None = None,
    validation_files: list[str] | None = None,
) -> Tuple["DatasetHandler| Subset", "DatasetHandler | Subset", Callable, Callable]:

    if dataset_name not in DATASET_HANDLER_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Get Dataset Handler
    DatasetHandlerClass = cast(Any, DATASET_HANDLER_MAP[dataset_name])

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

    if isinstance(train_path, list) and not train_path:
        raise ValueError("Resolved train dataset file list is empty.")
    if isinstance(val_path, list) and not val_path:
        raise ValueError("Resolved validation dataset file list is empty.")

    # Load datasets
    if is_split and val_path:
        train_dataset = DatasetHandlerClass(
            path=train_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        eval_dataset = DatasetHandlerClass(
            path=val_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataset_for_collate = train_dataset
    else:
        dataset = DatasetHandlerClass(
            path=train_path, tokenizer=tokenizer, max_length=max_length
        )
        train_size = int(0.9 * len(dataset))

        indices = list(range(len(dataset)))
        random.shuffle(indices)  # or use a seeded Generator for reproducibility
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        train_dataset = DatasetHandlerClass(
            data=dataset.__raw_items_range__(train_indices),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        eval_dataset = DatasetHandlerClass(
            data=dataset.__raw_items_range__(eval_indices),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataset_for_collate = train_dataset

    # Resolve collate_fn() for both cases of if condition above
    def resolve_collate(
        ds_obj: "DatasetHandler | Subset", fallback: "DatasetHandler"
    ) -> Callable:
        if hasattr(ds_obj, "collate_fn") and isinstance(ds_obj, DatasetHandler):
            return ds_obj.collate_fn
        if (
            isinstance(ds_obj, Subset)
            and hasattr(ds_obj, "dataset")
            and hasattr(ds_obj.dataset, "collate_fn")
        ):
            return getattr(ds_obj.dataset, "collate_fn")
        if fallback is not None and hasattr(fallback, "collate_fn"):
            return getattr(fallback, "collate_fn")
        raise CollateFnError(
            "resolve_collate() failed to resolve collatefn()! Exiting..."
        )

    collate_fn_train = resolve_collate(train_dataset, dataset_for_collate)
    collate_fn_eval = resolve_collate(eval_dataset, dataset_for_collate)

    return train_dataset, eval_dataset, collate_fn_train, collate_fn_eval


def load_and_prepare_raw_dataset(
    dataset_name: str,
    dataset_path: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 42,
    train_files: list[str] | None = None,
    validation_files: list[str] | None = None,
) -> Tuple["Dataset", "Dataset"]:

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
        train_dataset = DatasetClass(path=train_path).prepare_text_dataset()
        eval_dataset = DatasetClass(path=val_path).prepare_text_dataset()

        return train_dataset, eval_dataset

    full_dataset = DatasetClass(path=train_path).prepare_text_dataset()
    train_data, test_data = train_test_split(
        full_dataset.data.to_pylist(),
        test_size=test_size,
        shuffle=shuffle,
        random_state=seed,
    )
    train_dataset = DatasetClass.from_data(train_data)
    eval_dataset = DatasetClass.from_data(test_data)
    return train_dataset, eval_dataset
