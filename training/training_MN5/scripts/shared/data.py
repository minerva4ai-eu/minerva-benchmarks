import ast
import json
from typing import TYPE_CHECKING, Callable, Tuple

from torch.utils.data import random_split

from shared.datasets.config_datasets_handlers_map import (
    DATASET_HANDLER_MAP,
)
from shared.utils import print_rank

if TYPE_CHECKING:
    from torch.utils.data.dataset import Subset
    from transformers import PreTrainedTokenizer

    from shared.datasets.handlers import DatasetHandler


class CollateFnError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


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


def load_dataset(
    dataset_name: str,
    dataset_path: str,
    tokenizer: "PreTrainedTokenizer",
    max_length: int,
) -> Tuple["DatasetHandler", "DatasetHandler", Callable, Callable]:

    if dataset_name not in DATASET_HANDLER_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Get Dataset Handler
    DatasetHandlerClass = DATASET_HANDLER_MAP[dataset_name]

    train_path, val_path, is_split = parse_dataset_paths(dataset_path)

    print_rank(
        f"📂 Dataset input type: {'train/val split' if is_split else 'single dataset'}"
    )
    print_rank(f"  Train path: {train_path}")
    if val_path:
        print_rank(f"  Validation path: {val_path}")

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
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        dataset_for_collate = dataset

    # Resolve collate_fn() for both cases of if condition above
    def resolve_collate(
        ds_obj: "DatasetHandler | Subset", fallback: "DatasetHandler"
    ) -> Callable:
        if hasattr(ds_obj, "collate_fn") and isinstance(ds_obj, "DatasetHandler"):
            return ds_obj.collate_fn
        if (
            isinstance(ds_obj, "Subset")
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
