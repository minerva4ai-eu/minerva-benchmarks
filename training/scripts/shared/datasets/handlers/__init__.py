from typing import TYPE_CHECKING, Any, Optional

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pandas import DataFrame
    from transformers import PreTrainedTokenizer


class DatasetHandler(Dataset):
    data: Any
    path: str
    tokenizer: "PreTrainedTokenizer"
    max_length: int

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        max_length: int,
        path: str | None = None,
        data: list[dict[str, str]] | None = None,
    ):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __raw_items_range__(self, idxs):
        raise NotImplementedError

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def collate_fn(self, batch):
        raise NotImplementedError

    def apply_chat_template(self, item: dict) -> str:
        """
        Convert dataset item into messages format and apply the tokenizer's chat template.
        Subclasses MUST override this to define how their data maps to role/content messages.
        Returns the templated text string ready for tokenization.
        """
        raise NotImplementedError


class RawTextDataset:
    data: "DataFrame"

    def __init__(self, path: str | list[str]):
        "Must implement data reading and loading on __init__()"
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def prepare_text_dataset(self) -> "HFDataset":
        raise NotImplementedError

    def prepare_packed_dataset(self, tokenizer, max_length: int, save_path: str = ""):
        raise NotImplementedError

    @classmethod
    def from_data(cls, data: "DataFrame"):
        raise NotImplementedError
