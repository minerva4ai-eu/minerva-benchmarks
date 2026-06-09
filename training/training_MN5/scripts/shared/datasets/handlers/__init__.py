from typing import TYPE_CHECKING, Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class DatasetHandler(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: "PreTrainedTokenizer",
        max_length: int,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def collate_fn(self, batch):
        raise NotImplementedError

    # def data_collator(self, batch):
    #    raise NotImplementedError


class RawTextDataset:
    def __init__(self, path: str):
        "Must implement data reading and loading on __init__()"
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def prepare_text_dataset(self) -> "HFDataset":
        raise NotImplementedError

    @staticmethod
    def from_data(data: Any) -> "HFDataset":
        raise NotImplementedError
