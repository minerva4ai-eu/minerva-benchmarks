import json
from typing import Optional

import torch
from datasets import Dataset as HFDataset

from . import DatasetHandler, RawTextDataset
from . import utils as u


class AlpacaHandler(DatasetHandler):
    def __init__(
        self,
        tokenizer,
        path: Optional[str] = None,
        data: Optional[list[torch.Tensor]] = None,
        max_length=1024,
        pad_maxlength=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_maxlength = pad_maxlength
        if path:
            path = path.replace('"', "")
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            return
        assert isinstance(data, list), (
            "Data provided must be list of dictionaries of type 'list[dict[str, str]]'!!"
        )
        assert isinstance(data[0], dict), (
            "Data provided must be list of dictionaries of type 'list[dict[str, str]]]'!!"
        )
        self.data = data

    def __len__(self):
        return len(self.data)

    def __raw_items_range__(self, idxs):
        return [self.data[idx] for idx in idxs]

    @u.perf_timed("__getitem__")
    def __getitem__(self, idx):
        item = self.data[idx]
        templated_text = self.apply_chat_template(item)

        if self.pad_maxlength:
            enc = self.tokenizer(
                templated_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        else:
            enc = self.tokenizer(
                templated_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

    def apply_chat_template(self, item: dict) -> str:
        messages = [
            {
                "role": "user",
                "content": item.get("instruction", "")
                + (f"\n\n{item['input']}" if item.get("input") else ""),
            },
            {"role": "assistant", "content": item.get("output", "")},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    @u.perf_timed("collate_fn")
    def collate_fn(self, batch):
        input_ids_list, attn_list = zip(*batch)
        lengths = [b.size(0) for b in input_ids_list]
        max_len = max(lengths)

        input_ids = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, b in enumerate(input_ids_list):
            l = b.size(0)
            input_ids[i, :l] = b
            attention_mask[i, :l] = attn_list[i]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    # @staticmethod
    # def data_collator(batch):
    #    # batch is a list of tuples from __getitem__ if using Dataset of tuples
    #    # but since our Dataset returns tensors, HF will pass the dict if used with map-style datasets.
    #    # To be safe: if batch is list of dicts, handle that, otherwise handle list of tuples
    #    if isinstance(batch[0], dict):
    #        input_ids = [b["input_ids"] for b in batch]
    #        attn = [b["attention_mask"] for b in batch]
    #        return AlpacaHandler.collate_fn(list(zip(input_ids, attn)))
    #    else:
    #        return AlpacaHandler.collate_fn(batch)


class AlpacaRawDataset(RawTextDataset):
    def __init__(
        self,
        path,
    ):
        path = path.replace('"', "")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def prepare_text_dataset(
        self,
    ) -> HFDataset:
        def build_prompt(item):
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt = prompt + "\n\n" + item["input"]
            prompt = prompt + "\n\n### Response:\n" + item.get("output", "")
            return {"text": prompt}

        processed = [build_prompt(item) for item in self.data]

        # Must be a HuggingFace Dataset, not a torch Dataset — packing requires it
        return HFDataset.from_list(processed)

    @staticmethod
    def from_data(
        data: list[dict[str, str]],
    ):

        return HFDataset.from_list(data)
