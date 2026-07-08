import json

import torch

from . import DatasetHandler


class AlpacaHandler(DatasetHandler):
    def __init__(self, path, tokenizer, max_length=1024):
        path = path.replace('"', "")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("instruction", "")
        if item.get("input"):
            prompt = prompt + "\n\n" + item["input"]
        prompt = prompt + "\n\n### Response:\n" + item.get("output", "")
        enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

    @staticmethod
    def collate_fn(batch):
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

    @staticmethod
    def data_collator(batch):
        # batch is a list of tuples from __getitem__ if using Dataset of tuples
        # but since our Dataset returns tensors, HF will pass the dict if used with map-style datasets.
        # To be safe: if batch is list of dicts, handle that, otherwise handle list of tuples
        if isinstance(batch[0], dict):
            input_ids = [b["input_ids"] for b in batch]
            attn = [b["attention_mask"] for b in batch]
            return AlpacaHandler.collate_fn(list(zip(input_ids, attn)))
        else:
            return AlpacaHandler.collate_fn(batch)
