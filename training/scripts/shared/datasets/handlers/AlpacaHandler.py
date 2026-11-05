import json

import torch
from datasets import Dataset as HFDataset
from pandas import DataFrame

from . import DatasetHandler, RawTextDataset
from . import utils as u


class AlpacaHandler(DatasetHandler):
    def __init__(
        self,
        tokenizer,
        path: str | None = None,
        data: list[torch.Tensor] | None = None,
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
    def __init__(self, path, data: DataFrame = None):
        if data is not None:
            self.data = data
            return
        path = path.replace('"', "")
        with open(path, "r", encoding="utf-8") as f:
            self.data = DataFrame.from_dict(json.load(f))

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

        processed = [build_prompt(self.data.iloc[idx]) for idx in range(self.__len__())]

        # Must be a HuggingFace Dataset, not a torch Dataset — packing requires it
        return HFDataset.from_list(processed)

    def prepare_instruction_response_dataset(
        self, instructions_field="instruction", response_field="response"
    ) -> HFDataset:
        def build_instruction_response(item):
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt = prompt + "\n\n" + item["input"]
            return {instructions_field: prompt, response_field: item.get("output", "")}

        processed = [
            build_instruction_response(self.data.iloc[idx])
            for idx in range(self.__len__())
        ]

        # Must be a HuggingFace Dataset, not a torch Dataset — packing requires it
        return HFDataset.from_list(processed)

    def build_packed_instruction_dataset(
        self,
        raw_dataset,
        tokenizer,
        max_length,
        instructions_field="instruction",
        response_field="response",
    ):
        """Tokenize, concatenate, and chunk into fixed-length blocks."""

        def tokenize_fn(examples):
            input_ids_list, labels_list = [], []
            for prompt, response in zip(
                examples[instructions_field], examples[response_field]
            ):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response, add_special_tokens=False)[
                    "input_ids"
                ] + [tokenizer.eos_token_id]
                ids = prompt_ids + response_ids
                labels = [-100] * len(
                    prompt_ids
                ) + response_ids  # mask prompt, keep response as labels
                input_ids_list.append(ids)
                labels_list.append(labels)
            return {"input_ids": input_ids_list, "labels": labels_list}

        tokenized = raw_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing",
        )

        # Instead of truncating, pad the final block:
        def group_texts_pad(examples):
            concatenated_ids = sum(examples["input_ids"], [])
            concatenated_labels = sum(examples["labels"], [])

            # Create full blocks and keep remainder as partial block
            blocks_ids = []
            blocks_labels = []

            for i in range(0, len(concatenated_ids), max_length):
                block_ids = concatenated_ids[i : i + max_length]
                block_labels = concatenated_labels[i : i + max_length]

                # Pad if necessary
                if len(block_ids) < max_length:
                    pad_len = max_length - len(block_ids)
                    block_ids = block_ids + [tokenizer.pad_token_id] * pad_len
                    block_labels = (
                        block_labels + [-100] * pad_len
                    )  # Don't compute loss on padding

                blocks_ids.append(block_ids)
                blocks_labels.append(block_labels)

            return {"input_ids": blocks_ids, "labels": blocks_labels}

        """
        def group_texts(examples):
            concatenated_ids = sum(examples["input_ids"], [])
            concatenated_labels = sum(examples["labels"], [])
            total_len = (len(concatenated_ids) // max_length) * max_length
            return {
                "input_ids": [
                    concatenated_ids[i : i + max_length]
                    for i in range(0, total_len, max_length)
                ],
                "labels": [
                    concatenated_labels[i : i + max_length]
                    for i in range(0, total_len, max_length)
                ],
            }
        """
        packed = tokenized.map(
            group_texts_pad,
            batched=True,
            # FIX: Explicitly remove un-packed tokenized columns before packing
            remove_columns=tokenized.column_names,
            desc=f"Packing into {max_length}-token blocks",
        )
        return packed

    def prepare_packed_dataset(self, tokenizer, max_length: int, save_path: str = ""):

        raw_dataset = self.prepare_instruction_response_dataset()

        packed_dataset = self.build_packed_instruction_dataset(
            raw_dataset, tokenizer, max_length
        )

        if save_path:
            packed_dataset.save_to_disk(save_path)

        return packed_dataset

    @classmethod
    def from_data(
        cls,
        data: DataFrame,
    ):

        # return HFDataset.from_list(data)
        return cls("", data)
