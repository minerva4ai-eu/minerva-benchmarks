from typing import Tuple

import pandas as pd
import torch

from . import DatasetHandler


class SquadV2Handler(DatasetHandler):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        """
        path: Path to .parquet file (e.g. squad_v2.parquet)
        tokenizer: Hugging Face tokenizer
        max_length: Maximum token length for encoding
        """
        # Load dataset from parquet
        self.data = pd.read_parquet(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure essential columns exist
        required_cols = {"question", "context", "answers"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each row, build the input text:
          Question: ...
          Context: ...
          ### Answer:
          ...
        If answer is empty (unanswerable), output will be an empty string.
        """
        item = self.data.iloc[idx]

        question = item["question"].strip()
        context = item["context"].strip()

        # SQuAD v2 has {"text": [answers], "answer_start": [positions]} or empty lists
        answer_data = item["answers"]
        if isinstance(answer_data, dict) and "text" in answer_data:
            answer_texts = answer_data.get("text", [])
            answer = answer_texts[0] if len(answer_texts) > 0 else ""
        else:
            answer = str(answer_data) if isinstance(answer_data, str) else ""

        # Build the prompt like AlpacaHandler style
        prompt = f"Question: {question}\n\nContext: {context}\n\n### Answer:\n{answer}"

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

    @staticmethod
    def collate_fn(batch):
        """
        Pad variable-length input tensors into a uniform batch.
        """
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
        """
        Compatible with both tuple-style and dict-style datasets.
        """
        if isinstance(batch[0], dict):
            input_ids = [b["input_ids"] for b in batch]
            attn = [b["attention_mask"] for b in batch]
            return SquadV2Handler.collate_fn(list(zip(input_ids, attn)))
        else:
            return SquadV2Handler.collate_fn(batch)
