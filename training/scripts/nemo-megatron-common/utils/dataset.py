from typing import Any, Callable, Dict, List, Tuple

from transformers import PreTrainedTokenizer
import torch


def sft_tulu_tokenize_and_truncate(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length: int
) -> Tuple[List[int], List[int], List[int]]:
    """Prepare a conversation for Supervised Fine-Tuning (SFT).

    This function processes a conversation by tokenizing all messages and creating labels
    where only assistant responses contribute to the training loss. Non-assistant messages
    (user, system) are masked with -100 in the labels tensor.

    Adapted from: https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385
    """
    messages = row["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return input_ids.flatten().tolist(), labels.flatten().tolist(), attention_mask.flatten().tolist()

def make_sft_collate(
    tokenizer: PreTrainedTokenizer, max_seq_length: int, label_pad_token_id: int = -100
) -> Callable[[List[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create a collate function for SFT DataLoader.

    Returns a collate function that tokenizes, left-pads, and shifts sequences
    for causal language modeling. The returned tensors are ready for training:
    - input_ids are shifted left (excludes last token)
    - labels are shifted right (excludes first token)

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer with pad_token_id set.
        max_seq_length: Maximum sequence length for padding/truncation.
        label_pad_token_id: Token ID used to pad labels (default: -100, ignored by loss).
    """
    # Add 1 to account for the shift between input_ids and labels
    max_seq_length += 1

    def _apply(row: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
        return sft_tulu_tokenize_and_truncate(row, tokenizer, max_seq_length)

    def _pad_left(batch_seqs: List[List[int]], pad_id: int, length: int) -> List[List[int]]:
        return [[pad_id] * (length - len(s)) + s for s in batch_seqs]

    def _collate(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, labels, attn = [], [], []
        for row in batch:
            ids, lbl, at = _apply(row)
            input_ids.append(ids)
            labels.append(lbl)
            attn.append(at)

        input_ids = _pad_left(input_ids, tokenizer.pad_token_id, max_seq_length)
        labels = _pad_left(labels, label_pad_token_id, max_seq_length)
        attn = _pad_left(attn, 0, max_seq_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)

        return input_ids[..., :-1], labels[..., 1:], attn[..., :-1]

    return _collate
