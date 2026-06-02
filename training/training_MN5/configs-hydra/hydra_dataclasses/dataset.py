from dataclasses import dataclass

from omegaconf import MISSING

VALID_TRAINING_TASKS = frozenset({"Instruction-Finetuning", "Pretraining"})


@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    task: str = "Instruction-Finetuning"
    train: str = MISSING
    validation: str | None = None
    max_seq_len: int = MISSING
