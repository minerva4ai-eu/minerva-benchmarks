from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

VALID_TRAINING_TASKS = frozenset({"Instruction-Finetuning", "Pretraining"})


@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    task: str = MISSING
    train: List[str] | None = None
    validation: List[str] | None = None
    max_seq_len: int = MISSING
    
    
    def __post_init__(self):
        if isinstance(self.train, str):
            self.train = [self.train]
        if isinstance(self.validation, str):
            self.validation = [self.validation]
