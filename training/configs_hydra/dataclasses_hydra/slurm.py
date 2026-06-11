from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING


@dataclass
class SbatchConfig:
    gpus_per_node: int = field(default=1)
    cpus_per_task: int = field(default=1)
    cpus_per_gpu: int = field(default=1)
    tasks_per_node: int = field(default=1)
    chdir: str = MISSING
    nodes: int = field(default=1)
    gres: str = MISSING
    output: str = MISSING
    error: str = MISSING
    extra_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.nodes < 1:
            raise ValueError(f"nodes must be ≥ 1, got {self.nodes}")
        if self.gpus_per_node < 1:
            raise ValueError(f"gpus_per_node must be ≥ 1, got {self.gpus_per_node}")


@dataclass
class SrunConfig:
    gpus_per_node: int = field(default=1)
    cpus_per_task: int = field(default=1)
    cpus_per_gpu: int = field(default=1)
    tasks_per_node: int = field(default=1)
    nodes: int = field(default=1)
    gres: str = MISSING


@dataclass
class SlurmConfig:
    account: str = MISSING
    qos: str = MISSING
    partition: str = MISSING
    sbatch: SbatchConfig = field(default_factory=SbatchConfig)
    srun: SrunConfig = field(default_factory=SrunConfig)
