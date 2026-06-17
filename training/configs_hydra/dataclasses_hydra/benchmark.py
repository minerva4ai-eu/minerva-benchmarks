from dataclasses import dataclass
from typing import List, Optional

from configs_hydra.dataclasses_hydra import arch as a
from configs_hydra.dataclasses_hydra import dataset as d
from configs_hydra.dataclasses_hydra import framework as f
from configs_hydra.dataclasses_hydra import model as m
from configs_hydra.dataclasses_hydra import slurm as s
from omegaconf import MISSING, DictConfig

VALID_OPTIMIZERS = {"adam", "adamw", "sgd", "adafactor"}
VALID_PRECISIONS = set([precisiontype.value for precisiontype in a.PrecisionType])


@dataclass
class TrainArgsConfig:
    """Holds lists — generator expands these into individual combos."""

    batch_sizes: List[int]  # field(default_factory=lambda: [1, 4, 8])
    precisions: List[str] = MISSING  # field(default_factory=lambda: ["bf16"])
    grad_accums: List[int] = MISSING  # field(default_factory=lambda: [1])
    lr: List[float] = MISSING  # field(default_factory=lambda: [1e-4])
    optimizer: List[str] = MISSING  # field(default_factory=lambda: ["adamw"])
    gradient_checkpointing: List[bool] = (
        MISSING  # field(default_factory=lambda: [True, False])
    )
    steps: Optional[List[int]] = MISSING  # field(default_factory=lambda: [50])
    epochs: Optional[List[int]] = MISSING  # field(default_factory=lambda: [1])
    enable_compile: Optional[List[bool]] = MISSING

    def __post_init__(self):
        bad_precisions = set(self.precisions) - VALID_PRECISIONS

        if bad_precisions:
            raise ValueError(
                f"Unknown precisions: {bad_precisions}. Valid: {VALID_PRECISIONS}"
            )

        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: '{self.optimizer}'. Valid: {VALID_OPTIMIZERS}"
            )

        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ValueError("batch_sizes must be non-empty list of ints ≥ 1")

        for lr in self.lr:
            if lr <= 0:
                raise ValueError(f"lr must be > 0, got {self.lr}")

        assert (
            len(self.gradient_checkpointing) <= 2 and not self.gradient_checkpointing
        ), (
            f"training.combinations.gradient_checkpoint must only be [True], [False] or [True, False]!! Provided: {self.gradient_checkpointing}"
        )
        if len(self.gradient_checkpointing) == 2:
            assert self.gradient_checkpointing[0] != self.gradient_checkpointing[1], (
                f"training.combinations.gradient_checkpoint must only be [True], [False] or [True, False]!! Provided: {self.gradient_checkpointing}"
            )

        if self.steps is None and self.epochs is None:
            raise ValueError("Training config must specify either 'steps' or 'epochs'")


@dataclass
class MachineConfig:
    name: str = MISSING  # actual name of super computer running benchmarks
    name_pattern: str = MISSING  # configuration files naming convention pattern
    modules: List[str] = MISSING
    python_environment: str = MISSING
    singularity_container: str = MISSING
    singularity_binds: List[str] = MISSING
    singularity_args: List[str] = MISSING
    single_gpu_also_valid: bool = MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    output_dir: str = "results/"
    repeat: int = 1
    yaml_filename: str = ""


@dataclass
class BenchmarkConfig(DictConfig):
    id: str
    trainings: TrainArgsConfig
    arch: a.HPCArchitecture
    model: m.ModelConfig
    dataset: d.DatasetConfig
    framework: f.FrameworkConfig
    machine: MachineConfig
    experiment: ExperimentConfig
    slurm: s.SlurmConfig
