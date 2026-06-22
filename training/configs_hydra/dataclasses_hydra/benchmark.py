from dataclasses import dataclass, field
from typing import Dict, List

from configs_hydra.dataclasses_hydra import arch as a
from configs_hydra.dataclasses_hydra import dataset as d
from configs_hydra.dataclasses_hydra import framework as f
from configs_hydra.dataclasses_hydra import model as m
from configs_hydra.dataclasses_hydra import slurm as s
from omegaconf import MISSING, DictConfig


@dataclass
class MachineConfig:
    name: str = MISSING  # actual name of super computer running benchmarks
    name_pattern: str = MISSING  # configuration files naming convention pattern
    modules: List[str] | None = None
    python_environment: str | None = None
    singularity_container: str | None = None
    singularity_binds: List[str] | None = None
    singularity_args: List[str] | None = None
    single_gpu_also_valid: bool = MISSING
    env: Dict[str, str] = field(
        default_factory=dict
    )  # machine-specific environment variables


@dataclass
class ExperimentConfig:
    name: str = MISSING
    output_dir: str = "results/"
    repeat: int = 1
    yaml_filename: str = ""


@dataclass
class BenchmarkConfig(DictConfig):
    id: str
    trainings: m.TrainArgsConfig
    arch: a.HPCArchitecture
    model: m.ModelConfig
    dataset: d.DatasetConfig
    framework: f.FrameworkConfig
    machine: MachineConfig
    experiment: ExperimentConfig
    slurm: s.SlurmConfig
