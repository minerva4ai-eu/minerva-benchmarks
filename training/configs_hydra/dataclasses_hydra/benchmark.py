from dataclasses import dataclass, field
from enum import Enum

from configs_hydra.dataclasses_hydra import arch as a
from configs_hydra.dataclasses_hydra import dataset as d
from configs_hydra.dataclasses_hydra import framework as f
from configs_hydra.dataclasses_hydra import model as m
from configs_hydra.dataclasses_hydra import slurm as s
from omegaconf import MISSING, DictConfig


class EnvMode(str, Enum):
    venv = "venv"
    singularity = "singularity"


@dataclass
class MachineConfig:
    name: str = MISSING  # actual name of super computer running benchmarks
    name_pattern: str = MISSING  # configuration files naming convention pattern
    framework_name_pattern: str = (
        MISSING  # naming convention pattern specific for framework configurations
    )
    modules: list[str] | None = None
    runtime_env_mode: EnvMode = MISSING
    singularity_binds: list[str] | None = None
    singularity_args: list[str] | None = None
    single_gpu_also_valid: bool = MISSING
    env: dict[str, str] = field(
        default_factory=dict
    )  # machine-specific environment variables


@dataclass
class ExperimentConfig:
    name: str = MISSING
    output_dir: str = "results/"
    repeat: int = 1
    yaml_filename: str | None = ""
    env: dict[str, str] = field(default_factory=dict)


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
