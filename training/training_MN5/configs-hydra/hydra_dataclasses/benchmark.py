from dataclasses import dataclass

from hydra_dataclasses import arch as a
from hydra_dataclasses import dataset as d
from hydra_dataclasses import framework as f
from hydra_dataclasses import model as m
from omegaconf import MISSING, DictConfig


@dataclass
class MachineConfig:
    name: str = MISSING  # actual name of super computer running benchmarks
    name_pattern: str = MISSING  # configuration files naming convention pattern
    modules: str = MISSING  # space separated modules to load
    python_environment: str = MISSING
    singularity_container: str = MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    output_dir: str = "results/"
    repeat: int = 1


@dataclass
class BenchmarkConfig(DictConfig):
    arch: a.HPCArchitecture
    model: m.ModelConfig
    dataset: d.DatasetConfig
    framework: f.FrameworkConfig
    machine: MachineConfig
    experiment: ExperimentConfig
