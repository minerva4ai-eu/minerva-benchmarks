from hydra.core.config_store import ConfigStore

from .arch import HPCArchitecture
from .benchmark import BenchmarkConfig
from .dataset import DatasetConfig
from .framework import FrameworkConfig
from .model import ModelConfig, TrainArgsConfig
from .slurm import SlurmConfig

from omegaconf import OmegaConf

from pathlib import Path

# 1. Define and register the custom resolver
# This must happen BEFORE the @hydra.main decorator executes
OmegaConf.register_new_resolver(
    "abs_path", 
    lambda relative_path: str(Path.cwd() / relative_path)
)

def register_configs():

    # root schema — Hydra validates every composed config against this
    cs = ConfigStore.instance()

    register_base(cs)
    register_MN5(cs)


def register_base(cs: ConfigStore):
    # arch YAMLs
    cs.store(group="arch", name="base", node=HPCArchitecture)

    # dataset YAMLs
    cs.store(group="dataset", name="alpaca", node=DatasetConfig)
    cs.store(group="dataset", name="squadv2", node=DatasetConfig)

    # framework YAMLs
    cs.store(group="framework", name="accelerate", node=FrameworkConfig)
    cs.store(group="framework", name="torchrun", node=FrameworkConfig)
    cs.store(group="framework", name="deepspeed", node=FrameworkConfig)
    cs.store(group="framework", name="deepspeed-accelerate", node=FrameworkConfig)

    # model YAMLs
    # cs.store(group="model", name="base_training", node=TrainArgsConfig)

    cs.store(group="model", name="gemma3_1b", node=ModelConfig)
    cs.store(group="model", name="gemma3_12b", node=ModelConfig)
    cs.store(group="model", name="mistral_7b", node=ModelConfig)
    cs.store(group="model", name="llama3_8b", node=ModelConfig)
    cs.store(group="model", name="llama3_70b", node=ModelConfig)

    # slurm YAMLS
    cs.store(group="slurm", name="base", node=SlurmConfig)

    # experiment YAMLS
    cs.store(name="base", node=BenchmarkConfig)


def register_MN5(cs: ConfigStore):

    # arch YAMLs
    cs.store(group="arch", name="MN5", node=HPCArchitecture)

    # dataset YAMLs
    cs.store(group="dataset", name="MN5/alpaca-MN5", node=DatasetConfig)
    cs.store(group="dataset", name="MN5/squadv2-MN5", node=DatasetConfig)

    # framework YAMLs
    cs.store(group="framework", name="MN5/accelerate-MN5", node=FrameworkConfig)
    cs.store(group="framework", name="MN5/torchrun-MN5", node=FrameworkConfig)
    cs.store(group="framework", name="MN5/deepspeed-MN5", node=FrameworkConfig)
    cs.store(
        group="framework", name="MN5/deepspeed-accelerate-MN5", node=FrameworkConfig
    )

    # model YAMLs

    cs.store(group="model", name="MN5/gemma3_1b-MN5", node=ModelConfig)
    cs.store(group="model", name="MN5/gemma3_12b-MN5", node=ModelConfig)
    cs.store(group="model", name="MN5/mistral_7b-MN5", node=ModelConfig)
    cs.store(group="model", name="MN5/llama3_8b-MN5", node=ModelConfig)
    cs.store(group="model", name="MN5/llama3_70b-MN5", node=ModelConfig)

    # slurm YAMLS
    cs.store(group="slurm", name="MN5", node=SlurmConfig)

    # experiment YAMLS
    cs.store(name="MN5-singularity", node=BenchmarkConfig)
    cs.store(name="MN5-uv-venv", node=BenchmarkConfig)
