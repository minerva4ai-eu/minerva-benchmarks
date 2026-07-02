from hydra.core.config_store import ConfigStore

from .arch import HPCArchitecture
from .benchmark import BenchmarkConfig
from .dataset import DatasetConfig
from .framework import FrameworkConfig
from .model import ModelConfig, TrainArgsConfig
from .slurm import SlurmConfig


def register_configs():
    cs = ConfigStore.instance()

    # root schema — Hydra validates every composed config against this
    # cs.store(name="base", node=BenchmarkConfig)
    cs.store(name="base_MN5", node=BenchmarkConfig)

    # group schemas — each YAML must conform to these
    # model YAMLs
    cs.store(group="trainings", name="combinations", node=TrainArgsConfig)

    cs.store(group="model/gemma3_1b", name="gemma3_1b", node=ModelConfig)
    cs.store(group="model", name="gemma3_12b", node=ModelConfig)
    cs.store(group="model", name="mistral_7b", node=ModelConfig)
    cs.store(group="model", name="llama3_8b", node=ModelConfig)
    cs.store(group="model", name="llama3_70b", node=ModelConfig)

    # framework YAMLs
    cs.store(group="framework", name="accelerate", node=FrameworkConfig)
    cs.store(group="framework", name="torchrun", node=FrameworkConfig)
    cs.store(group="framework", name="deepspeed", node=FrameworkConfig)
    cs.store(group="framework", name="deepspeed-accelerate", node=FrameworkConfig)

    # arch YAMLs
    cs.store(group="arch", name="MN5", node=HPCArchitecture)

    # slurm YAMLS
    cs.store(group="slurm", name="base", node=SlurmConfig)
    cs.store(group="slurm", name="MN5", node=SlurmConfig)

    # dataset YAMLs
    cs.store(group="dataset", name="alpaca", node=DatasetConfig)
    cs.store(group="dataset", name="squadv2", node=DatasetConfig)
