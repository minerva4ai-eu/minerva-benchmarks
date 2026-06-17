from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from omegaconf import MISSING


class ArchitectureType(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    SSM = "ssm"


@dataclass
class ModelTrainingComboConfig:
    """A single resolved training combo — what the constraint rule receives."""

    batch_size: int = MISSING
    precision: str = MISSING
    grad_accum: int = MISSING
    max_model_length: int = MISSING
    lr: float = MISSING
    optimizer: str = "adamw"
    steps: Optional[int] = None
    epochs: Optional[int] = None
    gradient_checkpointing: bool = True
    enable_compile: bool = MISSING


@dataclass
class ModelConfig:
    # model specific train args
    training: ModelTrainingComboConfig

    name: str = MISSING  # MISSING = must be provided, no default
    path: str = MISSING
    params_billions: float = MISSING
    architecture_type: str = MISSING

    # architecture dims — used by memory rule
    total_params_billions: float = MISSING
    num_layers: int = MISSING
    hidden_dim: int = MISSING
    ffn_intermediate_dim: int = MISSING
    num_attention_heads: int = MISSING
    attention_type: str = MISSING
    tokenizer_type: str = MISSING
    num_kv_heads: int = MISSING  # = num_attention_heads for MHA
    vocab_size: int = MISSING
    head_dim: int = MISSING

    # GPU requirements — constraint metadata
    max_gpus_scale: int = MISSING
    frameworks_supported: List[str] = field(default_factory=list)
    parallelism_supported: List[str] = field(default_factory=list)

    # MoE-specific — only required when architecture_type == "moe"
    active_params_billions: Optional[float] = None
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None

    def __post_init__(self):
        # validate + normalise — converts plain string from YAML to enum
        try:
            self.architecture_type = ArchitectureType(self.architecture_type)
        except ValueError:
            raise ValueError(
                f"Invalid architecture_type: '{self.architecture_type}'. "
                f"Valid values: {[e.value for e in ArchitectureType]}"
            )

        # now you can use the enum in all subsequent checks
        if self.architecture_type == ArchitectureType.MOE:
            missing = [
                f
                for f in (
                    "active_params_billions",
                    "num_experts",
                    "top_k_experts",
                )
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"MoE model '{self.name}' missing required fields: {missing}"
                )
