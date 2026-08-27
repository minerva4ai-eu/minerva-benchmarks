from dataclasses import dataclass, field
from enum import Enum

from configs_hydra.dataclasses_hydra import arch as a
from omegaconf import MISSING

VALID_OPTIMIZERS = {"adam", "adamw", "sgd", "adafactor"}
VALID_PRECISIONS = set([precisiontype.value for precisiontype in a.PrecisionType])


class ArchitectureType(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    SSM = "ssm"


@dataclass
class TrainArgsConfig:
    """Holds lists — generator expands these into individual combos."""

    batch_sizes: list[int]  # field(default_factory=lambda: [1, 4, 8])
    precisions: list[str] | None = MISSING  # field(default_factory=lambda: ["bf16"])
    grad_accums: list[int] | None = MISSING  # field(default_factory=lambda: [1])
    lr: list[float] | None = MISSING  # field(default_factory=lambda: [1e-4])
    optimizer: list[str] | None = MISSING  # field(default_factory=lambda: ["adamw"])
    gradient_checkpointing: list[bool] | None = (
        MISSING  # field(default_factory=lambda: [True, False])
    )
    steps: list[int] | None = MISSING  # field(default_factory=lambda: [50])
    epochs: list[int] | None = MISSING  # field(default_factory=lambda: [1])
    enable_compile: list[bool] | None = MISSING

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
class ModelTrainingComboConfig:
    """A single resolved training combo — what the constraint rule receives."""

    batch_size: int = MISSING
    grad_accum: int = MISSING
    max_model_length: int = MISSING
    precision: str | None = MISSING
    lr: float | None = MISSING
    optimizer: str | None = "adamw"
    steps: int | None = None
    epochs: int | None = None
    gradient_checkpointing: bool | None = True  # WARNING! Is not used in any benchmark
    enable_compile: bool | None = MISSING


@dataclass
class ModelConfig:
    # model specific train args
    training: ModelTrainingComboConfig
    combinations: TrainArgsConfig

    name: str = MISSING  # MISSING = must be provided, no default
    path: str = MISSING
    # Cannot type check as Literal, but for now can be
    # only "dense" or "moe". Checked only on function rules.megatron_divisors
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
    frameworks_supported: list[str] | None = field(default_factory=list)
    parallelism_supported: list[str] | None = field(default_factory=list)
    megatron_parallelism_supported: list[str] = field(default_factory=list)

    # MoE-specific — only required when architecture_type == "moe"
    active_params_billions: float | None = None
    num_experts: int | None = None
    top_k_experts: int | None = None

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
