from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING


@dataclass
class ParallelismSpec:
    """Constraints for one parallelism strategy."""

    min_gpus: int = 1
    max_gpus: int = 999

    def __post_init__(self):
        if self.min_gpus < 1:
            raise ValueError(f"min_gpus must be ≥ 1, got {self.min_gpus}")
        if self.max_gpus < self.min_gpus:
            raise ValueError(
                f"max_gpus ({self.max_gpus}) must be ≥ min_gpus ({self.min_gpus})"
            )


@dataclass
class ScriptsConfig:
    run: str = MISSING
    finetune: str = MISSING
    shared: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.run:
            raise ValueError("scripts.run must be a non-empty path")
        if not self.finetune:
            raise ValueError("scripts.finetune must be a non-empty path")


VALID_FRAMEWORKS = frozenset({"torchrun", "accelerate", "deepspeed"})


@dataclass
class FrameworkConfig:
    name: str = MISSING
    parallelism: Dict[str, ParallelismSpec] = field(default_factory=dict)
    scripts: ScriptsConfig = field(default_factory=ScriptsConfig)

    def __post_init__(self):
        if self.name not in VALID_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework: '{self.name}'. Valid: {VALID_FRAMEWORKS}"
            )
        if not self.parallelism:
            raise ValueError(
                f"Framework '{self.name}' must define at least one parallelism"
            )
        # validate each spec
        for p_name, spec in self.parallelism.items():
            if not isinstance(spec, ParallelismSpec):
                raise ValueError(
                    f"Framework '{self.name}' parallelism '{p_name}' "
                    f"must be a ParallelismSpec, got {type(spec)}"
                )

    def supports_parallelism(self, name: str) -> bool:
        return name in self.parallelism

    def get_parallelism_spec(self, name: str) -> Optional[ParallelismSpec]:
        return self.parallelism.get(name)
