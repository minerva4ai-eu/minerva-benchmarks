from abc import ABC, abstractmethod
from dataclasses import dataclass

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig


@dataclass
class RuleResult:
    passed: bool
    rule_name: str
    reason: str = ""


class ConstraintRule(ABC):
    @abstractmethod
    def check(self, c: BenchmarkConfig) -> RuleResult:
        raise NotImplementedError
