from abc import ABC, abstractmethod
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class RuleResult:
    passed: bool
    rule_name: str
    reason: str = ""


class ConstraintRule(ABC):
    @abstractmethod
    def check(self, c: DictConfig) -> RuleResult:
        raise NotImplementedError
