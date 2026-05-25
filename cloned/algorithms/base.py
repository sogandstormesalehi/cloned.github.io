from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from cloned.spaces.base import Candidate, SearchSpace
from cloned.generators.base import ImageGenerator
from cloned.rewards.base import Reward


@dataclass
class SearchResult:
    best_candidate: Candidate
    best_prompt: str
    best_score: float
    history_best: List[float]
    history_overall: List[float]
    history_gmean: List[float] = field(default_factory=list) 


class SearchAlgorithm(ABC):
    @abstractmethod
    def run(
        self,
        space: SearchSpace,
        generator: ImageGenerator,
        reward: Reward,
        seed: int,
        out_dir: str | None = None,
    ) -> SearchResult:
        ...