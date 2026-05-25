from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence

@dataclass(frozen=True)
class Candidate:
    genes: Sequence[int]  

class SearchSpace(ABC):
    @abstractmethod
    def random_candidate(self) -> Candidate:
        ...

    @abstractmethod
    def decode(self, cand: Candidate) -> Any:
        ...

    @abstractmethod
    def mutate(self, cand: Candidate, mutation_rate: float) -> Candidate:
        ...

    @abstractmethod
    def crossover(self, a: Candidate, b: Candidate, crossover_rate: float) -> tuple[Candidate, Candidate]:
        ...