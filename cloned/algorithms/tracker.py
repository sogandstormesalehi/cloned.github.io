from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class StagnationTracker:
    def __init__(self, patience: int = 3, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = -1e18
        self._stagnant_iters: int = 0

    @property
    def stuck(self) -> bool:
        return self._stagnant_iters >= self.patience

    def update(self, best_score: float) -> None:
        if best_score > self._best + self.min_delta:
            self._best = best_score
            self._stagnant_iters = 0
        else:
            self._stagnant_iters += 1

    def reset(self) -> None:
        self._best = -1e18
        self._stagnant_iters = 0


def best_prompt_from_cache(reward_cache: Dict[str, List[float]]) -> Tuple[Optional[str], Optional[float]]:
    best_p, best_s = None, None
    for p, vals in reward_cache.items():
        m = float(np.mean(vals))
        if best_s is None or m > best_s:
            best_s, best_p = m, p
    return best_p, best_s


def global_mean_reward(reward_cache: Dict[str, List[float]]) -> float:
    all_vals = [v for vals in reward_cache.values() for v in vals]
    return float(np.mean(all_vals)) if all_vals else float("nan")