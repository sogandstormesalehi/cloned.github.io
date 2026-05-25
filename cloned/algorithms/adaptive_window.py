from __future__ import annotations

from typing import List, Optional
import numpy as np


class AdaptiveWindowController:

    def __init__(
        self,
        max_window:       int   = 200,
        min_window:       int   = 20,
        shrink_factor:    float = 0.6,
        grow_factor:      float = 1.15,
        buffer_size:      int   = 5,
        drift_threshold:  float = 0.5,
        stable_threshold: float = 0.0,
        stable_patience:  int   = 3,
        eps:              float = 1e-6,
    ):
        self.max_window       = max_window
        self.min_window       = min_window
        self.shrink_factor    = shrink_factor
        self.grow_factor      = grow_factor
        self.buffer_size      = buffer_size
        self.drift_threshold  = drift_threshold
        self.stable_threshold = stable_threshold
        self.stable_patience  = stable_patience
        self.eps              = eps

        self._window          = float(max_window)
        self._buffer:  List[float] = []
        self._stable_count    = 0
        self.log: List[dict]  = []   # one entry per generation

    @property
    def current_window(self) -> Optional[int]:
        w = int(round(self._window))
        return None if w >= self.max_window else w

    def update(self, gen_mean: float) -> bool:
        if np.isnan(gen_mean):
            return False

        self._buffer.append(gen_mean)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        if len(self._buffer) < 3:
            self.log.append({
                "gen_mean": gen_mean, "drift_score": None,
                "window": self._window, "action": "warmup",
            })
            return False

        arr   = np.array(self._buffer, dtype=np.float32)
        x     = np.arange(len(arr), dtype=np.float32)
        xc    = x - x.mean()
        denom = (xc ** 2).sum()
        slope = float((xc * (arr - arr.mean())).sum() / denom) \
                if denom > 1e-8 else 0.0
        var   = float(arr.var())

        drift_score = -slope / (var + self.eps)

        force_explore = False
        action        = "hold"

        if drift_score > self.drift_threshold:
            new_w         = max(float(self.min_window),
                                self._window * self.shrink_factor)
            action        = f"shrink {self._window:.0f}→{new_w:.0f}"
            self._window  = new_w
            self._stable_count = 0
            force_explore = True
        elif drift_score < self.stable_threshold:
            self._stable_count += 1
            if self._stable_count >= self.stable_patience:
                new_w        = min(float(self.max_window),
                                   self._window * self.grow_factor)
                action       = f"grow {self._window:.0f}→{new_w:.0f}"
                self._window = new_w
            else:
                action = f"stable ({self._stable_count}/{self.stable_patience})"
        else:
            self._stable_count = 0

        self.log.append({
            "gen_mean":    gen_mean,
            "slope":       slope,
            "var":         var,
            "drift_score": drift_score,
            "window":      int(round(self._window)),
            "action":      action,
            "force_explore": force_explore,
        })

        if force_explore:
            print(f"[AdaptiveWindow] drift_score={drift_score:.3f} > "
                  f"{self.drift_threshold} → {action}  "
                  f"force_explorer=True")
        elif "grow" in action or "shrink" in action:
            print(f"[AdaptiveWindow] drift_score={drift_score:.3f} → {action}")

        return force_explore