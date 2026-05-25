from __future__ import annotations
import json
import numpy as np
from typing import List, Optional
from PIL import Image

from cloned.rewards.base import Reward



def sigmoid_weight(
    t: float,
    steepness: float = 10.0,
    midpoint: float = 0.75,
) -> float:
    return float(1.0 / (1.0 + np.exp(-steepness * (t - midpoint))))



class DriftReward(Reward):
    def __init__(
        self,
        reward_ffa: Reward,
        reward_ppa: Reward,
        ffa_mean: float,
        ffa_std: float,
        ppa_mean: float,
        ppa_std: float,
        max_evals: int,
        steepness: float = 10.0,
        midpoint: float = 0.75,
        log_raw: bool = True,
    ):
        self.reward_ffa = reward_ffa
        self.reward_ppa = reward_ppa
        self.ffa_mean   = ffa_mean
        self.ffa_std    = max(ffa_std, 1e-6)
        self.ppa_mean   = ppa_mean
        self.ppa_std    = max(ppa_std, 1e-6)
        self.max_evals  = max_evals
        self.steepness  = steepness
        self.midpoint   = midpoint
        self.log_raw    = log_raw

        self._eval_count = 0
        self.raw_log: List[dict] = []


    @classmethod
    def from_calibration(
        cls,
        calibration_path: str,
        reward_ffa: Reward,
        reward_ppa: Reward,
        max_evals: int,
        steepness: float = 10.0,
        midpoint: float = 0.75,
        log_raw: bool = True,
        stat_key_a: str = "ffa",
        stat_key_b: str = "ppa",
    ) -> "DriftReward":
        with open(calibration_path) as f:
            stats = json.load(f)

        print(
            f"[DriftReward] Loaded calibration from {calibration_path}\n"
            f"  {stat_key_a} : mean={stats[stat_key_a]['mean']:.4f}  "
            f"std={stats[stat_key_a]['std']:.4f}\n"
            f"  {stat_key_b} : mean={stats[stat_key_b]['mean']:.4f}  "
            f"std={stats[stat_key_b]['std']:.4f}\n"
            f"  n_samples={stats['n_samples']}  subject={stats['subject']}"
        )

        return cls(
            reward_ffa=reward_ffa,
            reward_ppa=reward_ppa,
            ffa_mean=stats[stat_key_a]["mean"],
            ffa_std=stats[stat_key_a]["std"],
            ppa_mean=stats[stat_key_b]["mean"],
            ppa_std=stats[stat_key_b]["std"],
            max_evals=max_evals,
            steepness=steepness,
            midpoint=midpoint,
            log_raw=log_raw,
        )


    def score(self, images: List[Image.Image]) -> List[float]:
        raw_ffa = np.asarray(self.reward_ffa.score(images), dtype=np.float32)
        raw_ppa = np.asarray(self.reward_ppa.score(images), dtype=np.float32)

        norm_ffa = (raw_ffa - self.ffa_mean) / self.ffa_std
        norm_ppa = (raw_ppa - self.ppa_mean) / self.ppa_std

        n = len(images)
        blended = np.zeros(n, dtype=np.float32)

        for i in range(n):
            t     = min(self._eval_count / max(self.max_evals - 1, 1), 1.0)
            w_ppa = sigmoid_weight(t, self.steepness, self.midpoint)
            blended[i] = (1.0 - w_ppa) * norm_ffa[i] + w_ppa * norm_ppa[i]

            if self.log_raw:
                self.raw_log.append({
                    "eval_idx":     self._eval_count,
                    "t":            float(t),
                    "w_ppa":        float(w_ppa),
                    "raw_ffa":      float(raw_ffa[i]),
                    "raw_ppa":      float(raw_ppa[i]),
                    "norm_ffa":     float(norm_ffa[i]),
                    "norm_ppa":     float(norm_ppa[i]),
                    "blended_norm": float(blended[i]),
                })

            self._eval_count += 1

        return blended.tolist()


    @property
    def current_w_ppa(self) -> float:
        t = min(self._eval_count / max(self.max_evals - 1, 1), 1.0)
        return sigmoid_weight(t, self.steepness, self.midpoint)

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def get_raw_arrays(self):
        if not self.raw_log:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty, empty, empty, empty, empty

        def col(k):
            return np.array([e[k] for e in self.raw_log], dtype=np.float32)

        return (
            col("eval_idx"),
            col("w_ppa"),
            col("raw_ffa"),
            col("raw_ppa"),
            col("norm_ffa"),
            col("norm_ppa"),
            col("blended_norm"),
        )

    def save_log(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.raw_log, f, indent=2)

    def plot(self, save_path: Optional[str] = None) -> None:
        import matplotlib.pyplot as plt

        idxs, w_ppa, raw_ffa, raw_ppa, norm_ffa, norm_ppa, blended = (
            self.get_raw_arrays()
        )
        if len(idxs) == 0:
            print("[DriftReward] Nothing to plot yet.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(idxs, w_ppa, color="purple", lw=2)
        axes[0].axhline(0.5, ls="--", color="gray", lw=0.8)
        axes[0].set_ylabel("w_PPA")
        axes[0].set_title("Drift schedule  (0 = pure FFA-1 → 1 = pure PPA)")
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].plot(idxs, raw_ffa, color="steelblue",  alpha=0.6, label="raw FFA-1")
        axes[1].plot(idxs, raw_ppa, color="darkorange", alpha=0.6, label="raw PPA")
        axes[1].set_ylabel("Raw ROI score")
        axes[1].legend(fontsize=8)

        axes[2].plot(idxs, norm_ffa, color="steelblue",  alpha=0.4, lw=1, label="norm FFA-1")
        axes[2].plot(idxs, norm_ppa, color="darkorange", alpha=0.4, lw=1, label="norm PPA")
        axes[2].plot(idxs, blended,  color="green",      lw=2,      label="blended")
        axes[2].axhline(0, ls="--", color="gray", lw=0.8)
        axes[2].set_ylabel("Normalized score")
        axes[2].set_xlabel("Oracle evaluation index")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
            print(f"[DriftReward] Plot saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)