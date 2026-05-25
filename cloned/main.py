from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cloned.algorithms.drift_explorer import DriftExplorer
from cloned.algorithms.embedder import CLIPTextEmbedder
from cloned.algorithms.genetic_clip_text import CLIPTextGeneticSearch
from cloned.generators.sdxl_turbo import SDXLTurboGenerator
from cloned.rewards.base import Reward
from cloned.rewards.blackbox_rewards import BERGFMRINSDFsaverageHuzeROIReward
from cloned.rewards.drift_reward import sigmoid_weight
from cloned.spaces.base import Candidate
from cloned.spaces.structured_art_space import StructuredArtPromptSpace, art_data, flatten_art_data
from cloned.utils.seed import set_all_seeds

BERG_DIR         = "./berg_models"
CALIBRATION_PATH = "./search/clean_search/calibration/fmri_calibration_subj1.json"
OUT_DIR          = "./cloned/results/"
CLUSTER_PATH     = "./search/real_time_search/AdaptivePersonalization/prompt_clusters.npz"
DEVICE           = "cuda"
ORACLE_SUBJECT   = 1

EXPLORER_COOLDOWN      = 20
EXPLORER_MIN_EVALS     = 60
EXPLORER_N_CLUSTERS    = 3
EXPLORER_N_PER_CLUSTER = 3
SLOPE_WINDOW           = 20
SLOPE_THRESHOLD        = -0.005

DRIFT_STEEPNESS = 6.0


ONE_ROI_CONFIGS = [
    {"max_evals": 100},
    {"max_evals": 150},
]

TWO_ROI_CONFIGS = [
    {"max_evals": 150, "midpoint": 0.60},
    {"max_evals": 200, "midpoint": 0.60},
]

THREE_ROI_CONFIGS = [
    {"max_evals": 250, "drift_1": 0.40, "drift_2": 0.70},
    {"max_evals": 300, "drift_1": 0.40, "drift_2": 0.70},
]

FOUR_ROI_CONFIGS = [
    {"max_evals": 300, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78},
    {"max_evals": 400, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78},
]

ONE_ROI_TARGETS: List[str] = [
    "V1v", "OFA", "FFA-1", "PPA", "EBA", "OPA",
]

TWO_ROI_PAIRS: List[Tuple[str, str]] = [
    ("FFA-1", "PPA"),
    ("OFA",   "FFA-1"),
    ("EBA",   "OFA"),
    ("V1v",   "PPA"),
    ("OPA",   "PPA"),
    ("EBA",   "PPA"),
    ("PPA",   "V1v"),
    ("PPA",   "OFA"),
]

THREE_ROI_TRIPLETS: List[Tuple[str, str, str]] = [
    ("V1v",   "OFA",   "FFA-1"),
    ("FFA-1", "PPA",   "V1v"),
    ("OFA",   "FFA-1", "PPA"),
]

FOUR_ROI_QUADS: List[Tuple[str, str, str, str]] = [
    ("V1v",  "OFA",   "FFA-1", "PPA"),
    ("OFA",  "FFA-1", "PPA",   "RSC"),
]

ROI_CAL_KEY = {
    "V1v":       "V1v",
    "PPA":       "PPA",
    "FFA-1":     "FFA-1",
    "EBA":       "EBA",
    "OFA":       "OFA",
    "OPA":       "OPA",
    "FBA-1":     "FBA-1",
    "RSC":       "RSC",
    "aTL-faces": "aTL-faces",
}

SEEDS  = list(range(40, 45))
N_SAVE = 10


_GENETIC_BASE = dict(
    population_size  = 20,
    n_init           = 20,
    mutation_rate    = 0.3,
    crossover_rate   = 0.5,
    elite_frac       = 0.2,
    n_images_to_save = N_SAVE,
    device           = DEVICE,
)

_SURROGATE_BASE = dict(
    surrogate_warmup     = 40,
    surrogate_window     = 40,
    surrogate_beta       = 0.2,
    surrogate_gamma      = 0.2,
    step_penalty         = 0.1,
    elite_window         = 20,
    use_state_descriptor = True,
    state_window         = 20,
    use_surrogate        = True,
)

_CLIP_BASE = dict(
    surrogate_pool_size = 1000,
    surrogate_k_eval    = 20,
    clip_name           = "openai/clip-vit-base-patch32",
    n_pca               = 50,
)

_SALIENCE_BASE = dict(
    use_salience_operators        = True,
    salience_ema_alpha            = 0.7,
    salience_update_every         = 1,
    salience_warmup               = 60,
    use_surrogate_salience        = False,
    semantic_mutation_temperature = 0.15,
)


class OneROIReward(Reward):
    """Single static target — no drift."""
    def __init__(self, reward_a, a_mean, a_std):
        self.reward_a    = reward_a
        self.a_mean      = a_mean
        self.a_std       = max(a_std, 1e-6)
        self._eval_count = 0
        self.raw_log: List[dict] = []

    @classmethod
    def from_calibration(cls, calibration_path, reward_a, roi_a_key):
        with open(calibration_path) as f:
            rois = json.load(f)["rois"]
        return cls(reward_a,
                   rois[roi_a_key]["mean"],
                   rois[roi_a_key]["std"])

    def score(self, images):
        raw_a  = np.asarray(self.reward_a.score(images), dtype=np.float32)
        norm_a = (raw_a - self.a_mean) / self.a_std
        for i in range(len(images)):
            self.raw_log.append({
                "eval_idx": self._eval_count,
                "raw_a":    float(raw_a[i]),
                "norm_a":   float(norm_a[i]),
            })
            self._eval_count += 1
        return norm_a.tolist()

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.raw_log, f, indent=2)

    def get_arrays(self):
        if not self.raw_log:
            e = np.array([], dtype=np.float32)
            return e, e
        def c(k):
            return np.array([e[k] for e in self.raw_log], dtype=np.float32)
        return c("eval_idx"), c("raw_a")


class TwoROIDriftReward(Reward):
    def __init__(self, reward_a, reward_b, a_mean, a_std, b_mean, b_std,
                 max_evals, midpoint=0.60, steepness=6.0):
        self.reward_a    = reward_a
        self.reward_b    = reward_b
        self.a_mean      = a_mean
        self.a_std       = max(a_std, 1e-6)
        self.b_mean      = b_mean
        self.b_std       = max(b_std, 1e-6)
        self.max_evals   = max_evals
        self.midpoint    = midpoint
        self.steepness   = steepness
        self._eval_count = 0
        self.raw_log: List[dict] = []

    @classmethod
    def from_calibration(cls, calibration_path, reward_a, reward_b,
                         roi_a_key, roi_b_key, max_evals, midpoint=0.60, steepness=6.0):
        with open(calibration_path) as f:
            rois = json.load(f)["rois"]
        return cls(reward_a, reward_b,
                   rois[roi_a_key]["mean"], rois[roi_a_key]["std"],
                   rois[roi_b_key]["mean"], rois[roi_b_key]["std"],
                   max_evals, midpoint, steepness)

    def score(self, images):
        raw_a  = np.asarray(self.reward_a.score(images), dtype=np.float32)
        raw_b  = np.asarray(self.reward_b.score(images), dtype=np.float32)
        norm_a = (raw_a - self.a_mean) / self.a_std
        norm_b = (raw_b - self.b_mean) / self.b_std
        out    = np.zeros(len(images), dtype=np.float32)
        for i in range(len(images)):
            t      = min(self._eval_count / max(self.max_evals - 1, 1), 1.0)
            w_b    = sigmoid_weight(t, self.steepness, self.midpoint)
            out[i] = (1.0 - w_b) * norm_a[i] + w_b * norm_b[i]
            self.raw_log.append({
                "eval_idx": self._eval_count, "t": float(t), "w_b": float(w_b),
                "raw_a": float(raw_a[i]), "raw_b": float(raw_b[i]),
                "blended": float(out[i]),
            })
            self._eval_count += 1
        return out.tolist()

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.raw_log, f, indent=2)

    def get_arrays(self):
        if not self.raw_log:
            e = np.array([], dtype=np.float32)
            return e, e, e, e, e
        def c(k):
            return np.array([e[k] for e in self.raw_log], dtype=np.float32)
        return c("eval_idx"), c("w_b"), c("raw_a"), c("raw_b"), c("blended")

    def plot(self, roi_a, roi_b, save_path=None, title=""):
        if not self.raw_log:
            return
        idxs, w_b, raw_a, raw_b, blended = self.get_arrays()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(idxs, w_b, color="purple", lw=2)
        axes[0].set_ylabel("w_B"); axes[0].set_title(title)
        axes[1].plot(idxs, raw_a, color="steelblue",  lw=1.5, label=roi_a)
        axes[1].plot(idxs, raw_b, color="darkorange", lw=1.5, label=roi_b)
        axes[1].set_ylabel("Raw score"); axes[1].legend(fontsize=8)
        axes[2].plot(idxs, blended, color="green", lw=2)
        axes[2].set_ylabel("Blended"); axes[2].set_xlabel("Eval")
        for ax in axes:
            ax.axvline(self.midpoint * self.max_evals, color="purple", ls="--", lw=1, alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.close(fig)


class ThreeROIDriftReward(Reward):
    def __init__(self, reward_a, reward_b, reward_c,
                 a_mean, a_std, b_mean, b_std, c_mean, c_std,
                 max_evals, drift_1=0.40, drift_2=0.70, steepness=6.0):
        self.reward_a    = reward_a
        self.reward_b    = reward_b
        self.reward_c    = reward_c
        self.a_mean      = a_mean;  self.a_std = max(a_std, 1e-6)
        self.b_mean      = b_mean;  self.b_std = max(b_std, 1e-6)
        self.c_mean      = c_mean;  self.c_std = max(c_std, 1e-6)
        self.max_evals   = max_evals
        self.drift_1     = drift_1
        self.drift_2     = drift_2
        self.steepness   = steepness
        self._eval_count = 0
        self.raw_log: List[dict] = []

    @classmethod
    def from_calibration(cls, calibration_path, reward_a, reward_b, reward_c,
                         roi_a_key, roi_b_key, roi_c_key, max_evals,
                         drift_1=0.40, drift_2=0.70, steepness=6.0):
        with open(calibration_path) as f:
            rois = json.load(f)["rois"]
        return cls(reward_a, reward_b, reward_c,
                   rois[roi_a_key]["mean"], rois[roi_a_key]["std"],
                   rois[roi_b_key]["mean"], rois[roi_b_key]["std"],
                   rois[roi_c_key]["mean"], rois[roi_c_key]["std"],
                   max_evals, drift_1, drift_2, steepness)

    def _weights(self, t):
        s1 = sigmoid_weight(t, self.steepness, self.drift_1)
        s2 = sigmoid_weight(t, self.steepness, self.drift_2)
        return 1.0 - s1, s1 * (1.0 - s2), s1 * s2

    def score(self, images):
        raw_a  = np.asarray(self.reward_a.score(images), dtype=np.float32)
        raw_b  = np.asarray(self.reward_b.score(images), dtype=np.float32)
        raw_c  = np.asarray(self.reward_c.score(images), dtype=np.float32)
        norm_a = (raw_a - self.a_mean) / self.a_std
        norm_b = (raw_b - self.b_mean) / self.b_std
        norm_c = (raw_c - self.c_mean) / self.c_std
        out    = np.zeros(len(images), dtype=np.float32)
        for i in range(len(images)):
            t             = min(self._eval_count / max(self.max_evals - 1, 1), 1.0)
            w_a, w_b, w_c = self._weights(t)
            out[i]        = w_a * norm_a[i] + w_b * norm_b[i] + w_c * norm_c[i]
            self.raw_log.append({
                "eval_idx": self._eval_count, "t": float(t),
                "w_a": float(w_a), "w_b": float(w_b), "w_c": float(w_c),
                "raw_a": float(raw_a[i]), "raw_b": float(raw_b[i]), "raw_c": float(raw_c[i]),
                "blended": float(out[i]),
            })
            self._eval_count += 1
        return out.tolist()

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.raw_log, f, indent=2)

    def get_arrays(self):
        if not self.raw_log:
            e = np.array([], dtype=np.float32)
            return e, e, e, e, e, e, e, e
        def c(k):
            return np.array([e[k] for e in self.raw_log], dtype=np.float32)
        return (c("eval_idx"), c("w_a"), c("w_b"), c("w_c"),
                c("raw_a"), c("raw_b"), c("raw_c"), c("blended"))

    def plot(self, roi_a, roi_b, roi_c, save_path=None, title=""):
        if not self.raw_log:
            return
        idxs, w_a, w_b, w_c, raw_a, raw_b, raw_c, blended = self.get_arrays()
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].stackplot(idxs, w_a, w_b, w_c, labels=[roi_a, roi_b, roi_c],
                          colors=["steelblue", "darkorange", "forestgreen"], alpha=0.7)
        axes[0].set_ylabel("ROI weight"); axes[0].set_title(title)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[1].plot(idxs, raw_a, color="steelblue",   lw=1.2, label=f"raw {roi_a}")
        axes[1].plot(idxs, raw_b, color="darkorange",  lw=1.2, label=f"raw {roi_b}")
        axes[1].plot(idxs, raw_c, color="forestgreen", lw=1.2, label=f"raw {roi_c}")
        axes[1].set_ylabel("Raw score"); axes[1].legend(fontsize=8)
        axes[2].plot(idxs, blended, color="green", lw=2)
        axes[2].set_ylabel("Blended"); axes[2].set_xlabel("Eval")
        for ax in axes:
            for t in [self.drift_1, self.drift_2]:
                ax.axvline(t * self.max_evals, color="purple", ls="--", lw=1, alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.close(fig)


class FourROIDriftReward(Reward):
    def __init__(self, reward_a, reward_b, reward_c, reward_d,
                 a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std,
                 max_evals, drift_1=0.33, drift_2=0.56, drift_3=0.78, steepness=6.0):
        self.reward_a    = reward_a
        self.reward_b    = reward_b
        self.reward_c    = reward_c
        self.reward_d    = reward_d
        self.a_mean      = a_mean;  self.a_std = max(a_std, 1e-6)
        self.b_mean      = b_mean;  self.b_std = max(b_std, 1e-6)
        self.c_mean      = c_mean;  self.c_std = max(c_std, 1e-6)
        self.d_mean      = d_mean;  self.d_std = max(d_std, 1e-6)
        self.max_evals   = max_evals
        self.drift_1     = drift_1
        self.drift_2     = drift_2
        self.drift_3     = drift_3
        self.steepness   = steepness
        self._eval_count = 0
        self.raw_log: List[dict] = []

    @classmethod
    def from_calibration(cls, calibration_path,
                         reward_a, reward_b, reward_c, reward_d,
                         roi_a_key, roi_b_key, roi_c_key, roi_d_key,
                         max_evals, drift_1=0.33, drift_2=0.56, drift_3=0.78,
                         steepness=6.0):
        with open(calibration_path) as f:
            rois = json.load(f)["rois"]
        return cls(reward_a, reward_b, reward_c, reward_d,
                   rois[roi_a_key]["mean"], rois[roi_a_key]["std"],
                   rois[roi_b_key]["mean"], rois[roi_b_key]["std"],
                   rois[roi_c_key]["mean"], rois[roi_c_key]["std"],
                   rois[roi_d_key]["mean"], rois[roi_d_key]["std"],
                   max_evals, drift_1, drift_2, drift_3, steepness)

    def _weights(self, t):
        s1 = sigmoid_weight(t, self.steepness, self.drift_1)
        s2 = sigmoid_weight(t, self.steepness, self.drift_2)
        s3 = sigmoid_weight(t, self.steepness, self.drift_3)
        return 1.0 - s1, s1 * (1.0 - s2), s2 * (1.0 - s3), s3

    def score(self, images):
        raw_a  = np.asarray(self.reward_a.score(images), dtype=np.float32)
        raw_b  = np.asarray(self.reward_b.score(images), dtype=np.float32)
        raw_c  = np.asarray(self.reward_c.score(images), dtype=np.float32)
        raw_d  = np.asarray(self.reward_d.score(images), dtype=np.float32)
        norm_a = (raw_a - self.a_mean) / self.a_std
        norm_b = (raw_b - self.b_mean) / self.b_std
        norm_c = (raw_c - self.c_mean) / self.c_std
        norm_d = (raw_d - self.d_mean) / self.d_std
        out    = np.zeros(len(images), dtype=np.float32)
        for i in range(len(images)):
            t                  = min(self._eval_count / max(self.max_evals - 1, 1), 1.0)
            w_a, w_b, w_c, w_d = self._weights(t)
            out[i]             = w_a * norm_a[i] + w_b * norm_b[i] + w_c * norm_c[i] + w_d * norm_d[i]
            self.raw_log.append({
                "eval_idx": self._eval_count, "t": float(t),
                "w_a": float(w_a), "w_b": float(w_b), "w_c": float(w_c), "w_d": float(w_d),
                "raw_a": float(raw_a[i]), "raw_b": float(raw_b[i]),
                "raw_c": float(raw_c[i]), "raw_d": float(raw_d[i]),
                "blended": float(out[i]),
            })
            self._eval_count += 1
        return out.tolist()

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.raw_log, f, indent=2)

    def get_arrays(self):
        if not self.raw_log:
            e = np.array([], dtype=np.float32)
            return e, e, e, e, e, e, e, e, e, e
        def c(k):
            return np.array([e[k] for e in self.raw_log], dtype=np.float32)
        return (c("eval_idx"), c("w_a"), c("w_b"), c("w_c"), c("w_d"),
                c("raw_a"), c("raw_b"), c("raw_c"), c("raw_d"), c("blended"))

    def plot(self, roi_a, roi_b, roi_c, roi_d, save_path=None, title=""):
        if not self.raw_log:
            return
        idxs, w_a, w_b, w_c, w_d, raw_a, raw_b, raw_c, raw_d, blended = self.get_arrays()
        colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].stackplot(idxs, w_a, w_b, w_c, w_d,
                          labels=[roi_a, roi_b, roi_c, roi_d],
                          colors=colors, alpha=0.7)
        axes[0].set_ylabel("ROI weight"); axes[0].set_title(title)
        axes[0].legend(fontsize=8, loc="upper right")
        for raw, roi, col in [(raw_a, roi_a, colors[0]), (raw_b, roi_b, colors[1]),
                               (raw_c, roi_c, colors[2]), (raw_d, roi_d, colors[3])]:
            axes[1].plot(idxs, raw, color=col, lw=1.2, label=f"raw {roi}")
        axes[1].set_ylabel("Raw score"); axes[1].legend(fontsize=8)
        axes[2].plot(idxs, blended, color="green", lw=2)
        axes[2].set_ylabel("Blended"); axes[2].set_xlabel("Eval")
        for ax in axes:
            for t in [self.drift_1, self.drift_2, self.drift_3]:
                ax.axvline(t * self.max_evals, color="purple", ls="--", lw=1, alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.close(fig)


def _make_berg(roi: str) -> BERGFMRINSDFsaverageHuzeROIReward:
    return BERGFMRINSDFsaverageHuzeROIReward(
        berg_dir=BERG_DIR, subject=ORACLE_SUBJECT, roi=roi,
        device="cuda", pooling="topk_mean", topk_frac=0.1,
        combine_hemispheres="concat",
    )



def build_one_roi_reward(roi, berg_cache, cfg):
    return OneROIReward.from_calibration(
        CALIBRATION_PATH, berg_cache[roi], ROI_CAL_KEY[roi],
    )

def build_two_roi_drift(roi_a, roi_b, berg_cache, cfg):
    return TwoROIDriftReward.from_calibration(
        CALIBRATION_PATH, berg_cache[roi_a], berg_cache[roi_b],
        ROI_CAL_KEY[roi_a], ROI_CAL_KEY[roi_b],
        cfg["max_evals"], cfg["midpoint"], DRIFT_STEEPNESS,
    )

def build_three_roi_drift(roi_a, roi_b, roi_c, berg_cache, cfg):
    return ThreeROIDriftReward.from_calibration(
        CALIBRATION_PATH,
        berg_cache[roi_a], berg_cache[roi_b], berg_cache[roi_c],
        ROI_CAL_KEY[roi_a], ROI_CAL_KEY[roi_b], ROI_CAL_KEY[roi_c],
        cfg["max_evals"], cfg["drift_1"], cfg["drift_2"], DRIFT_STEEPNESS,
    )

def build_four_roi_drift(roi_a, roi_b, roi_c, roi_d, berg_cache, cfg):
    return FourROIDriftReward.from_calibration(
        CALIBRATION_PATH,
        berg_cache[roi_a], berg_cache[roi_b],
        berg_cache[roi_c], berg_cache[roi_d],
        ROI_CAL_KEY[roi_a], ROI_CAL_KEY[roi_b],
        ROI_CAL_KEY[roi_c], ROI_CAL_KEY[roi_d],
        cfg["max_evals"], cfg["drift_1"], cfg["drift_2"], cfg["drift_3"],
        DRIFT_STEEPNESS,
    )



def _build_explorer(seed, clip_embedder):
    return DriftExplorer.from_npz(
        cluster_path      = CLUSTER_PATH,
        clip_embedder     = clip_embedder,
        cooldown          = EXPLORER_COOLDOWN,
        min_evals         = EXPLORER_MIN_EVALS,
        n_clusters        = EXPLORER_N_CLUSTERS,
        n_per_cluster     = EXPLORER_N_PER_CLUSTER,
        seed              = seed,
        use_slope_trigger = True,
        slope_window      = SLOPE_WINDOW,
        slope_threshold   = SLOPE_THRESHOLD,
        use_lru_injection = False,
    )

def _needs_explorer(algo_name: str) -> bool:
    return any(k in algo_name for k in ("explorer", "adaptive", "salience"))



def _build_algos(max_evals, device, norm_option_embs):
    base = dict(**_GENETIC_BASE, max_evals=max_evals)

    genetic_clip_stateless = CLIPTextGeneticSearch(
        **base, **_CLIP_BASE,
        use_surrogate=True, surrogate_warmup=40,
        surrogate_beta=0.2, surrogate_gamma=0.2, step_penalty=0.1,
        surrogate_window=None, elite_window=None,
        use_state_descriptor=False, state_window=20,
        use_salience_operators=False,
    )

    genetic_clip_explorer = CLIPTextGeneticSearch(
        **base, **_SURROGATE_BASE, **_CLIP_BASE,
        use_salience_operators=False,
    )

    genetic_clip_adaptive_salience = CLIPTextGeneticSearch(
        **base, **_CLIP_BASE,
        use_surrogate=True, surrogate_warmup=40,
        surrogate_beta=0.2, surrogate_gamma=0.2, step_penalty=0.1,
        use_state_descriptor=True, state_window=20,
        surrogate_window=None, elite_window=None,
        adaptive_window=True,
        adaptive_window_max=100,
        adaptive_window_min=20,
        adaptive_window_shrink=0.6,
        adaptive_window_grow=1.15,
        adaptive_window_buffer=5,
        adaptive_drift_threshold=0.1,
        adaptive_stable_threshold=0.0,
        adaptive_stable_patience=3,
        **_SALIENCE_BASE,
        clip_option_embeddings=norm_option_embs,
    )

    return {
        "genetic_clip_stateless":         genetic_clip_stateless,
        "genetic_clip_explorer":          genetic_clip_explorer,
        "genetic_clip_adaptive_salience": genetic_clip_adaptive_salience,
    }


def _save_common(seed_dir, result):
    np.save(os.path.join(seed_dir, "history_best.npy"),
            np.asarray(result.history_best, np.float32))
    np.save(os.path.join(seed_dir, "history_gmean.npy"),
            np.asarray(result.history_gmean, np.float32))


def save_one_roi(seed_dir, result, reward, roi, algo_name, seed):
    _save_common(seed_dir, result)
    reward.save_log(os.path.join(seed_dir, "reward_log.json"))
    _, raw_a = reward.get_arrays()
    np.save(os.path.join(seed_dir, "raw_a.npy"), raw_a)
    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump({
            "best_score": float(result.best_score),
            "best_prompt": result.best_prompt,
            "roi": roi,
            "algo": algo_name, "seed": seed,
            "max_evals": len(raw_a),
            "n_rois": 1,
        }, f, indent=2)


def save_two_roi(seed_dir, result, reward, roi_a, roi_b, algo_name, seed):
    _save_common(seed_dir, result)
    reward.save_log(os.path.join(seed_dir, "drift_log.json"))
    idxs, w_b, raw_a, raw_b, blended = reward.get_arrays()
    for name, arr in [("raw_a", raw_a), ("raw_b", raw_b), ("w_b", w_b), ("blended", blended)]:
        np.save(os.path.join(seed_dir, f"{name}.npy"), arr)
    reward.plot(roi_a=roi_a, roi_b=roi_b,
                save_path=os.path.join(seed_dir, "drift_curves.png"),
                title=f"{roi_a}→{roi_b} | {algo_name} | seed {seed}")
    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump({
            "best_score": float(result.best_score),
            "best_prompt": result.best_prompt,
            "roi_a": roi_a, "roi_b": roi_b,
            "algo": algo_name, "seed": seed,
            "max_evals": reward.max_evals,
            "n_rois": 2,
        }, f, indent=2)


def save_three_roi(seed_dir, result, reward, roi_a, roi_b, roi_c, algo_name, seed):
    _save_common(seed_dir, result)
    reward.save_log(os.path.join(seed_dir, "drift_log.json"))
    idxs, w_a, w_b, w_c, raw_a, raw_b, raw_c, blended = reward.get_arrays()
    for name, arr in [("raw_a", raw_a), ("raw_b", raw_b), ("raw_c", raw_c),
                      ("w_a", w_a), ("w_b", w_b), ("w_c", w_c), ("blended", blended)]:
        np.save(os.path.join(seed_dir, f"{name}.npy"), arr)
    reward.plot(roi_a=roi_a, roi_b=roi_b, roi_c=roi_c,
                save_path=os.path.join(seed_dir, "drift_curves.png"),
                title=f"{roi_a}→{roi_b}→{roi_c} | {algo_name} | seed {seed}")
    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump({
            "best_score": float(result.best_score),
            "best_prompt": result.best_prompt,
            "roi_a": roi_a, "roi_b": roi_b, "roi_c": roi_c,
            "algo": algo_name, "seed": seed,
            "max_evals": reward.max_evals,
            "n_rois": 3,
        }, f, indent=2)


def save_four_roi(seed_dir, result, reward, roi_a, roi_b, roi_c, roi_d, algo_name, seed):
    _save_common(seed_dir, result)
    reward.save_log(os.path.join(seed_dir, "drift_log.json"))
    idxs, w_a, w_b, w_c, w_d, raw_a, raw_b, raw_c, raw_d, blended = reward.get_arrays()
    for name, arr in [("raw_a", raw_a), ("raw_b", raw_b), ("raw_c", raw_c), ("raw_d", raw_d),
                      ("w_a", w_a), ("w_b", w_b), ("w_c", w_c), ("w_d", w_d), ("blended", blended)]:
        np.save(os.path.join(seed_dir, f"{name}.npy"), arr)
    reward.plot(roi_a=roi_a, roi_b=roi_b, roi_c=roi_c, roi_d=roi_d,
                save_path=os.path.join(seed_dir, "drift_curves.png"),
                title=f"{roi_a}→{roi_b}→{roi_c}→{roi_d} | {algo_name} | seed {seed}")
    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump({
            "best_score": float(result.best_score),
            "best_prompt": result.best_prompt,
            "roi_a": roi_a, "roi_b": roi_b, "roi_c": roi_c, "roi_d": roi_d,
            "algo": algo_name, "seed": seed,
            "max_evals": reward.max_evals,
            "n_rois": 4,
        }, f, indent=2)


def _run_budget(rois_list, build_reward_fn, save_fn, cfg,
                space, generator, device, norm_option_embs,
                clip_embedder, berg_cache, base_dir):
    algos = _build_algos(cfg["max_evals"], device, norm_option_embs)
    for rois in rois_list:
        tag = "_".join(ROI_CAL_KEY[r] for r in rois)
        for algo_name, algo in algos.items():
            print(f"\n  evals={cfg['max_evals']} / {tag} / {algo_name}")
            for seed in SEEDS:
                seed_dir = os.path.join(base_dir, tag, algo_name,
                                        f"evals_{cfg['max_evals']}", f"seed_{seed:06d}")
                os.makedirs(seed_dir, exist_ok=True)
                if os.path.isfile(os.path.join(seed_dir, "result.json")):
                    print(f"    seed {seed}  [skip]")
                    continue
                print(f"    seed {seed} ──────")
                set_all_seeds(seed)
                reward   = build_reward_fn(*rois, berg_cache, cfg)
                explorer = _build_explorer(seed, clip_embedder) if _needs_explorer(algo_name) else None
                result   = algo.run(space=space, generator=generator,
                                    reward=reward, seed=seed, out_dir=seed_dir,
                                    drift_explorer=explorer)
                save_fn(seed_dir, result, reward, *rois, algo_name, seed)
                print(f"    best={result.best_score:.4f}")


def run_one_roi(space, generator, device, norm_option_embs, clip_embedder, berg_cache):
    print("\n" + "═" * 65)
    print("  EXPERIMENT 0 — One-ROI  (no drift)")
    print("═" * 65)
    base_dir = os.path.join(OUT_DIR, "one_roi")
    rois_list = [(roi,) for roi in ONE_ROI_TARGETS]
    for cfg in ONE_ROI_CONFIGS:
        _run_budget(rois_list, build_one_roi_reward, save_one_roi, cfg,
                    space, generator, device, norm_option_embs,
                    clip_embedder, berg_cache, base_dir)


def run_two_roi(space, generator, device, norm_option_embs, clip_embedder, berg_cache):
    print("\n" + "═" * 65)
    print("  EXPERIMENT 1 — Two-ROI drift")
    print("═" * 65)
    base_dir = os.path.join(OUT_DIR, "two_roi")
    for cfg in TWO_ROI_CONFIGS:
        _run_budget(TWO_ROI_PAIRS, build_two_roi_drift, save_two_roi, cfg,
                    space, generator, device, norm_option_embs,
                    clip_embedder, berg_cache, base_dir)


def run_three_roi(space, generator, device, norm_option_embs, clip_embedder, berg_cache):
    print("\n" + "═" * 65)
    print("  EXPERIMENT 2 — Three-ROI drift")
    print("═" * 65)
    base_dir = os.path.join(OUT_DIR, "three_roi")
    for cfg in THREE_ROI_CONFIGS:
        _run_budget(THREE_ROI_TRIPLETS, build_three_roi_drift, save_three_roi, cfg,
                    space, generator, device, norm_option_embs,
                    clip_embedder, berg_cache, base_dir)


def run_four_roi(space, generator, device, norm_option_embs, clip_embedder, berg_cache):
    print("\n" + "═" * 65)
    print("  EXPERIMENT 3 — Four-ROI drift")
    print("═" * 65)
    base_dir = os.path.join(OUT_DIR, "four_roi")
    for cfg in FOUR_ROI_CONFIGS:
        _run_budget(FOUR_ROI_QUADS, build_four_roi_drift, save_four_roi, cfg,
                    space, generator, device, norm_option_embs,
                    clip_embedder, berg_cache, base_dir)

def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"

    n_one   = len(ONE_ROI_TARGETS)    * len(ONE_ROI_CONFIGS)   * 3 * len(SEEDS)
    n_two   = len(TWO_ROI_PAIRS)      * len(TWO_ROI_CONFIGS)   * 3 * len(SEEDS)
    n_three = len(THREE_ROI_TRIPLETS) * len(THREE_ROI_CONFIGS) * 3 * len(SEEDS)
    n_four  = len(FOUR_ROI_QUADS)     * len(FOUR_ROI_CONFIGS)  * 3 * len(SEEDS)

    print(f"Device  : {device}")
    print(f"Seeds   : {SEEDS[0]}–{SEEDS[-1]}  (n={len(SEEDS)})")
    print(f"Output  : {OUT_DIR}")
    print()
    print(f"Exp 0 — One-ROI   : {len(ONE_ROI_TARGETS)} targets × {len(ONE_ROI_CONFIGS)} budgets × 3 algos × {len(SEEDS)} seeds = {n_one}")
    print(f"Exp 1 — Two-ROI   : {len(TWO_ROI_PAIRS)} pairs × {len(TWO_ROI_CONFIGS)} budgets × 3 algos × {len(SEEDS)} seeds = {n_two}")
    print(f"Exp 2 — Three-ROI : {len(THREE_ROI_TRIPLETS)} triplets × {len(THREE_ROI_CONFIGS)} budgets × 3 algos × {len(SEEDS)} seeds = {n_three}")
    print(f"Exp 3 — Four-ROI  : {len(FOUR_ROI_QUADS)} quads × {len(FOUR_ROI_CONFIGS)} budgets × 3 algos × {len(SEEDS)} seeds = {n_four}")
    print(f"Grand total: {n_one + n_two + n_three + n_four} runs")

    _, options    = flatten_art_data(art_data)
    clip_embedder = CLIPTextEmbedder(device=device)
    option_embs   = [torch.from_numpy(clip_embedder.batch_numpy(opts)) for opts in options]
    space         = StructuredArtPromptSpace(art_data=art_data, option_embeddings=option_embs)
    generator     = SDXLTurboGenerator(device=device, batch_size=32)

    norm_option_embs = []
    for emb in option_embs:
        e     = emb.numpy().astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        norm_option_embs.append(e / norms)

    all_rois = (
        set(ONE_ROI_TARGETS)
        | {roi for pair    in TWO_ROI_PAIRS      for roi in pair}
        | {roi for triplet in THREE_ROI_TRIPLETS for roi in triplet}
        | {roi for quad    in FOUR_ROI_QUADS     for roi in quad}
    )
    print(f"\nLoading BERG models: {sorted(all_rois)}")
    berg_cache = {roi: _make_berg(roi) for roi in all_rois}
    print("BERG models ready.\n")

    run_one_roi(  space, generator, device, norm_option_embs, clip_embedder, berg_cache)
    run_two_roi(  space, generator, device, norm_option_embs, clip_embedder, berg_cache)
    run_three_roi(space, generator, device, norm_option_embs, clip_embedder, berg_cache)
    run_four_roi( space, generator, device, norm_option_embs, clip_embedder, berg_cache)

    print(f"\nAll done → {OUT_DIR}")


if __name__ == "__main__":
    main()