from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from cloned.spaces.base import SearchSpace, Candidate
from cloned.generators.base import ImageGenerator
from cloned.rewards.base import Reward


class EvaluationManager:
    def __init__(self, image_embedder=None, surrogate_size: Optional[int] = None):
        self.reward_cache:    Dict[str, List[float]]  = {}
        self.img_cache:       Dict[str, Image.Image]  = {}
        self.surrogate_img_cache: Dict[str, Image.Image] = {}
        self.vision_emb_cache: Dict[str, np.ndarray] = {}
        self.ind_by_prompt:   Dict[str, List[int]]    = {}
        self._bb_call_log:    List[dict]              = []
        self.pred_error_log: List[dict] = []

        self._image_embedder = image_embedder
        self._surrogate_size = surrogate_size

    def log_prediction_errors(
        self,
        prompts: List[str],
        predicted_scores: np.ndarray,
    ) -> None:
        for p, pred in zip(prompts, predicted_scores):
            if p not in self.reward_cache:
                continue
            actual = float(np.mean(self.reward_cache[p]))
            self.pred_error_log.append({
                "prompt":     p,
                "predicted":  float(pred),
                "actual":     actual,
                "error":      float(pred) - actual,
                "eval_index": len(self.pred_error_log),
            })

    def recent_mean_prediction_error(self, window: int = 10) -> float:
        if len(self.pred_error_log) < 2:
            return 0.0
        recent = self.pred_error_log[-window:]
        return float(np.mean([e["error"] for e in recent]))

    def _cache_image(self, prompt: str, img: Image.Image) -> None:
        self.img_cache[prompt] = img
        if self._surrogate_size is not None:
            size = self._surrogate_size
            small = img.resize((size, size), Image.BICUBIC)
            self.surrogate_img_cache[prompt] = small
        else:
            self.surrogate_img_cache[prompt] = img

    @property
    def eval_count(self) -> int:
        return sum(len(v) for v in self.reward_cache.values())

    @property
    def bb_call_log(self) -> List[dict]:
        return self._bb_call_log

    def register_ind(self, ind: List[int], prompt: str) -> None:
        self.ind_by_prompt[prompt] = list(ind)

    def evaluate(
        self,
        space: SearchSpace,
        generator: ImageGenerator,
        rewarder: Reward,
        inds: Sequence[Sequence[int]],
        *,
        batch_size: int,
        seed: int,
    ) -> List[float]:
        prompts = [space.decode(Candidate(tuple(ind))) for ind in inds]

        for ind, p in zip(inds, prompts):
            self.ind_by_prompt[p] = list(ind)

        to_gen = [p for p in prompts if p not in self.img_cache]
        if to_gen:
            imgs: List[Image.Image] = []
            for i in range(0, len(to_gen), batch_size):
                imgs.extend(generator.generate(to_gen[i : i + batch_size], seed=seed))
            for p, im in zip(to_gen, imgs):
                self._cache_image(p, im)

            if self._image_embedder is not None:
                new_imgs   = [self.surrogate_img_cache[p] for p in to_gen]
                vision_embs = self._image_embedder.image_numpy(new_imgs)
                for p, emb in zip(to_gen, vision_embs):
                    self.vision_emb_cache[p] = emb.astype(np.float32)

        before_counts = {p: len(v) for p, v in self.reward_cache.items()}

        scores = []
        for p in prompts:
            im = self.img_cache[p]
            s = float(rewarder.score([im])[0])
            self.reward_cache.setdefault(p, []).append(s)
            scores.append(float(np.mean(self.reward_cache[p])))

        self._log_new_calls(before_counts)
        return scores

    def get_vision_embs_and_rewards(self) -> tuple[np.ndarray, np.ndarray, List[str]]:
        prompts, embs, y = [], [], []
        for p, vals in self.reward_cache.items():
            if p in self.vision_emb_cache:
                prompts.append(p)
                embs.append(self.vision_emb_cache[p])
                y.append(float(np.mean(vals)))
        if not embs:
            return np.empty((0, 0), np.float32), np.empty((0,), np.float32), []
        return np.stack(embs).astype(np.float32), np.array(y, np.float32), prompts

    def get_recent_vision_embs_and_rewards(
        self, since_eval: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        recent_prompts = [
            entry["prompt"]
            for entry in self._bb_call_log[since_eval:]
            if entry["prompt"] in self.vision_emb_cache
        ]
        seen_p: set = set()
        unique_recent = []
        for p in recent_prompts:
            if p not in seen_p:
                seen_p.add(p)
                unique_recent.append(p)
        if not unique_recent:
            return None, None
        embs = np.stack([self.vision_emb_cache[p] for p in unique_recent]).astype(np.float32)
        y    = np.array([float(np.mean(self.reward_cache[p])) for p in unique_recent], np.float32)
        return embs, y

    def get_training_data(self, space: SearchSpace):
        X, y = [], []
        for p, vals in self.reward_cache.items():
            if p in self.ind_by_prompt:
                X.append(space.featurize(self.ind_by_prompt[p]))
                y.append(float(np.mean(vals)))
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    def top_anchors(self, k: int) -> List[List[int]]:
        items = sorted(
            ((p, float(np.mean(v))) for p, v in self.reward_cache.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            self.ind_by_prompt[p]
            for p, _ in items[:k]
            if p in self.ind_by_prompt
        ]

    def _log_new_calls(self, before_counts: Dict[str, int]) -> None:
        for p, vals in self.reward_cache.items():
            prev = before_counts.get(p, 0)
            for j in range(prev, len(vals)):
                self._bb_call_log.append(
                    {
                        "call_index":         len(self._bb_call_log),
                        "prompt":             p,
                        "genes":              [int(g) for g in self.ind_by_prompt.get(p, [])],
                        "raw_score":          float(vals[j]),
                        "cached_mean_score":  float(np.mean(vals[: j + 1])),
                        "n_evals_for_prompt": j + 1,
                        "has_vision_emb":     p in self.vision_emb_cache,
                    }
                )

    def save_logs(self, out_dir: str, surrogate=None) -> None:
        with open(os.path.join(out_dir, "blackbox_calls.json"), "w") as f:
            json.dump(self._bb_call_log, f, indent=2)
        if self._bb_call_log:
            np.save(
                os.path.join(out_dir, "blackbox_raw_scores.npy"),
                np.asarray([x["raw_score"] for x in self._bb_call_log], dtype=np.float32),
            )
        if surrogate is not None and surrogate.fit_log:
            with open(os.path.join(out_dir, "surrogate_fit_log.json"), "w") as f:
                json.dump(surrogate.fit_log, f, indent=2)

        if self.pred_error_log:
            with open(os.path.join(out_dir, "pred_error_log.json"), "w") as f:
                json.dump(self.pred_error_log, f, indent=2)
            np.save(
                os.path.join(out_dir, "pred_errors.npy"),
                np.array([e["error"] for e in self.pred_error_log], dtype=np.float32),
            )

        if self.vision_emb_cache:
            prompts_with_embs = list(self.vision_emb_cache.keys())
            emb_matrix = np.stack([self.vision_emb_cache[p] for p in prompts_with_embs])
            np.save(os.path.join(out_dir, "vision_embs.npy"), emb_matrix)