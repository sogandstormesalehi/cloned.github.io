from __future__ import annotations

import json
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from .evaluation import EvaluationManager


class _MCTSNode:
    __slots__ = ("visits", "total_gain")

    def __init__(self):
        self.visits     = 0
        self.total_gain = 0.0

    @property
    def mean_gain(self) -> float:
        return self.total_gain / self.visits if self.visits > 0 else 0.0


class DriftExplorer:

    def __init__(
        self,
        prompts: np.ndarray,
        labels: np.ndarray,
        centroids_pca: np.ndarray,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        clip_embedder,
        tree_node_count: Optional[int]       = None,
        tree_parent: Optional[np.ndarray]    = None,
        tree_level: Optional[np.ndarray]     = None,
        tree_centroid: Optional[np.ndarray]  = None,
        tree_children: Optional[np.ndarray]  = None,
        tree_leaf_prompts: Optional[np.ndarray] = None,
        trigger_mode: str = "prediction_error",
        injection_mode: str = "uncertainty",
        plateau_window: int = 25,
        plateau_rel_tol: float = 0.01,
        pred_error_window: int = 10,
        pred_error_threshold: float = 0.1,
        slope_window: int = 20,
        slope_threshold: float = -0.005,
        n_traversals: int = 3,     
        n_per_leaf: int = 3,     
        mcts_c: float = 1.0,         
        cooldown: int = 20,
        min_cooldown: int = 10,
        max_cooldown: int = 40,
        cooldown_growth: float = 1.4,  
        cooldown_decay: float  = 0.75,   
        adaptive_cooldown_delta: float = 0.01,
        min_evals: int = 60,
        seed: int = 0,
        use_slope_trigger: bool = False,
        use_lru_injection: bool = False,
        n_clusters: int = 3,        
        n_per_cluster: int = 3,       
    ):
        self.prompts        = prompts
        self.labels         = labels
        self.centroids_pca  = centroids_pca
        self.pca_mean       = pca_mean
        self.pca_components = pca_components
        self.clip_embedder  = clip_embedder

        if use_slope_trigger and trigger_mode == "prediction_error":
            trigger_mode = "slope"
        if use_lru_injection and injection_mode == "uncertainty":
            injection_mode = "lru"
        n_per_leaf = n_per_leaf or n_per_cluster

        self.trigger_mode            = trigger_mode
        self.injection_mode          = injection_mode
        self.plateau_window          = plateau_window
        self.plateau_rel_tol         = plateau_rel_tol
        self.pred_error_window       = pred_error_window
        self.pred_error_threshold    = pred_error_threshold
        self.slope_window            = slope_window
        self.slope_threshold         = slope_threshold

        self.n_traversals            = n_traversals
        self.n_per_leaf              = n_per_leaf
        self.mcts_c                  = mcts_c

        self._cooldown               = float(cooldown)
        self.min_cooldown            = min_cooldown
        self.max_cooldown            = max_cooldown
        self.cooldown_growth         = cooldown_growth
        self.cooldown_decay          = cooldown_decay
        self.adaptive_cooldown_delta = adaptive_cooldown_delta
        self.min_evals               = min_evals
        self._rng                    = np.random.default_rng(seed)

        self._last_trigger_eval    = -int(self._cooldown)
        self._best_at_last_trigger = float("-inf")
        self._total_triggers       = 0
        self._trigger_log: List[dict] = []
        self._emb_cache: dict[str, np.ndarray] = {}

        self._cluster_last_used = np.full(len(centroids_pca), -1, dtype=np.int32)

        self._has_tree = False

        if self._has_tree:
            self._tree_node_count   = int(tree_node_count)
            self._tree_parent       = tree_parent
            self._tree_level        = tree_level
            self._tree_centroid     = tree_centroid
            self._tree_children     = tree_children
            self._tree_leaf_prompts = tree_leaf_prompts
            self._mcts: List[_MCTSNode] = [
                _MCTSNode() for _ in range(self._tree_node_count)
            ]
            n_leaves = int((tree_level == 3).sum())
            print(f"[DriftExplorer] MCTS tree loaded: "
                  f"{self._tree_node_count} nodes, {n_leaves} leaves")
        else:
            print(f"[DriftExplorer] flat mode: {len(centroids_pca)} clusters")


    @classmethod
    def from_npz(
        cls,
        cluster_path: str,
        clip_embedder,
        trigger_mode: str = "prediction_error",
        injection_mode: str = "uncertainty",
        plateau_window: int = 25,
        plateau_rel_tol: float = 0.01,
        pred_error_window: int = 10,
        pred_error_threshold: float = 0.1,
        slope_window: int = 20,
        slope_threshold: float = -0.005,
        n_traversals: int = 3,
        n_per_leaf: int = 3,
        mcts_c: float = 1.0,
        cooldown: int = 20,
        min_cooldown: int = 10,
        max_cooldown: int = 40,
        cooldown_growth: float = 1.4,
        cooldown_decay: float  = 0.75,
        adaptive_cooldown_delta: float = 0.01,
        min_evals: int = 60,
        seed: int = 0,
        use_slope_trigger: bool = False,
        use_lru_injection: bool = False,
        n_clusters: int = 3,
        n_per_cluster: int = 3,
    ) -> "DriftExplorer":
        print(f"[DriftExplorer] loading clusters from {cluster_path}")
        d = np.load(cluster_path, allow_pickle=True)

        def _get(key):
            return d[key] if key in d else None

        obj = cls(
            prompts            = d["prompts"],
            labels             = d["labels"],
            centroids_pca      = d["centroids_pca"],
            pca_mean           = d["pca_mean"],
            pca_components     = d["pca_components"],
            clip_embedder      = clip_embedder,
            tree_node_count    = _get("tree_node_count"),
            tree_parent        = _get("tree_parent"),
            tree_level         = _get("tree_level"),
            tree_centroid      = _get("tree_centroid"),
            tree_children      = _get("tree_children"),
            tree_leaf_prompts  = _get("tree_leaf_prompts"),
            trigger_mode            = trigger_mode,
            injection_mode          = injection_mode,
            plateau_window          = plateau_window,
            plateau_rel_tol         = plateau_rel_tol,
            pred_error_window       = pred_error_window,
            pred_error_threshold    = pred_error_threshold,
            slope_window            = slope_window,
            slope_threshold         = slope_threshold,
            n_traversals            = n_traversals,
            n_per_leaf              = n_per_leaf,
            mcts_c                  = mcts_c,
            cooldown                = cooldown,
            min_cooldown            = min_cooldown,
            max_cooldown            = max_cooldown,
            cooldown_growth         = cooldown_growth,
            cooldown_decay          = cooldown_decay,
            adaptive_cooldown_delta = adaptive_cooldown_delta,
            min_evals               = min_evals,
            seed                    = seed,
            use_slope_trigger       = use_slope_trigger,
            use_lru_injection       = use_lru_injection,
            n_clusters              = n_clusters,
            n_per_cluster           = n_per_cluster,
        )
        return obj



    def step(
        self,
        history_best: List[float],
        evaluated_prompts: List[str],
        eval_count: int,
        history_gmean: Optional[List[float]] = None,
        evaluator=None,
        surrogate=None,
    ) -> List[str]:
        if eval_count < self.min_evals:
            return []
        if (eval_count - self._last_trigger_eval) < int(self._cooldown):
            return []

        triggered, detail = self._check_trigger(
            history_best, history_gmean, evaluator
        )
        if not triggered:
            return []

        self._update_cooldown(history_best)

        self._last_trigger_eval    = eval_count
        self._best_at_last_trigger = float(history_best[-1]) if history_best else 0.0
        self._total_triggers      += 1

        candidates, paths = self._inject(evaluated_prompts, eval_count, history_best, evaluator)

        self._trigger_log.append({
            "eval_count":       eval_count,
            "best_at_trigger":  self._best_at_last_trigger,
            "n_injected":       len(candidates),
            "trigger_mode":     self.trigger_mode,
            "trigger_detail":   detail,
            "mcts_paths":       [[int(n) for n in p] for p in paths],
            "current_cooldown": int(self._cooldown),
        })
        print(
            f"[DriftExplorer] {self.trigger_mode} @ eval {eval_count}  "
            f"best={self._best_at_last_trigger:.4f}  "
            f"inject={len(candidates)}  cooldown→{int(self._cooldown)}  "
            f"({detail})"
        )
        return candidates

    def backpropagate(self, reward_gain: float, paths: List[List[int]]) -> None:
        if not self._has_tree:
            return
        for path in paths:
            for nid in path:
                self._mcts[nid].visits     += 1
                self._mcts[nid].total_gain += reward_gain



    def _check_trigger(self, history_best, history_gmean, evaluator):
        if self.trigger_mode == "prediction_error":
            return self._trigger_pred_error(evaluator, history_best, history_gmean)
        elif self.trigger_mode == "slope":
            return self._trigger_slope(history_gmean or history_best)
        elif self.trigger_mode == "plateau":
            return self._trigger_plateau(history_best)
        raise ValueError(f"Unknown trigger_mode: {self.trigger_mode!r}")

    def _trigger_pred_error(self, evaluator, history_best, history_gmean):
        if evaluator is not None and len(evaluator.pred_error_log) >= self.pred_error_window:
            err    = evaluator.recent_mean_prediction_error(self.pred_error_window)
            detail = f"mean_pred_error={err:.4f} thr={self.pred_error_threshold:.4f}"
            return err > self.pred_error_threshold, detail
        triggered, detail = self._trigger_slope(history_gmean or history_best)
        return triggered, f"[fallback→slope] {detail}"

    def _trigger_slope(self, signal):
        clean = [v for v in signal[-self.slope_window:] if not np.isnan(v)]
        if len(clean) < 4:
            return False, "insufficient data"
        arr   = np.array(clean, dtype=np.float32)
        x     = np.arange(len(arr), dtype=np.float32)
        xc    = x - x.mean()
        denom = (xc**2).sum()
        slope = float((xc*(arr-arr.mean())).sum()/denom) if denom > 1e-8 else 0.0
        return slope < self.slope_threshold, f"slope={slope:.5f} thr={self.slope_threshold}"

    def _trigger_plateau(self, history_best):
        if len(history_best) < self.plateau_window:
            return False, "insufficient data"
        w   = history_best[-self.plateau_window:]
        rng = max(w) - min(w)
        tol = self.plateau_rel_tol * (abs(np.mean(w)) + 1e-8)
        return rng < tol, f"range={rng:.5f} tol={tol:.5f}"


    def _update_cooldown(self, history_best: List[float]) -> None:
        if not history_best:
            return
        improved = (float(history_best[-1]) - self._best_at_last_trigger) \
                   > self.adaptive_cooldown_delta
        if improved:
            self._cooldown = min(self.max_cooldown,
                                 self._cooldown * self.cooldown_growth)
        else:
            self._cooldown = max(self.min_cooldown,
                                 self._cooldown * self.cooldown_decay)



    def _inject(
        self,
        evaluated_prompts: List[str],
        eval_count: int,
        history_best: Optional[List[float]],
        evaluator,
    ) -> tuple[List[str], List[List[int]]]:
        exclude = set(evaluated_prompts)

        if self._has_tree and self.injection_mode == "mcts":
            return self._mcts_inject(exclude, eval_count)
        else:
            candidates = self._flat_inject(evaluated_prompts, eval_count,
                                           history_best, evaluator)
            return candidates, []

    def _mcts_inject(
        self,
        exclude: set,
        eval_count: int,
    ) -> tuple[List[str], List[List[int]]]:
        candidates: List[str] = []
        paths: List[List[int]] = []
        used_leaves: set[int] = set()   
        
        for _ in range(self.n_traversals):
            path = self._ucb1_traverse(used_leaves)
            if path is None:
                break
            leaf_nid = path[-1]
            used_leaves.add(leaf_nid)

            prompt_idxs = self._tree_leaf_prompts[leaf_nid]
            available   = [i for i in prompt_idxs
                           if str(self.prompts[i]) not in exclude]
            if not available:
                continue
            k       = min(self.n_per_leaf, len(available))
            sampled = self._rng.choice(available, size=k, replace=False)
            candidates.extend([str(self.prompts[i]) for i in sampled])
            paths.append(path)

        return candidates, paths

    def _ucb1_traverse(self, used_leaves: set) -> Optional[List[int]]:
        path = [0]   
        node = 0

        while True:
            children = self._tree_children[node]
            if len(children) == 0:
                # Leaf node
                if node in used_leaves:
                    return None
                return path

            level = int(self._tree_level[node])

            best_score = -np.inf
            best_child = None
            log_total  = np.log(self._total_triggers + 1)

            for child in children:
                c_stats = self._mcts[child]
                ucb = c_stats.mean_gain + self.mcts_c * np.sqrt(
                    log_total / (c_stats.visits + 1)
                )
                if ucb > best_score:
                    best_score = ucb
                    best_child = child

            if best_child is None:
                return None

            path.append(int(best_child))
            node = int(best_child)


    def _flat_inject(
        self,
        evaluated_prompts: List[str],
        eval_count: int,
        history_best: Optional[List[float]],
        evaluator,
    ) -> List[str]:
        """Original furthest-cluster logic used when no tree is available."""
        if not evaluated_prompts:
            chosen = self._rng.choice(
                len(self.centroids_pca),
                size=min(self.n_traversals, len(self.centroids_pca)),
                replace=False,
            )
        else:
            eval_pca   = self._project(evaluated_prompts)
            diff       = self.centroids_pca[:, None, :] - eval_pca[None, :, :]
            min_dists  = np.linalg.norm(diff, axis=-1).min(axis=1)
            chosen     = np.argsort(min_dists)[::-1][: self.n_traversals]

        for c in chosen:
            self._cluster_last_used[c] = eval_count

        exclude = set(evaluated_prompts)
        candidates = []
        for c in chosen:
            idxs      = np.where(self.labels == c)[0]
            available = [i for i in idxs if self.prompts[i] not in exclude]
            if not available:
                continue
            k       = min(self.n_per_leaf, len(available))
            sampled = self._rng.choice(available, size=k, replace=False)
            candidates.extend([str(self.prompts[i]) for i in sampled])
        return candidates

    def _project(self, prompts: List[str]) -> np.ndarray:
        to_embed = [p for p in prompts if p not in self._emb_cache]
        if to_embed:
            raw  = self.clip_embedder.batch_numpy(to_embed).astype(np.float32)
            raw  = normalize(raw, norm="l2")
            proj = (raw - self.pca_mean) @ self.pca_components.T
            for p, emb in zip(to_embed, proj):
                self._emb_cache[p] = emb
        return np.stack([self._emb_cache[p] for p in prompts])


    @property
    def trigger_log(self) -> List[dict]:
        return self._trigger_log

    def save_log(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._trigger_log, f, indent=2)