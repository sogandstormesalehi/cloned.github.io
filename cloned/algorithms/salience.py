from __future__ import annotations

import random as _random

from cloned.spaces.base import Candidate
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch



class SalienceTracker:
    """
    Per-dimension salience weights learned from (prompt, reward) observations.

    Parameters
    ----------
    dim_names           : list of str — one per gene dimension
    option_embeddings   : list of (n_opts_d, D) torch.Tensor | None
                          CLIP text embeddings per dimension.
                          Required for use_clip_smoothing=True.
    ema_alpha           : EMA decay for weight updates (0.7 = moderate)
    min_obs_per_dim     : minimum total observations before weights deviate
                          from flat. Default 60 (eta² needs ~60 obs for
                          reasonable stability with ~40 options per dim).
    use_clip_smoothing  : use CLIP-smoothed group means for eta-squared
    clip_smooth_alpha   : sharpness of similarity weighting
                          2.0 = moderate smoothing, 4.0 = sharper
    """

    def __init__(
        self,
        dim_names:          List[str],
        option_embeddings:  Optional[List[torch.Tensor]] = None,
        ema_alpha:          float = 0.7,
        min_obs_per_dim:    int   = 60,
        use_clip_smoothing: bool  = True,
        clip_smooth_alpha:  float = 2.0,
    ):
        self.dim_names          = dim_names
        self.n_dims             = len(dim_names)
        self.ema_alpha          = ema_alpha
        self.min_obs_per_dim    = min_obs_per_dim
        self.use_clip_smoothing = use_clip_smoothing
        self.clip_smooth_alpha  = clip_smooth_alpha

        # pre-compute normalised CLIP cosine similarity matrices per dimension
        # sim_matrices[d] is (n_opts_d, n_opts_d) float32 numpy array
        self.sim_matrices: Optional[List[np.ndarray]] = None
        if use_clip_smoothing and option_embeddings is not None:
            self.sim_matrices = []
            for emb in option_embeddings:
                e = emb.float().numpy()
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
                self.sim_matrices.append((e @ e.T).astype(np.float32))
            print(f"[SalienceTracker] CLIP similarity matrices built for "
                  f"{len(self.sim_matrices)} dimensions")

        self._weights      = np.ones(self.n_dims, dtype=np.float64) / self.n_dims
        self._update_count = 0

    @property
    def weights(self) -> np.ndarray:
        """Normalised salience weights, shape (n_dims,), sums to 1."""
        return self._weights.copy()

    # ── smooth reward estimates per option value ──────────────────────────────

    def _smooth_group_means(
        self,
        dim:     int,
        inds:    np.ndarray,   # (N, D) int32
        rewards: np.ndarray,   # (N,) float64
    ) -> np.ndarray:
        """
        CLIP-smoothed mean reward for each option in dimension dim.

        For each option v:
          weights_i = sim(v, chosen_option_i)^alpha
          mean_v    = sum(weights * rewards) / sum(weights)

        Options never chosen and dissimilar to anything observed
        fall back to grand_mean.

        Returns float64 array of shape (n_opts_d,).
        """
        sim_mat    = self.sim_matrices[dim]   # (n_opts, n_opts)
        n_opts     = sim_mat.shape[0]
        alpha      = self.clip_smooth_alpha
        grand_mean = rewards.mean()
        obs_vals   = inds[:, dim]             # (N,) which option was chosen

        smoothed = np.full(n_opts, grand_mean, dtype=np.float64)
        for v in range(n_opts):
            sims    = sim_mat[v, obs_vals].astype(np.float64)  # (N,)
            weights = sims ** alpha
            w_sum   = weights.sum()
            if w_sum > 1e-10:
                smoothed[v] = float((weights * rewards).sum() / w_sum)

        return smoothed

    # ── eta-squared variants ──────────────────────────────────────────────────

    def _eta_squared_plain(
        self,
        dim:      int,
        inds:     np.ndarray,
        rewards:  np.ndarray,
        ss_total: float,
    ) -> float:
        groups: dict = defaultdict(list)
        for i, val in enumerate(inds[:, dim]):
            groups[val].append(rewards[i])
        if len(groups) < 2:
            return 0.0
        grand_mean  = rewards.mean()
        ss_between  = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2
            for v in groups.values()
        )
        return float(ss_between / ss_total)

    def _eta_squared_smooth(
        self,
        dim:      int,
        inds:     np.ndarray,
        rewards:  np.ndarray,
        ss_total: float,
    ) -> float:
        """
        Eta-squared on CLIP-smoothed option means.

        More stable than plain eta-squared because each option gets a
        reward estimate informed by all semantically similar observations,
        not just exact matches.

        Weighted by sqrt(count) to give more weight to well-observed
        options without letting very common options dominate.
        """
        smoothed   = self._smooth_group_means(dim, inds, rewards)  # (n_opts,)
        n_opts     = len(smoothed)
        obs_vals   = inds[:, dim]
        counts     = np.bincount(obs_vals, minlength=n_opts).astype(np.float64)

        # sqrt weighting: less extreme than count, more than uniform
        w     = np.sqrt(counts)
        w_sum = w.sum()
        if w_sum < 1e-10:
            return 0.0

        weighted_mean = float((w * smoothed).sum() / w_sum)
        ss_between    = float((w * (smoothed - weighted_mean) ** 2).sum())
        return float(ss_between / max(ss_total, 1e-12))

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, evaluator, space) -> np.ndarray:
        """
        Recompute salience from all evaluated (prompt, reward) pairs.
        No-op until min_obs_per_dim observations have been collected.
        """
        if not evaluator.reward_cache:
            return self._weights.copy()

        records: List[tuple] = []
        for prompt, rewards in evaluator.reward_cache.items():
            ind = evaluator.ind_by_prompt.get(prompt)
            if ind is None:
                continue
            records.append((list(ind), float(np.mean(rewards))))

        if len(records) < self.min_obs_per_dim:
            return self._weights.copy()

        inds    = np.array([r[0] for r in records], dtype=np.int32)
        rewards = np.array([r[1] for r in records], dtype=np.float64)

        grand_mean = rewards.mean()
        ss_total   = float(np.sum((rewards - grand_mean) ** 2))

        if ss_total < 1e-12:
            return self._weights.copy()

        use_smooth = (self.use_clip_smoothing
                      and self.sim_matrices is not None)

        eta2 = np.zeros(self.n_dims, dtype=np.float64)
        for d in range(self.n_dims):
            if np.unique(inds[:, d]).size < 2:
                continue
            if use_smooth:
                eta2[d] = self._eta_squared_smooth(d, inds, rewards, ss_total)
            else:
                eta2[d] = self._eta_squared_plain(d, inds, rewards, ss_total)

        alpha         = self.ema_alpha
        new_raw       = alpha * self._weights + (1.0 - alpha) * eta2
        total         = new_raw.sum()
        self._weights = new_raw / total if total > 1e-12 else new_raw
        self._update_count += 1
        return self._weights.copy()

    def log_top(self, k: int = 5) -> str:
        order = np.argsort(self._weights)[::-1]
        return "  ".join(f"{self.dim_names[i]}={self._weights[i]:.3f}"
                         for i in order[:k])


# ── salience-biased crossover (unchanged from original) ───────────────────────

def salience_crossover(
    parent_a: List[int],
    parent_b: List[int],
    salience: np.ndarray,
) -> List[int]:
    """
    Salience-biased crossover.

    IMPORTANT: parent_a must be the FITTER parent — call site is responsible.

    High-salience dims → strongly favour fitter parent's gene (p up to 0.95).
    Low-salience dims  → near-coin-flip (p ≈ 0.5).
    """
    s_max = salience.max()
    if s_max < 1e-12:
        return [parent_a[i] if _random.random() < 0.5 else parent_b[i]
                for i in range(len(parent_a))]
    s_norm = salience / s_max
    return [
        parent_a[i] if _random.random() < 0.5 + 0.45 * float(s_norm[i])
        else parent_b[i]
        for i in range(len(parent_a))
    ]


# ── semantic mutation (NEW) ───────────────────────────────────────────────────

def salience_mutate_semantic(
    individual:          List[int],
    space,
    option_embeddings:   List[torch.Tensor],
    salience:            np.ndarray,
    base_rate:           float = 0.10,
    population_variance: float = 1.0,
    temperature:         float = 0.10,
    max_rate:            float = 0.70,
) -> List[int]:
    """
    Salience-adaptive mutation with CLIP-guided within-dimension exploration.

    Level 1 — WHETHER to mutate dimension d (from salience):
      p_mut_d = base_rate + (1 - s_d) * explore_budget * convergence_boost
      Low-salience dims mutate more; high-salience dims mutate less.

    Level 2 — WHERE to mutate within dimension d (CLIP-guided, NEW):
      Sample new value ∝ softmax(sim(current_val, all_vals) / temperature)
      excluding current value (always produce a different gene).

      Low temperature  (0.05) → stay semantically close to current value
      High temperature (0.20) → explore more broadly within the dimension

    This is better than uniform random mutation because semantically similar
    options are more likely to preserve whatever made the current value good,
    while still introducing variation. Think of it as gradient-free local
    search in the discrete semantic space of each category.

    Parameters
    ----------
    individual          : current gene vector (list of int)
    space               : SearchSpace (for space.options)
    option_embeddings   : list of (n_opts_d, D) torch.Tensor per dimension
    salience            : weight vector from SalienceTracker.weights
    base_rate           : minimum mutation probability
    population_variance : normalised reward variance [0,1]
    temperature         : CLIP softmax temperature (0.05–0.20 recommended)
    max_rate            : hard cap on per-dimension mutation probability
    """
    s_max = salience.max()
    s_norm = (salience / s_max) if s_max > 1e-12 else np.ones_like(salience) * 0.5

    explore_budget    = max_rate - base_rate
    convergence_boost = 1.0 + 2.0 * (1.0 - float(np.clip(population_variance, 0, 1)))

    child = list(individual)
    for i, s in enumerate(s_norm):
        p_mut = base_rate + (1.0 - float(s)) * explore_budget * convergence_boost
        p_mut = float(np.clip(p_mut, base_rate, max_rate))

        if _random.random() >= p_mut:
            continue

        n_opts = len(space.options[i])
        if n_opts <= 1:
            continue

        cur = child[i]
        emb = option_embeddings[i] if option_embeddings is not None else None

        if emb is None or emb.shape[0] != n_opts:
            # fallback: uniform random excluding current
            choices  = [v for v in range(n_opts) if v != cur]
            child[i] = _random.choice(choices)
            continue

        # CLIP cosine similarities from current option to all options
        e    = emb.float()
        e    = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sims = (e @ e[cur]).numpy()   # (n_opts,)

        # mask out current value — always mutate to something different
        sims[cur] = -1e9

        # softmax sampling
        logits = sims / temperature
        logits = logits - logits.max()   # numerical stability
        probs  = np.exp(logits)
        probs /= probs.sum()

        child[i] = int(np.random.choice(n_opts, p=probs))

    return child


# ── backward-compatible flat salience mutation ────────────────────────────────

def salience_mutate(
    individual:          List[int],
    options_lengths:     List[int],
    salience:            np.ndarray,
    base_rate:           float = 0.15,
    population_variance: float = 1.0,
    max_rate:            float = 0.70,
) -> List[int]:
    """
    Original salience_mutate with uniform random within-dimension selection.
    Kept for backward compatibility and for use in GeneticSearch (DINO).
    Prefer salience_mutate_semantic when option_embeddings are available.
    """
    s_max = salience.max()
    s_norm = (salience / s_max) if s_max > 1e-12 else np.ones_like(salience) * 0.5

    explore_budget    = max_rate - base_rate
    convergence_boost = 1.0 + 2.0 * (1.0 - float(np.clip(population_variance, 0, 1)))

    child = list(individual)
    for i, s in enumerate(s_norm):
        p_mut = base_rate + (1.0 - float(s)) * explore_budget * convergence_boost
        p_mut = float(np.clip(p_mut, base_rate, max_rate))
        if _random.random() < p_mut:
            n_opts = options_lengths[i]
            if n_opts > 1:
                cur     = child[i]
                choices = [v for v in range(n_opts) if v != cur]
                child[i] = _random.choice(choices)
    return child


# ── surrogate-based salience (replaces eta-squared) ──────────────────────────

def surrogate_sensitivity(
    surrogate,
    population:        List[List[int]],
    space,
    option_embeddings: Optional[List[torch.Tensor]] = None,
    n_samples:         int   = 50,
    ema_alpha:         float = 0.7,
    prev_weights:      Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-dimension salience from surrogate sensitivity analysis.

    For each dimension d, asks: "how much can predicted reward improve
    by optimally changing only dimension d, averaged across the population?"

    sensitivity_d = mean over population of:
        max_v surrogate(prompt with dim_d swapped to v) - surrogate(current)

    This captures interaction effects (the surrogate sees the full prompt)
    and uses the surrogate's actual learned model rather than marginal
    variance, making it much more informative than eta-squared with small
    sample sizes.

    Parameters
    ----------
    surrogate         : fitted CLIPTextSurrogate with acquisition_with_uncertainty
    population        : list of gene vectors (current elite candidates)
    space             : SearchSpace
    option_embeddings : (n_opts_d, 512) tensors per dimension, for encoding
                        option strings into prompts. If None, uses space.decode
                        directly (slower but correct).
    n_samples         : number of population members to sample for efficiency
    ema_alpha         : EMA decay when blending with prev_weights
    prev_weights      : previous weight vector for EMA smoothing

    Returns
    -------
    np.ndarray shape (n_dims,) normalised salience weights
    """
    if surrogate._eval_embs is None:
        # surrogate not fitted yet — return flat weights
        n_dims = len(space.options)
        return np.ones(n_dims, dtype=np.float32) / n_dims

    n_dims = len(space.options)
    sample = population[:n_samples] if len(population) > n_samples else population

    if not sample:
        n_dims = len(space.options)
        return np.ones(n_dims, dtype=np.float32) / n_dims

    # get baseline surrogate scores for the sampled population
    base_prompts = [space.decode(Candidate(tuple(ind))) for ind in sample]
    base_scores, _, _ = surrogate.acquisition_with_uncertainty(base_prompts)

    sensitivity = np.zeros(n_dims, dtype=np.float64)

    for d in range(n_dims):
        n_opts = len(space.options[d])
        gains  = []

        for ind in sample:
            base_score = base_scores[
                base_prompts.index(space.decode(Candidate(tuple(ind))))
            ]
            # try all option values for dimension d
            best_gain = 0.0
            for v in range(n_opts):
                if v == ind[d]:
                    continue
                swapped = list(ind)
                swapped[d] = v
                p = space.decode(Candidate(tuple(swapped)))
                try:
                    score, _, _ = surrogate.acquisition_with_uncertainty([p])
                    gain = float(score[0]) - float(base_score)
                    if gain > best_gain:
                        best_gain = gain
                except Exception:
                    continue
            gains.append(best_gain)

        sensitivity[d] = float(np.mean(gains)) if gains else 0.0

    # normalise to sum to 1
    total = sensitivity.sum()
    if total > 1e-10:
        weights = (sensitivity / total).astype(np.float32)
    else:
        weights = np.ones(n_dims, dtype=np.float32) / n_dims

    # EMA blend with previous weights
    if prev_weights is not None and len(prev_weights) == n_dims:
        weights = ema_alpha * prev_weights + (1.0 - ema_alpha) * weights
        total   = weights.sum()
        if total > 1e-10:
            weights /= total

    return weights


def surrogate_sensitivity_fast(
    surrogate,
    population:        List[List[int]],
    space,
    n_samples:         int   = 20,
    ema_alpha:         float = 0.7,
    prev_weights:      Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Faster version: batch all swapped prompts into one surrogate call.

    Instead of calling acquisition per candidate per dimension, builds
    a single large batch of all (candidate, dimension, value) combinations
    and scores them in one pass. Much faster when pool_size is large.

    With n_samples=20 and n_dims=16 and mean n_opts=40:
      batch_size = 20 * 16 * 40 = 12800 — scored in one CLIP embed + Ridge call.
    This takes ~200ms vs ~30s for the naive version.
    """
    if surrogate._eval_embs is None:
        n_dims = len(space.options)
        return np.ones(n_dims, dtype=np.float32) / n_dims

    n_dims = len(space.options)
    sample = population[:n_samples] if len(population) > n_samples else population

    if not sample:
        return np.ones(n_dims, dtype=np.float32) / n_dims

    # baseline scores
    base_prompts = [space.decode(Candidate(tuple(ind))) for ind in sample]
    base_scores, _, _ = surrogate.acquisition_with_uncertainty(base_prompts)
    base_score_map = {p: float(s) for p, s in zip(base_prompts, base_scores)}

    # build all swapped prompts in one list, track (sample_i, dim_d, val_v)
    all_prompts = []
    index_map   = []   # (sample_i, dim_d, val_v) for each entry in all_prompts

    for i, ind in enumerate(sample):
        for d in range(n_dims):
            for v in range(len(space.options[d])):
                if v == ind[d]:
                    continue
                swapped = list(ind)
                swapped[d] = v
                all_prompts.append(space.decode(Candidate(tuple(swapped))))
                index_map.append((i, d, v))

    if not all_prompts:
        return np.ones(n_dims, dtype=np.float32) / n_dims

    # single surrogate call for all swapped prompts
    try:
        all_scores, _, _ = surrogate.acquisition_with_uncertainty(all_prompts)
    except Exception:
        return np.ones(n_dims, dtype=np.float32) / n_dims

    # compute per-dimension best gain per sample
    # best_gain[i, d] = max gain from optimal swap of dim d for sample i
    best_gain = np.zeros((len(sample), n_dims), dtype=np.float64)

    for idx, (i, d, v) in enumerate(index_map):
        base  = base_score_map[base_prompts[i]]
        gain  = float(all_scores[idx]) - base
        if gain > best_gain[i, d]:
            best_gain[i, d] = gain

    sensitivity = best_gain.mean(axis=0)   # (n_dims,)

    # normalise
    total = sensitivity.sum()
    weights = (sensitivity / total).astype(np.float32) if total > 1e-10               else np.ones(n_dims, dtype=np.float32) / n_dims

    # EMA blend
    if prev_weights is not None and len(prev_weights) == n_dims:
        weights = ema_alpha * prev_weights + (1.0 - ema_alpha) * weights
        total   = weights.sum()
        if total > 1e-10:
            weights /= total

    return weights


# ── aliases for backward compatibility with genetic_clip_text.py ─────────────

class CLIPSmoothedSalienceTracker(SalienceTracker):
    """
    SalienceTracker with CLIP smoothing always enabled.
    Named separately for clarity and backward compatibility.

    Parameters
    ----------
    clip_option_embeddings  : list of (n_opts_d, D) numpy arrays (normalised)
    smoothing_temperature   : not used directly here — kept for API compat
                              (smoothing is controlled by clip_smooth_alpha)
    min_obs_for_smoothing   : not used separately — same as min_obs_per_dim
    """

    def __init__(
        self,
        dim_names:              list,
        clip_option_embeddings: Optional[List[np.ndarray]] = None,
        ema_alpha:              float = 0.7,
        min_obs_per_dim:        int   = 60,
        smoothing_temperature:  float = 0.15,   # kept for API compat
        min_obs_for_smoothing:  int   = 10,      # kept for API compat
        clip_smooth_alpha:      float = 2.0,
    ):
        import torch as _torch
        # convert numpy arrays to torch tensors if needed
        opt_embs = None
        if clip_option_embeddings is not None:
            opt_embs = []
            for e in clip_option_embeddings:
                if isinstance(e, np.ndarray):
                    opt_embs.append(_torch.from_numpy(e.astype(np.float32)))
                else:
                    opt_embs.append(e)

        super().__init__(
            dim_names          = dim_names,
            option_embeddings  = opt_embs,
            ema_alpha          = ema_alpha,
            min_obs_per_dim    = min_obs_per_dim,
            use_clip_smoothing = True,
            clip_smooth_alpha  = clip_smooth_alpha,
        )
        self.smoothing_temperature = smoothing_temperature


def salience_semantic_mutate(
    individual:             List[int],
    options_lengths:        List[int],
    salience:               np.ndarray,
    clip_option_embeddings: Optional[List[np.ndarray]] = None,
    base_rate:              float = 0.10,
    population_variance:    float = 1.0,
    temperature:            float = 0.10,
    max_rate:               float = 0.70,
) -> List[int]:
    """
    Salience-adaptive mutation with optional CLIP-guided within-dimension sampling.
    Accepts options_lengths (list of ints) and numpy arrays directly —
    no fake space object needed.

    If clip_option_embeddings is None, falls back to uniform random mutation.
    """
    s_max  = salience.max()
    s_norm = (salience / s_max) if s_max > 1e-12 else np.ones_like(salience) * 0.5

    explore_budget    = max_rate - base_rate
    convergence_boost = 1.0 + 2.0 * (1.0 - float(np.clip(population_variance, 0, 1)))

    # convert embeddings to torch once if provided
    opt_tensors = None
    if clip_option_embeddings is not None:
        import torch as _torch
        opt_tensors = []
        for e in clip_option_embeddings:
            if isinstance(e, np.ndarray):
                opt_tensors.append(_torch.from_numpy(e.astype(np.float32)))
            else:
                opt_tensors.append(e)

    child = list(individual)
    for i, s in enumerate(s_norm):
        p_mut = base_rate + (1.0 - float(s)) * explore_budget * convergence_boost
        p_mut = float(np.clip(p_mut, base_rate, max_rate))

        if _random.random() >= p_mut:
            continue

        n_opts = options_lengths[i]
        if n_opts <= 1:
            continue

        cur = child[i]

        if opt_tensors is not None:
            emb = opt_tensors[i]
            if emb.shape[0] == n_opts:
                # CLIP-guided sampling
                e    = emb.float()
                e    = e / (e.norm(dim=1, keepdim=True) + 1e-8)
                sims = (e @ e[cur]).numpy()
                sims[cur] = -1e9
                logits = sims / temperature
                logits = logits - logits.max()
                probs  = np.exp(logits)
                probs /= probs.sum()
                child[i] = int(np.random.choice(n_opts, p=probs))
                continue

        # fallback: uniform random excluding current
        choices  = [v for v in range(n_opts) if v != cur]
        child[i] = _random.choice(choices)

    return child