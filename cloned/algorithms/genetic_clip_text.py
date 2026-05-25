from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from cloned.algorithms.genetic import GeneticSearch
from cloned.algorithms.base import SearchResult
from cloned.algorithms.embedder import CLIPTextEmbedder
from cloned.algorithms.evaluation import EvaluationManager
from cloned.algorithms.tracker import best_prompt_from_cache
from cloned.spaces.base import Candidate
from cloned.algorithms.surrogate import CLIPTextSurrogate
from cloned.algorithms.salience import (
    SalienceTracker, CLIPSmoothedSalienceTracker,
    salience_crossover, salience_mutate, salience_semantic_mutate,
    surrogate_sensitivity_fast,
)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


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

        self._window:       float       = float(max_window)
        self._buffer:       List[float] = []
        self._stable_count: int         = 0
        self.log:           List[dict]  = []

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
            self.log.append({"gen_mean": gen_mean, "drift_score": None,
                             "window": self._window, "action": "warmup"})
            return False

        arr   = np.array(self._buffer, dtype=np.float32)
        x     = np.arange(len(arr), dtype=np.float32)
        xc    = x - x.mean()
        denom = (xc ** 2).sum()
        slope = float((xc * (arr - arr.mean())).sum() / denom) \
                if denom > 1e-8 else 0.0
        var   = float(arr.var())

        drift_score   = -slope / (var + self.eps)
        force_explore = False
        action        = "hold"

        if drift_score > self.drift_threshold:
            new_w        = max(float(self.min_window),
                               self._window * self.shrink_factor)
            action       = f"shrink {self._window:.0f}→{new_w:.0f}"
            self._window = new_w
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

        self.log.append({"gen_mean": gen_mean, "slope": slope, "var": var,
                         "drift_score": drift_score,
                         "window": int(round(self._window)),
                         "action": action, "force_explore": force_explore})

        if force_explore:
            print(f"[AdaptiveWindow] drift_score={drift_score:.3f} → {action}  "
                  f"force_explorer=True")
        elif "grow" in action or "shrink" in action:
            print(f"[AdaptiveWindow] drift_score={drift_score:.3f} → {action}")

        return force_explore


class CLIPTextGeneticSearch(GeneticSearch):
    def __init__(
        self,
        *,
        clip_name:            str   = "openai/clip-vit-base-patch32",
        n_pca:                int   = 50,
        elite_min_diversity:  float = 0.0,
        use_salience_operators:        bool  = False,
        salience_ema_alpha:            float = 0.7,
        salience_update_every:         int   = 1,
        salience_warmup:               int   = 60,
        clip_option_embeddings:        Optional[List] = None,
        semantic_mutation_temperature: float = 0.15,
        use_surrogate_salience:        bool  = True,
        adaptive_window:           bool  = False,
        adaptive_window_max:       int   = 40,
        adaptive_window_min:       int   = 20,
        adaptive_window_shrink:    float = 0.6,
        adaptive_window_grow:      float = 1.15,
        adaptive_window_buffer:    int   = 5,
        adaptive_drift_threshold:  float = 0.5,
        adaptive_stable_threshold: float = 0.0,
        adaptive_stable_patience:  int   = 3,
        **kwargs,
    ):
        super().__init__(
            use_salience_operators = use_salience_operators,
            salience_ema_alpha     = salience_ema_alpha,
            salience_update_every  = salience_update_every,
            salience_warmup        = salience_warmup,
            **kwargs,
        )
        self.clip_name                     = clip_name
        self.n_pca                         = n_pca
        self.elite_min_diversity           = elite_min_diversity
        self.clip_option_embeddings        = clip_option_embeddings
        self.semantic_mutation_temperature = semantic_mutation_temperature
        self.use_surrogate_salience        = use_surrogate_salience

        self.adaptive_window           = adaptive_window
        self.adaptive_window_max       = adaptive_window_max
        self.adaptive_window_min       = adaptive_window_min
        self.adaptive_window_shrink    = adaptive_window_shrink
        self.adaptive_window_grow      = adaptive_window_grow
        self.adaptive_window_buffer    = adaptive_window_buffer
        self.adaptive_drift_threshold  = adaptive_drift_threshold
        self.adaptive_stable_threshold = adaptive_stable_threshold
        self.adaptive_stable_patience  = adaptive_stable_patience

        self._clip:                Optional[CLIPTextEmbedder]         = None
        self._clip_text_surrogate: Optional[CLIPTextSurrogate]        = None
        self._clip_for_elites:     Optional[CLIPTextEmbedder]         = None
        self._norm_option_embs:    Optional[List]                     = None
        self._surrogate_sal_weights: Optional[np.ndarray]             = None
        self._aw_controller:       Optional[AdaptiveWindowController]  = None

        self._prior_pool_prompts: Optional[List[str]] = None
        self._prior_pool_frac:    float               = 0.3

    def set_prior_pool(self, prior_prompts: List[str], prior_frac: float = 0.3) -> None:
        self._prior_pool_prompts = list(prior_prompts)
        self._prior_pool_frac    = float(prior_frac)
        print(f"[clip_genetic] prior pool: {len(prior_prompts)} prompts  "
              f"frac={prior_frac:.2f}")

    def _init_salience(self, space) -> SalienceTracker:
        dim_names = getattr(space, "dim_names", None)
        if dim_names is None:
            dim_names = [f"dim_{i}" for i in range(len(space.options))]

        if self.clip_option_embeddings is not None:
            self._norm_option_embs = []
            for emb in self.clip_option_embeddings:
                e     = np.array(emb, dtype=np.float32)
                norms = np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
                self._norm_option_embs.append(e / norms)
        else:
            self._norm_option_embs = None

        if self.clip_option_embeddings is not None:
            print(f"[clip_genetic] using CLIPSmoothedSalienceTracker  "
                  f"temperature={self.semantic_mutation_temperature}")
            return CLIPSmoothedSalienceTracker(
                dim_names              = dim_names,
                clip_option_embeddings = self._norm_option_embs,
                ema_alpha              = self.salience_ema_alpha,
                min_obs_per_dim        = self.salience_warmup,
                smoothing_temperature  = self.semantic_mutation_temperature,
                min_obs_for_smoothing  = max(10, self.salience_warmup // 6),
            )
        else:
            return SalienceTracker(
                dim_names       = dim_names,
                ema_alpha       = self.salience_ema_alpha,
                min_obs_per_dim = self.salience_warmup,
            )

    def _breed_one(self, space, elites, seen, evaluator,
                   pop_var: float = 1.0) -> Optional[List[int]]:
        parent_a     = random.choice(elites)
        use_salience = self._salience_active(evaluator)

        if random.random() < self.crossover_rate and len(elites) > 1:
            parent_b = random.choice(elites)
            if use_salience:
                score_a = float(np.mean(evaluator.reward_cache.get(
                    space.decode(Candidate(tuple(parent_a))), [0.0])))
                score_b = float(np.mean(evaluator.reward_cache.get(
                    space.decode(Candidate(tuple(parent_b))), [0.0])))
                if score_b > score_a:
                    parent_a, parent_b = parent_b, parent_a
                child = salience_crossover(parent_a, parent_b, self._salience.weights)
            else:
                child = [parent_a[i] if random.random() < 0.5 else parent_b[i]
                         for i in range(len(parent_a))]
        else:
            child = list(parent_a)

        if use_salience:
            options_lengths = [len(opts) for opts in space.options]
            child = salience_semantic_mutate(
                child, options_lengths, self._salience.weights,
                clip_option_embeddings = self._norm_option_embs,
                base_rate              = self.mutation_rate * 0.5,
                population_variance    = pop_var,
                max_rate               = min(self.mutation_rate * 3.0, 0.8),
                temperature            = self.semantic_mutation_temperature,
            )
        else:
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    child[i] = random.randrange(len(space.options[i]))

        p = space.decode(Candidate(tuple(child)))
        if p in seen or p in evaluator.reward_cache:
            return None
        seen.add(p)
        evaluator.register_ind(child, p)
        return child

    def _select_diverse_elites(self, scored: list, space, n_elite: int) -> list:
        if self.elite_min_diversity <= 0 or self._clip_for_elites is None:
            return [ind for _, ind in scored[:n_elite]]

        prompts = [space.decode(Candidate(tuple(ind))) for _, ind in scored]
        embs    = self._clip_for_elites.text_numpy(prompts)
        norms   = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

        selected_idx, selected_norms = [], []
        for i in range(len(scored)):
            if len(selected_idx) >= n_elite:
                break
            if selected_norms:
                if max(float(norms[i] @ e) for e in selected_norms) \
                        > 1.0 - self.elite_min_diversity:
                    continue
            selected_idx.append(i)
            selected_norms.append(norms[i])

        taken = set(selected_idx)
        for i in range(len(scored)):
            if len(selected_idx) >= n_elite: break
            if i not in taken: selected_idx.append(i)

        return [scored[i][1] for i in selected_idx]

    def run(self, space, generator, reward, seed, out_dir=None,
            drift_explorer=None):
        if out_dir:
            ensure_dir(out_dir)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._clip            = CLIPTextEmbedder(device=self.device,
                                                  clip_name=self.clip_name)
        self._clip_for_elites = self._clip
        self._clip_text_surrogate = CLIPTextSurrogate(
            clip                  = self._clip,
            n_ensemble            = self.surrogate_n_ensemble,
            beta                  = self.surrogate_beta,
            gamma                 = self.surrogate_gamma,
            step_penalty          = self.step_penalty,
            use_recency_weighting = self.use_recency_weighting,
            surrogate_window      = self.surrogate_window,
            use_state_descriptor  = self.use_state_descriptor,
            state_window          = self.state_window,
            n_pca                 = self.n_pca,
        )

        if self.use_salience_operators:
            self._salience = self._init_salience(space)
        else:
            self._salience = None

        if self.adaptive_window:
            self._aw_controller = AdaptiveWindowController(
                max_window       = self.adaptive_window_max,
                min_window       = self.adaptive_window_min,
                shrink_factor    = self.adaptive_window_shrink,
                grow_factor      = self.adaptive_window_grow,
                buffer_size      = self.adaptive_window_buffer,
                drift_threshold  = self.adaptive_drift_threshold,
                stable_threshold = self.adaptive_stable_threshold,
                stable_patience  = self.adaptive_stable_patience,
            )
            self._clip_text_surrogate.surrogate_window = self.adaptive_window_max
            print(f"[clip_genetic] adaptive_window ON  "
                  f"max={self.adaptive_window_max}  min={self.adaptive_window_min}")
        else:
            self._aw_controller = None

        evaluator  = EvaluationManager()
        seen: set  = set()
        history_best, history_gmean = [], []

        population = self._make_random(space, seen, self.n_init)
        evaluator.evaluate(space, generator, reward, population,
                           batch_size=16, seed=seed)
        for ind in population:
            p = space.decode(Candidate(tuple(ind)))
            if p not in evaluator.ind_by_prompt:
                evaluator.register_ind(ind, p)

        running_best  = self._best(evaluator)
        init_scores   = [float(np.mean(evaluator.reward_cache[p]))
                         for p in evaluator.reward_cache]
        init_mean     = float(np.mean(init_scores)) if init_scores else float("nan")
        history_best  = [running_best] * evaluator.eval_count
        history_gmean = [init_mean]    * evaluator.eval_count
        print(f"[clip_genetic] warmup done — best={running_best:.4f}  "
              f"mean={init_mean:.4f}  evals={evaluator.eval_count}")

        if self._salience is not None:
            self._salience.update(evaluator, space)

        gen = 0
        while evaluator.eval_count < self.max_evals:
            gen += 1
            remaining = self.max_evals - evaluator.eval_count
            if remaining <= 0:
                break

            if self._salience is not None and gen % self.salience_update_every == 0:
                self._salience.update(evaluator, space)

            recent_pool = self._recent_elite_pool(evaluator, space)
            scored      = self._score_population(
                space,
                recent_pool if recent_pool is not None else population,
                evaluator,
            )
            n_elite = max(1, int(round(len(scored) * self.elite_frac)))
            elites  = self._select_diverse_elites(scored, space, n_elite)

            surrogate_active = (
                self.use_surrogate
                and evaluator.eval_count >= self.surrogate_warmup
            )
            pop_var = self._population_variance_norm(space, population, evaluator)

            if surrogate_active:
                prompts, rewards, indices = self._get_eval_prompts_and_rewards(evaluator)
                self._clip_text_surrogate.fit(
                    prompts, rewards,
                    history_gmean = history_gmean,
                    eval_indices  = indices,
                )
                self._clip_text_surrogate.set_current_state(
                    history_gmean, evaluator.eval_count
                )

                if (self.use_surrogate_salience
                        and self._salience is not None
                        and self._salience_active(evaluator)
                        and gen % self.salience_update_every == 0):
                    dim_names = getattr(space, "dim_names",
                                        [f"dim_{i}" for i in range(len(space.options))])
                    self._surrogate_sal_weights = surrogate_sensitivity_fast(
                        surrogate    = self._clip_text_surrogate,
                        population   = elites,
                        space        = space,
                        n_samples    = min(20, len(elites) + 10),
                        ema_alpha    = self.salience_ema_alpha,
                        prev_weights = self._surrogate_sal_weights,
                    )
                    self._salience._weights = \
                        self._surrogate_sal_weights.astype(np.float64)
                    print(f"[clip_genetic] surrogate salience top-5: "
                          f"{self._salience.log_top(5)}")

                population = self._clip_surrogate_generation(
                    gen, space, generator, reward, evaluator,
                    elites, seen, remaining, seed, pop_var=pop_var,
                )
            else:
                population = self._standard_generation(
                    space, generator, reward, evaluator,
                    elites, seen, remaining, seed,
                    surrogate=None, pop_var=pop_var,
                )

            if drift_explorer is not None:
                remaining_now = self.max_evals - evaluator.eval_count
                if remaining_now > 0:
                    inject_prompts = drift_explorer.step(
                        history_best      = history_gmean,
                        evaluated_prompts = list(evaluator.reward_cache.keys()),
                        eval_count        = evaluator.eval_count,
                        history_gmean     = history_gmean,
                    )
                    if inject_prompts:
                        inject_inds = self._prompts_to_inds(
                            inject_prompts, space, seen, evaluator, remaining_now,
                        )
                        if inject_inds:
                            evaluator.evaluate(space, generator, reward,
                                               inject_inds, batch_size=16, seed=seed)

            current_best  = self._best(evaluator)
            running_best  = max(running_best, current_best)
            new_evals     = evaluator.eval_count - len(history_best)
            gen_log       = evaluator.bb_call_log[len(history_best):]
            gen_scores    = [float(np.mean(evaluator.reward_cache[e["prompt"]]))
                             for e in gen_log
                             if e["prompt"] in evaluator.reward_cache]
            gen_mean      = float(np.mean(gen_scores)) if gen_scores else float("nan")
            history_best.extend( [running_best] * max(new_evals, 1))
            history_gmean.extend([gen_mean]     * max(new_evals, 1))

            if self._aw_controller is not None and not np.isnan(gen_mean):
                force_explore = self._aw_controller.update(gen_mean)
                new_w = self._aw_controller.current_window
                self._clip_text_surrogate.surrogate_window = new_w
                self.elite_window = (
                    max(self.adaptive_window_min, new_w)
                    if new_w is not None else None
                )
                if force_explore and drift_explorer is not None:
                    remaining_force = self.max_evals - evaluator.eval_count
                    if remaining_force > 0:
                        inject_prompts = drift_explorer.step(
                            history_best      = history_gmean,
                            evaluated_prompts = list(evaluator.reward_cache.keys()),
                            eval_count        = evaluator.eval_count,
                            history_gmean     = history_gmean,
                        )
                        if inject_prompts:
                            inject_inds = self._prompts_to_inds(
                                inject_prompts, space, seen,
                                evaluator, remaining_force,
                            )
                            if inject_inds:
                                evaluator.evaluate(
                                    space, generator, reward, inject_inds,
                                    batch_size=16, seed=seed,
                                )

            print(
                f"[clip_genetic] gen {gen}  evals={evaluator.eval_count}  "
                f"best={running_best:.4f}  gen_mean={gen_mean:.4f}"
                + ("  [clip_surrogate]" if surrogate_active else "")
                + ("  [salience]"       if self._salience_active(evaluator) else "")
                + (f"  [aw={self._clip_text_surrogate.surrogate_window}]"
                   if self._aw_controller is not None else "")
            )

        if out_dir:
            np.save(os.path.join(out_dir, "all_raw_scores.npy"),
                    np.array([e["raw_score"] for e in evaluator.bb_call_log],
                             dtype=np.float32))
            if self._salience is not None:
                np.save(os.path.join(out_dir, "salience_weights.npy"),
                        self._salience.weights)
            if self._aw_controller is not None:
                with open(os.path.join(out_dir, "adaptive_window_log.json"), "w") as f:
                    json.dump(self._aw_controller.log, f, indent=2)
            scored_prompts = sorted(
                [(p, float(np.mean(v))) for p, v in evaluator.reward_cache.items()
                 if p in evaluator.img_cache],
                key=lambda x: x[1], reverse=True,
            )[:self.n_images_to_save]
            if scored_prompts:
                imgs_dir = os.path.join(out_dir, "images")
                ensure_dir(imgs_dir)
                for rank, (prompt, score) in enumerate(scored_prompts):
                    call_idx = next(
                        (e["call_index"] for e in evaluator.bb_call_log
                         if e["prompt"] == prompt), 0
                    )
                    evaluator.img_cache[prompt].save(
                        os.path.join(imgs_dir,
                                     f"top{rank+1:02d}_{call_idx:04d}_{score:.4f}.png"))
            evaluator.save_logs(out_dir)

        best_p, best_s = best_prompt_from_cache(evaluator.reward_cache)
        if best_p is None:
            raise RuntimeError("No best prompt found.")
        return SearchResult(
            best_candidate = Candidate(tuple(evaluator.ind_by_prompt[best_p])),
            best_prompt    = best_p,
            best_score     = float(best_s),
            history_best   = history_best,
            history_overall= history_best,
            history_gmean  = history_gmean,
        )


    def _get_eval_prompts_and_rewards(self, evaluator: EvaluationManager):
        ordered, seen_p = [], set()
        for e in evaluator.bb_call_log:
            p = e["prompt"]
            if p not in seen_p:
                seen_p.add(p)
                ordered.append((p, e["call_index"]))
        prompts      = [p   for p, _ in ordered]
        rewards      = np.array(
            [float(np.mean(evaluator.reward_cache[p])) for p in prompts],
            dtype=np.float32,
        )
        eval_indices = [idx for _, idx in ordered]
        return prompts, rewards, eval_indices

    def _clip_surrogate_generation(
        self, gen, space, generator, reward, evaluator,
        elites, seen, remaining, seed,
        pop_var: float = 1.0,
    ) -> list:
        pool = self._breed_pool(space, elites, seen,
                                self.surrogate_pool_size, evaluator,
                                pop_var=pop_var)
        if not pool:
            return elites

        pool_prompts = [space.decode(Candidate(tuple(ind))) for ind in pool]

        prior_injected = 0
        if self._prior_pool_prompts:
            n_prior    = max(1, int(self.surrogate_pool_size * self._prior_pool_frac))
            candidates = [p for p in self._prior_pool_prompts
                        if p not in seen and p not in evaluator.reward_cache]
            if candidates:
                sample         = random.sample(candidates, min(n_prior, len(candidates)))
                pool_prompts   = pool_prompts + sample
                pool           = pool + [None] * len(sample)
                prior_injected = len(sample)

        acq, _, _ = self._clip_text_surrogate.acquisition_with_uncertainty(pool_prompts)

        k        = min(self.surrogate_k_eval, remaining)
        ranked   = np.argsort(acq)[::-1]
        top_idx  = [i for i in ranked if pool[i] is not None][:k]
        top_inds = [pool[i] for i in top_idx]

        top_prompts    = [pool_prompts[i] for i in top_idx]
        top_preds      = acq[top_idx]

        to_eval = [ind for ind in top_inds
                if space.decode(Candidate(tuple(ind)))
                not in evaluator.reward_cache]
        to_eval_prompts = [space.decode(Candidate(tuple(ind))) for ind in to_eval]

        if to_eval:
            evaluator.evaluate(space, generator, reward, to_eval,
                            batch_size=16, seed=seed)
            pred_map = dict(zip(top_prompts, top_preds))
            preds_for_eval = np.array([pred_map[p] for p in to_eval_prompts
                                    if p in pred_map], dtype=np.float32)
            prompts_for_eval = [p for p in to_eval_prompts if p in pred_map]
            if len(prompts_for_eval) > 0:
                evaluator.log_prediction_errors(prompts_for_eval, preds_for_eval)

        if prior_injected:
            print(f"  [prior_pool] injected {prior_injected} prior prompts")

        return elites + top_inds