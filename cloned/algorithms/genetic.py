from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from cloned.algorithms.base import SearchAlgorithm, SearchResult
from cloned.spaces.base import SearchSpace, Candidate
from cloned.generators.base import ImageGenerator
from cloned.rewards.base import Reward
from cloned.algorithms.surrogate import SurrogateModel, build_embedder
from .evaluation import EvaluationManager
from .tracker import best_prompt_from_cache
from .salience import SalienceTracker, salience_crossover, salience_mutate


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


class GeneticSearch(SearchAlgorithm):
    def __init__(
        self,
        *,
        max_evals: int = 200,
        population_size: int = 20,
        n_init: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
        elite_frac: float = 0.3,
        use_surrogate: bool = False,
        surrogate_warmup: int = 40,
        surrogate_pool_size: int = 200,
        surrogate_k_eval: int = 20,
        surrogate_beta: float = 0.1,
        surrogate_gamma: float = 0.1,
        surrogate_n_ensemble: int = 5,
        surrogate_gen_batch: int = 32,
        surrogate_backbone: str = "dino",
        dino_model: str = "dino_vits8",
        berg_dir: Optional[str] = None,
        berg_surrogate_subject: Optional[int] = None,
        berg_surrogate_roi: str = "FFA-1",
        step_penalty: float = 0.0,
        surrogate_window: Optional[int] = None,
        use_recency_weighting: bool = False,
        elite_window: Optional[int] = None,
        use_state_descriptor: bool = False,
        state_window: int = 20,
        n_images_to_save: int = 10,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        use_salience_operators: bool = False,
        salience_ema_alpha: float = 0.7,
        salience_update_every: int = 1,
        salience_warmup: int = 20,
    ):
        self.max_evals              = max_evals
        self.population_size        = population_size
        self.n_init                 = n_init
        self.mutation_rate          = mutation_rate
        self.crossover_rate         = crossover_rate
        self.elite_frac             = elite_frac
        self.use_surrogate          = use_surrogate
        self.surrogate_warmup       = surrogate_warmup
        self.surrogate_pool_size    = surrogate_pool_size
        self.surrogate_k_eval       = surrogate_k_eval
        self.surrogate_beta         = surrogate_beta
        self.surrogate_gamma        = surrogate_gamma
        self.surrogate_n_ensemble   = surrogate_n_ensemble
        self.surrogate_gen_batch    = surrogate_gen_batch
        self.surrogate_backbone     = surrogate_backbone
        self.dino_model             = dino_model
        self.berg_dir               = berg_dir
        self.berg_surrogate_subject = berg_surrogate_subject
        self.berg_surrogate_roi     = berg_surrogate_roi
        self.step_penalty           = step_penalty
        self.surrogate_window       = surrogate_window
        self.use_recency_weighting  = use_recency_weighting
        self.elite_window           = elite_window
        self.use_state_descriptor   = use_state_descriptor
        self.state_window           = state_window
        self.n_images_to_save       = n_images_to_save
        self.device                 = device
        self.out_dir                = out_dir
        self.use_salience_operators = use_salience_operators
        self.salience_ema_alpha     = salience_ema_alpha
        self.salience_update_every  = salience_update_every
        self.salience_warmup        = salience_warmup

        self._salience: Optional[SalienceTracker] = None

    def _init_salience(self, space: SearchSpace) -> SalienceTracker:
        """Create a SalienceTracker sized to match space.options."""
        dim_names = getattr(space, "dim_names", None)
        if dim_names is None:
            dim_names = [f"dim_{i}" for i in range(len(space.options))]
        return SalienceTracker(
            dim_names       = dim_names,
            ema_alpha       = self.salience_ema_alpha,
            min_obs_per_dim = self.salience_warmup,
        )

    def _salience_active(self, evaluator: EvaluationManager) -> bool:
        return (self.use_salience_operators
                and self._salience is not None
                and evaluator.eval_count >= self.salience_warmup)

    def _population_variance_norm(
        self, space: SearchSpace, population: list, evaluator: EvaluationManager
    ) -> float:
        scores = []
        for ind in population:
            p = space.decode(Candidate(tuple(ind)))
            v = evaluator.reward_cache.get(p)
            if v:
                scores.append(float(np.mean(v)))
        if len(scores) < 2:
            return 1.0
        var = float(np.var(scores))
        lo, hi = min(scores), max(scores)
        max_var = ((hi - lo) ** 2) / 4.0 + 1e-12
        return float(np.clip(var / max_var, 0.0, 1.0))

    def run(
        self,
        space: SearchSpace,
        generator: ImageGenerator,
        reward: Reward,
        seed: int,
        out_dir: Optional[str] = None,
        drift_explorer=None,
    ) -> SearchResult:
        out_dir = out_dir or self.out_dir
        if out_dir:
            ensure_dir(out_dir)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.use_salience_operators:
            self._salience = self._init_salience(space)
        else:
            self._salience = None

        evaluator = EvaluationManager()
        seen: set = set()
        history_best, history_gmean = [], []

        surrogate = None
        if self.use_surrogate or self.step_penalty > 0:
            embedder  = build_embedder(
                backbone=self.surrogate_backbone,
                device=self.device,
                dino_model=self.dino_model,
                berg_dir=self.berg_dir,
                berg_subject=self.berg_surrogate_subject,
                berg_roi=self.berg_surrogate_roi,
            )
            surrogate = SurrogateModel(
                embedder=embedder,
                n_ensemble=self.surrogate_n_ensemble,
                beta=self.surrogate_beta,
                gamma=self.surrogate_gamma,
                step_penalty=self.step_penalty,
                use_recency_weighting=self.use_recency_weighting,
                surrogate_window=self.surrogate_window,
                use_state_descriptor=self.use_state_descriptor,
                state_window=self.state_window,
            )

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
        print(f"[genetic] init — best={running_best:.4f}  mean={init_mean:.4f}  "
              f"evals={evaluator.eval_count}")

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
                if self._salience_active(evaluator):
                    print(f"[genetic] salience top-5: "
                          f"{self._salience.log_top(5)}")

            recent_pool = self._recent_elite_pool(evaluator, space)
            if recent_pool is not None:
                scored = self._score_population(space, recent_pool, evaluator)
            else:
                scored = self._score_population(space, population, evaluator)
            n_elite = max(1, int(round(len(scored) * self.elite_frac)))
            elites  = [ind for _, ind in scored[:n_elite]]

            surrogate_active = (
                self.use_surrogate
                and surrogate is not None
                and evaluator.eval_count >= self.surrogate_warmup
            )

            pop_var = self._population_variance_norm(space, population, evaluator)

            if surrogate_active:
                eval_images, eval_rewards, eval_indices = \
                    self._get_eval_images_and_rewards(evaluator)
                surrogate.fit(
                    eval_images, eval_rewards,
                    history_gmean=history_gmean,
                    eval_indices=eval_indices,
                )
                surrogate.set_current_state(history_gmean, evaluator.eval_count)
                population = self._surrogate_generation(
                    gen, space, generator, reward, evaluator,
                    elites, seen, surrogate, remaining, seed,
                    pop_var=pop_var,
                )
            else:
                population = self._standard_generation(
                    space, generator, reward, evaluator,
                    elites, seen, remaining, seed,
                    surrogate=surrogate,
                    pop_var=pop_var,
                )

            explorer_fired = False
            if drift_explorer is not None:
                remaining = self.max_evals - evaluator.eval_count
                if remaining > 0:
                    inject_prompts = drift_explorer.step(
                        history_best=history_gmean,
                        evaluated_prompts=list(evaluator.reward_cache.keys()),
                        eval_count=evaluator.eval_count,
                        history_gmean=history_gmean,
                    )
                    if inject_prompts:
                        inject_inds = self._prompts_to_inds(
                            inject_prompts, space, seen, evaluator, remaining,
                        )
                        if inject_inds:
                            evaluator.evaluate(
                                space, generator, reward, inject_inds,
                                batch_size=16, seed=seed,
                            )
                            explorer_fired = True

            current_best = self._best(evaluator)
            running_best = max(running_best, current_best)
            new_evals    = evaluator.eval_count - len(history_best)
            gen_log      = evaluator.bb_call_log[len(history_best):]
            gen_scores   = [float(np.mean(evaluator.reward_cache[e["prompt"]]))
                            for e in gen_log if e["prompt"] in evaluator.reward_cache]
            gen_mean     = float(np.mean(gen_scores)) if gen_scores else float("nan")
            history_best.extend( [running_best] * max(new_evals, 1))
            history_gmean.extend([gen_mean]     * max(new_evals, 1))
            print(
                f"[genetic] gen {gen}  evals={evaluator.eval_count}  "
                f"best={running_best:.4f}  gen_mean={gen_mean:.4f}"
                + (f"  [surrogate:{self.surrogate_backbone}]" if surrogate_active else "")
                + ("  [explorer]" if explorer_fired else "")
            )

        if out_dir:
            np.save(
                os.path.join(out_dir, "all_raw_scores.npy"),
                np.array([e["raw_score"] for e in evaluator.bb_call_log],
                         dtype=np.float32),
            )
            if self._salience is not None:
                np.save(
                    os.path.join(out_dir, "salience_weights.npy"),
                    self._salience.weights,
                )
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
                        os.path.join(
                            imgs_dir,
                            f"top{rank+1:02d}_{call_idx:04d}_{score:.4f}.png",
                        )
                    )
                print(f"[genetic] saved top-{len(scored_prompts)} images → {imgs_dir}")

        best_p, best_s = best_prompt_from_cache(evaluator.reward_cache)
        if best_p is None:
            raise RuntimeError("No best prompt found.")
        if out_dir:
            evaluator.save_logs(out_dir)

        return SearchResult(
            best_candidate=Candidate(tuple(evaluator.ind_by_prompt[best_p])),
            best_prompt=best_p,
            best_score=float(best_s),
            history_best=history_best,
            history_overall=history_best,
            history_gmean=history_gmean,
        )


    def _get_eval_images_and_rewards(self, evaluator):
        """Return (images, rewards, eval_indices) in bb_call_log order."""
        ordered     = []
        seen_p: set = set()
        for e in evaluator.bb_call_log:
            p = e["prompt"]
            if p not in seen_p and p in evaluator.img_cache:
                seen_p.add(p)
                ordered.append((p, e["call_index"]))

        images      = [evaluator.img_cache[p]   for p, _ in ordered]
        rewards     = np.array(
            [float(np.mean(evaluator.reward_cache[p])) for p, _ in ordered],
            dtype=np.float32,
        )
        eval_indices = [idx for _, idx in ordered]
        return images, rewards, eval_indices

    def _prompts_to_inds(self, prompts, space, seen, evaluator, remaining):
        inds = []
        for p in prompts:
            if len(inds) >= remaining:
                break
            if p in evaluator.reward_cache or p in seen:
                continue
            if p in evaluator.ind_by_prompt:
                ind = evaluator.ind_by_prompt[p]
            else:
                ind = space.encode(p)
                if ind is None:
                    continue
                evaluator.register_ind(ind, p)
            seen.add(p)
            inds.append(ind)
        return inds

    def _standard_generation(
        self, space, generator, reward, evaluator,
        elites, seen, remaining, seed,
        surrogate=None,
        pop_var: float = 1.0,
    ) -> list:
        next_gen = list(elites)
        while len(next_gen) < self.population_size:
            if evaluator.eval_count >= self.max_evals:
                break
            child = self._breed_one(space, elites, seen, evaluator,
                                    pop_var=pop_var)
            if child is not None:
                next_gen.append(child)

        to_eval = self._unevaluated(space, next_gen, evaluator)[:remaining]

        if self.step_penalty > 0 and surrogate is not None and to_eval:
            to_eval = self._apply_step_penalty(
                to_eval, space, generator, evaluator, surrogate, seed
            )

        if to_eval:
            evaluator.evaluate(space, generator, reward, to_eval,
                               batch_size=16, seed=seed)
        return next_gen

    def _surrogate_generation(
        self, gen, space, generator, reward, evaluator,
        elites, seen, surrogate, remaining, seed,
        pop_var: float = 1.0,
    ) -> list:
        pool = self._breed_pool(space, elites, seen,
                                self.surrogate_pool_size, evaluator,
                                pop_var=pop_var)

        pool_prompts = [space.decode(Candidate(tuple(ind))) for ind in pool]
        pool_images  = []
        for i in range(0, len(pool_prompts), self.surrogate_gen_batch):
            batch = pool_prompts[i : i + self.surrogate_gen_batch]
            pool_images.extend(
                generator.generate(batch, seed=seed + gen * 1000 + i)
            )
        for p, img in zip(pool_prompts, pool_images):
            evaluator.img_cache[p] = img

        acq      = surrogate.acquisition(pool_images)
        k        = min(self.surrogate_k_eval, remaining)
        top_idx  = np.argsort(acq)[::-1][:k]
        top_inds = [pool[i] for i in top_idx]

        to_eval = [
            ind for ind in top_inds
            if space.decode(Candidate(tuple(ind))) not in evaluator.reward_cache
        ]
        if to_eval:
            evaluator.evaluate(space, generator, reward, to_eval,
                               batch_size=16, seed=seed)
        return elites + top_inds

    def _apply_step_penalty(
        self, candidates, space, generator, evaluator, surrogate, seed,
    ) -> list:
        prompts = [space.decode(Candidate(tuple(ind))) for ind in candidates]
        images  = generator.generate(prompts, seed=seed)
        for p, img in zip(prompts, images):
            evaluator.img_cache[p] = img

        cand_embs = surrogate.embed(images)
        eval_embs = surrogate.eval_embs
        if eval_embs is None:
            return candidates

        cand_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
        eval_norm = eval_embs / (np.linalg.norm(eval_embs, axis=1, keepdims=True) + 1e-8)
        max_sim   = (cand_norm @ eval_norm.T).max(axis=1)

        raw_scores = np.array([
            float(np.mean(evaluator.reward_cache.get(
                space.decode(Candidate(tuple(ind))), [0.0]
            )))
            for ind in candidates
        ])
        penalised = raw_scores - self.step_penalty * max_sim
        order     = np.argsort(penalised)[::-1]
        return [candidates[i] for i in order]


    def _breed_one(self, space, elites, seen, evaluator,
                     pop_var: float = 1.0):
        """
        Produce one child using salience-biased crossover + adaptive mutation.
    
        FIX: salience_crossover requires parent_a to be the FITTER parent.
        We explicitly score both parents and swap if needed before calling it.
        Without this fix, the salience bias on crossover is applied randomly
        rather than consistently favouring the better parent's high-salience genes.
        """
        parent_a = random.choice(elites)
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
                child = salience_crossover(
                    parent_a, parent_b, self._salience.weights
                )
            else:
                child = [parent_a[i] if random.random() < 0.5 else parent_b[i]
                        for i in range(len(parent_a))]
        else:
            child = list(parent_a)
    
        if use_salience:
            options_lengths = [len(opts) for opts in space.options]
            child = salience_mutate(
                child,
                options_lengths,
                self._salience.weights,
                base_rate           = self.mutation_rate * 0.5,
                population_variance = pop_var,
                max_rate            = min(self.mutation_rate * 3.0, 0.8),
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

    def _breed_pool(
        self, space, elites, seen, pool_size, evaluator,
        pop_var: float = 1.0,
    ):
        pool, tries = [], 0
        while len(pool) < pool_size and tries < pool_size * 10:
            tries += 1
            child = self._breed_one(space, elites, seen, evaluator,
                                    pop_var=pop_var)
            if child is not None:
                pool.append(child)
        return pool

    def _make_random(self, space, seen, n):
        out, tries = [], 0
        while len(out) < n and tries < n * 50:
            tries += 1
            ind = list(space.random_candidate().genes)
            p   = space.decode(Candidate(tuple(ind)))
            if p in seen:
                continue
            seen.add(p)
            out.append(ind)
        return out

    def _score_population(self, space, population, evaluator):
        scored = []
        for ind in population:
            p = space.decode(Candidate(tuple(ind)))
            s = float(np.mean(evaluator.reward_cache.get(p, [0.0])))
            scored.append((s, ind))
        scored.sort(reverse=True)
        return scored

    def _recent_elite_pool(self, evaluator, space) -> Optional[List[List[int]]]:
        if self.elite_window is None:
            return None
        ordered = []
        seen_p: set = set()
        for e in evaluator.bb_call_log:
            p = e["prompt"]
            if p not in seen_p:
                seen_p.add(p)
                ordered.append(p)
        recent_prompts = set(ordered[-self.elite_window:])
        inds = [
            evaluator.ind_by_prompt[p]
            for p in recent_prompts
            if p in evaluator.ind_by_prompt
        ]
        return inds if inds else None

    def _unevaluated(self, space, population, evaluator):
        return [ind for ind in population
                if space.decode(Candidate(tuple(ind))) not in evaluator.reward_cache]

    def _best(self, evaluator) -> float:
        _, best_s = best_prompt_from_cache(evaluator.reward_cache)
        return float(best_s) if best_s is not None else float("nan")