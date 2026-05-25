from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image



def compute_state_descriptor(
    history_gmean: List[float],
    eval_idx: int,
    window: int = 20,
) -> np.ndarray:
    if len(history_gmean) == 0:
        return np.zeros(3, dtype=np.float32)
    end   = min(eval_idx, len(history_gmean))
    start = max(0, end - window)
    h     = np.array(history_gmean[start:end], dtype=np.float32)
    if len(h) == 0:
        return np.zeros(3, dtype=np.float32)
    mean_r = float(h.mean())
    var_r  = float(h.var())
    if len(h) >= 2:
        x     = np.arange(len(h), dtype=np.float32)
        xc    = x - x.mean()
        denom = (xc ** 2).sum()
        slope = float((xc * (h - h.mean())).sum() / denom) if denom > 1e-8 else 0.0
    else:
        slope = 0.0
    return np.array([mean_r, slope, var_r], dtype=np.float32)



def _fit_pca_on_train(
    eval_embs: np.ndarray,   
    pool_embs: np.ndarray, 
    n_pca:     int = 50,
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_train = len(eval_embs)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(eval_embs)         
    X_pool  = scaler.transform(pool_embs)             

    n_comp  = min(n_pca, n_train - 1, X_train.shape[1])
    pca     = PCA(n_components=n_comp, random_state=0)
    X_train = pca.fit_transform(X_train)             
    X_pool  = pca.transform(X_pool)                   
    return X_train, X_pool, scaler, pca



class ImageEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Returns L2-normalised float32 (N, D)."""


class DinoEmbedder(ImageEmbedder):
    def __init__(self, model_name: str = "dino_vits8", device: str = "cuda"):
        import torch
        import torchvision.transforms as T
        model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.model     = model.to(device).eval()
        self.device    = device
        self.transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])
        print(f"[DinoEmbedder] loaded {model_name} on {device}")

    def embed(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch   = images[i : i + batch_size]
                tensors = torch.stack(
                    [self.transform(img) for img in batch]
                ).to(self.device)
                embs = self.model(tensors)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)


class BERGEmbedder(ImageEmbedder):
    def __init__(self, berg_dir: str, subject: int,
                 roi: str = "nsdgeneral", device: str = "auto"):
        from berg import BERG
        berg      = BERG(berg_dir)
        self.berg = berg
        self.model = berg.get_encoding_model(
            "fmri-nsd_fsaverage-huze",
            subject=subject, selection={"roi": roi}, device=device,
        )
        print(f"[BERGEmbedder] subject={subject}  roi={roi}")

    def _to_uint8(self, images):
        import torchvision.transforms as T
        resize = T.Resize((224, 224))
        return np.stack([
            np.array(resize(img.convert("RGB")), dtype=np.uint8).transpose(2, 0, 1)
            for img in images
        ])

    def embed(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(images), batch_size):
            stims    = self._to_uint8(images[i : i + batch_size])
            lh, rh   = self.berg.encode(self.model, stims)
            voxels   = np.concatenate(
                [np.asarray(lh, dtype=np.float32),
                 np.asarray(rh, dtype=np.float32)], axis=1
            )
            voxels  /= np.linalg.norm(voxels, axis=1, keepdims=True) + 1e-8
            all_embs.append(voxels)
        return np.concatenate(all_embs, axis=0)


def build_embedder(
    backbone: str = "dino",
    device: str = "cuda",
    dino_model: str = "dino_vits8",
    berg_dir: Optional[str] = None,
    berg_subject: Optional[int] = None,
    berg_roi: str = "nsdgeneral",
) -> ImageEmbedder:
    if backbone == "dino":
        return DinoEmbedder(model_name=dino_model, device=device)
    elif backbone == "berg":
        if berg_dir is None or berg_subject is None:
            raise ValueError("backbone='berg' requires berg_dir and berg_subject.")
        return BERGEmbedder(berg_dir=berg_dir, subject=berg_subject,
                            roi=berg_roi, device=device)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'dino' or 'berg'.")


def _acquisition(
    eval_embs:    np.ndarray,
    eval_rewards: np.ndarray,
    pool_embs:    np.ndarray,
    pool_embs_raw: np.ndarray,
    eval_embs_raw: np.ndarray,
    n_ensemble:   int,
    beta:         float,
    gamma:        float,
    step_penalty: float,
    use_recency_weighting: bool,
    n_pca:        int = 50,
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    n_train = len(eval_rewards)
    X_train, X_pool, _, _ = _fit_pca_on_train(eval_embs, pool_embs, n_pca)

    if use_recency_weighting:
        decay     = 0.02
        recency_w = np.exp(-decay * np.arange(n_train - 1, -1, -1)).astype(np.float32)
        recency_w = recency_w / recency_w.sum() * n_train
    else:
        recency_w = None

    alphas      = np.logspace(-1, 2, n_ensemble)
    predictions = np.stack([
        Ridge(alpha=a).fit(X_train, eval_rewards,
                           sample_weight=recency_w).predict(X_pool)
        for a in alphas
    ])
    mean_pred   = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    pool_norm = pool_embs_raw / (
        np.linalg.norm(pool_embs_raw, axis=1, keepdims=True) + 1e-8)
    eval_norm = eval_embs_raw / (
        np.linalg.norm(eval_embs_raw, axis=1, keepdims=True) + 1e-8)
    max_sim   = (pool_norm @ eval_norm.T).max(axis=1)
    diversity = 1.0 - max_sim

    return (mean_pred
            + beta         * uncertainty
            + gamma        * diversity
            - step_penalty * max_sim)


class SurrogateModel:
    def __init__(
        self,
        embedder: ImageEmbedder,
        n_ensemble: int = 5,
        beta: float = 0.1,
        gamma: float = 0.1,
        step_penalty: float = 0.0,
        use_recency_weighting: bool = False,
        surrogate_window: Optional[int] = None,
        use_state_descriptor: bool = False,
        state_window: int = 20,
    ):
        self.embedder              = embedder
        self.n_ensemble            = n_ensemble
        self.beta                  = beta
        self.gamma                 = gamma
        self.step_penalty          = step_penalty
        self.use_recency_weighting = use_recency_weighting
        self.surrogate_window      = surrogate_window
        self.use_state_descriptor  = use_state_descriptor
        self.state_window          = state_window
        self._eval_embs:    Optional[np.ndarray] = None
        self._eval_rewards: Optional[np.ndarray] = None
        self._current_state: Optional[np.ndarray] = None

    def fit(
        self,
        eval_images:   List[Image.Image],
        eval_rewards:  np.ndarray,
        history_gmean: Optional[List[float]] = None,
        eval_indices:  Optional[List[int]]   = None,
    ) -> None:
        images  = list(eval_images)
        rewards = np.asarray(eval_rewards, dtype=np.float32)
        if self.surrogate_window is not None and len(images) > self.surrogate_window:
            images  = images[-self.surrogate_window:]
            rewards = rewards[-self.surrogate_window:]
            if eval_indices is not None:
                eval_indices = eval_indices[-self.surrogate_window:]
        embs = self.embedder.embed(images)
        if self.use_state_descriptor and history_gmean is not None:
            states = np.stack([
                compute_state_descriptor(history_gmean, idx, self.state_window)
                for idx in (eval_indices or range(len(images)))
            ])
            embs = np.concatenate([embs, states], axis=1)
        self._eval_embs    = embs
        self._eval_rewards = rewards

    def set_current_state(self, history_gmean: List[float], eval_count: int) -> None:
        if self.use_state_descriptor:
            self._current_state = compute_state_descriptor(
                history_gmean, eval_idx=eval_count, window=self.state_window
            )

    def acquisition(self, pool_images: List[Image.Image]) -> np.ndarray:
        if self._eval_embs is None:
            raise RuntimeError("Call fit() before acquisition().")
        pool_embs_raw = self.embedder.embed(pool_images)
        if self.use_state_descriptor and self._current_state is not None:
            state_tiled = np.tile(self._current_state, (len(pool_embs_raw), 1))
            pool_embs   = np.concatenate([pool_embs_raw, state_tiled], axis=1)
        else:
            pool_embs = pool_embs_raw
        d_raw         = pool_embs_raw.shape[1]
        eval_embs_raw = self._eval_embs[:, :d_raw]
        return _acquisition(
            eval_embs=self._eval_embs, eval_rewards=self._eval_rewards,
            pool_embs=pool_embs, pool_embs_raw=pool_embs_raw,
            eval_embs_raw=eval_embs_raw, n_ensemble=self.n_ensemble,
            beta=self.beta, gamma=self.gamma, step_penalty=self.step_penalty,
            use_recency_weighting=self.use_recency_weighting,
        )

    def acquisition_with_uncertainty(
        self, pool_images: List[Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.linear_model import Ridge

        pool_embs_raw = self.embedder.embed(pool_images)
        if self.use_state_descriptor and self._current_state is not None:
            state_tiled = np.tile(self._current_state, (len(pool_embs_raw), 1))
            pool_embs   = np.concatenate([pool_embs_raw, state_tiled], axis=1)
        else:
            pool_embs = pool_embs_raw

        eval_embs    = self._eval_embs
        eval_rewards = self._eval_rewards
        d_raw        = pool_embs_raw.shape[1]

        X_train, X_pool, _, _ = _fit_pca_on_train(eval_embs, pool_embs)

        preds = np.stack([
            Ridge(alpha=a).fit(X_train, eval_rewards).predict(X_pool)
            for a in np.logspace(-1, 2, self.n_ensemble)
        ])
        mean_pred, uncertainty = preds.mean(0), preds.std(0)

        eval_norm = self._eval_embs[:, :d_raw] / (
            np.linalg.norm(self._eval_embs[:, :d_raw], axis=1, keepdims=True) + 1e-8)
        pool_norm = pool_embs_raw / (
            np.linalg.norm(pool_embs_raw, axis=1, keepdims=True) + 1e-8)
        max_sim   = (pool_norm @ eval_norm.T).max(axis=1)
        diversity = 1.0 - max_sim

        acq = (mean_pred + self.beta * uncertainty
               + self.gamma * diversity - self.step_penalty * max_sim)
        return acq, mean_pred, uncertainty

    def embed(self, images: List[Image.Image]) -> np.ndarray:
        return self.embedder.embed(images)

    @property
    def eval_embs(self) -> Optional[np.ndarray]:
        return self._eval_embs

    @property
    def eval_rewards(self) -> Optional[np.ndarray]:
        return self._eval_rewards



class CLIPTextSurrogate:
    def __init__(
        self,
        clip,
        n_ensemble: int = 5,
        beta: float = 0.1,
        gamma: float = 0.1,
        step_penalty: float = 0.0,
        use_recency_weighting: bool = False,
        surrogate_window: Optional[int] = None,
        use_state_descriptor: bool = False,
        state_window: int = 20,
        n_pca: int = 50,
    ):
        self.clip                  = clip
        self.n_ensemble            = n_ensemble
        self.beta                  = beta
        self.gamma                 = gamma
        self.step_penalty          = step_penalty
        self.use_recency_weighting = use_recency_weighting
        self.surrogate_window      = surrogate_window
        self.use_state_descriptor  = use_state_descriptor
        self.state_window          = state_window
        self.n_pca                 = n_pca
        self._eval_embs:     Optional[np.ndarray] = None
        self._eval_rewards:  Optional[np.ndarray] = None
        self._current_state: Optional[np.ndarray] = None
        self._train_prompts: List[str]             = []
        self._sample_weights: Optional[np.ndarray] = None
        self.fit_log: List[dict] = []

    def fit(
        self,
        eval_prompts:   List[str],
        eval_rewards:   np.ndarray,
        history_gmean:  Optional[List[float]] = None,
        eval_indices:   Optional[List[int]]   = None,
        sample_weights: Optional[np.ndarray]  = None,
    ) -> None:
        prompts = list(eval_prompts)
        rewards = np.asarray(eval_rewards, dtype=np.float32)
        if self.surrogate_window is not None and len(prompts) > self.surrogate_window:
            prompts = prompts[-self.surrogate_window:]
            rewards = rewards[-self.surrogate_window:]
            if eval_indices   is not None:
                eval_indices   = list(eval_indices)[-self.surrogate_window:]
            if sample_weights is not None:
                sample_weights = sample_weights[-self.surrogate_window:]

        embs = self.clip.text_numpy(prompts)
        if self.use_state_descriptor and history_gmean is not None:
            states = np.stack([
                compute_state_descriptor(history_gmean, idx, self.state_window)
                for idx in (eval_indices or range(len(prompts)))
            ])
            embs = np.concatenate([embs, states], axis=1)

        self._train_prompts  = list(prompts)
        self._eval_embs      = embs
        self._eval_rewards   = rewards
        self._sample_weights = sample_weights
        if len(rewards) > 0:
            self.fit_log.append({
                "fit_index":   len(self.fit_log),
                "n_train":     len(rewards),
                "reward_mean": float(rewards.mean()),
                "reward_max":  float(rewards.max()),
            })

    def set_current_state(self, history_gmean: List[float], eval_count: int) -> None:
        if self.use_state_descriptor:
            self._current_state = compute_state_descriptor(
                history_gmean, eval_idx=eval_count, window=self.state_window
            )

    def acquisition_with_uncertainty(
        self, pool_prompts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._eval_embs is None:
            raise RuntimeError("Call fit() before acquisition_with_uncertainty().")

        from sklearn.linear_model import Ridge

        pool_embs_raw = self.clip.text_numpy(pool_prompts)
        if self.use_state_descriptor and self._current_state is not None:
            state_tiled = np.tile(self._current_state, (len(pool_embs_raw), 1))
            pool_embs   = np.concatenate([pool_embs_raw, state_tiled], axis=1)
        else:
            pool_embs = pool_embs_raw

        eval_embs    = self._eval_embs
        eval_rewards = self._eval_rewards
        n_train      = len(eval_rewards)
        d_raw        = pool_embs_raw.shape[1]

        X_train, X_pool, _, _ = _fit_pca_on_train(eval_embs, pool_embs, self.n_pca)

        weights = self._sample_weights
        if self.use_recency_weighting and weights is None:
            decay   = 0.02
            weights = np.exp(-decay * np.arange(n_train - 1, -1, -1)).astype(np.float32)
            weights = weights / weights.sum() * n_train

        preds = np.stack([
            Ridge(alpha=a).fit(X_train, eval_rewards,
                               sample_weight=weights).predict(X_pool)
            for a in np.logspace(-1, 2, self.n_ensemble)
        ])
        mean_pred, uncertainty = preds.mean(0), preds.std(0)

        eval_raw  = eval_embs[:, :d_raw]
        eval_norm = eval_raw / (np.linalg.norm(eval_raw, axis=1, keepdims=True) + 1e-8)
        pool_norm = pool_embs_raw / (
            np.linalg.norm(pool_embs_raw, axis=1, keepdims=True) + 1e-8)
        max_sim   = (pool_norm @ eval_norm.T).max(axis=1)
        diversity = 1.0 - max_sim

        acq = (mean_pred
               + self.beta         * uncertainty
               + self.gamma        * diversity
               - self.step_penalty * max_sim)
        return acq, mean_pred, uncertainty

    def acquisition(self, pool_prompts: List[str]) -> np.ndarray:
        acq, _, _ = self.acquisition_with_uncertainty(pool_prompts)
        return acq

    @property
    def eval_embs(self) -> Optional[np.ndarray]:
        return self._eval_embs

    @property
    def eval_rewards(self) -> Optional[np.ndarray]:
        return self._eval_rewards


class PairwiseCLIPSurrogate:
    def __init__(
        self,
        clip,
        mode:                  str   = "ranknet",
        n_ensemble:            int   = 5,
        beta:                  float = 0.1,
        gamma:                 float = 0.1,
        step_penalty:          float = 0.0,
        use_recency_weighting: bool  = False,
        surrogate_window:      Optional[int]  = None,
        use_state_descriptor:  bool  = False,
        state_window:          int   = 20,
        n_pca:                 int   = 50,
        max_pairs:             int   = 5000,  
    ):
        self.clip                  = clip
        self.mode                  = mode
        self.n_ensemble            = n_ensemble
        self.beta                  = beta
        self.gamma                 = gamma
        self.step_penalty          = step_penalty
        self.use_recency_weighting = use_recency_weighting
        self.surrogate_window      = surrogate_window
        self.use_state_descriptor  = use_state_descriptor
        self.state_window          = state_window
        self.n_pca                 = n_pca
        self.max_pairs             = max_pairs

        self._eval_embs:      Optional[np.ndarray] = None
        self._eval_rewards:   Optional[np.ndarray] = None
        self._current_state:  Optional[np.ndarray] = None
        self._sample_weights: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(42)
        self.fit_log: List[dict] = []


    def fit(
        self,
        eval_prompts:   List[str],
        eval_rewards:   np.ndarray,
        history_gmean:  Optional[List[float]] = None,
        eval_indices:   Optional[List[int]]   = None,
        sample_weights: Optional[np.ndarray]  = None,
    ) -> None:
        prompts = list(eval_prompts)
        rewards = np.asarray(eval_rewards, dtype=np.float32)
        if self.surrogate_window is not None and len(prompts) > self.surrogate_window:
            prompts = prompts[-self.surrogate_window:]
            rewards = rewards[-self.surrogate_window:]
            if eval_indices   is not None:
                eval_indices   = list(eval_indices)[-self.surrogate_window:]
            if sample_weights is not None:
                sample_weights = sample_weights[-self.surrogate_window:]

        embs = self.clip.text_numpy(prompts)
        if self.use_state_descriptor and history_gmean is not None:
            states = np.stack([
                compute_state_descriptor(history_gmean, idx, self.state_window)
                for idx in (eval_indices or range(len(prompts)))
            ])
            embs = np.concatenate([embs, states], axis=1)

        self._eval_embs      = embs
        self._eval_rewards   = rewards
        self._sample_weights = sample_weights
        if len(rewards) > 0:
            self.fit_log.append({
                "fit_index":   len(self.fit_log),
                "n_train":     len(rewards),
                "reward_mean": float(rewards.mean()),
                "reward_max":  float(rewards.max()),
            })

    def set_current_state(self, history_gmean: List[float], eval_count: int) -> None:
        if self.use_state_descriptor:
            self._current_state = compute_state_descriptor(
                history_gmean, eval_idx=eval_count, window=self.state_window
            )


    def acquisition_with_uncertainty(
        self, pool_prompts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._eval_embs is None:
            raise RuntimeError("Call fit() before acquisition_with_uncertainty().")

        pool_embs_raw = self.clip.text_numpy(pool_prompts)
        if self.use_state_descriptor and self._current_state is not None:
            state_tiled = np.tile(self._current_state, (len(pool_embs_raw), 1))
            pool_embs   = np.concatenate([pool_embs_raw, state_tiled], axis=1)
        else:
            pool_embs = pool_embs_raw

        eval_embs    = self._eval_embs
        eval_rewards = self._eval_rewards
        n_train      = len(eval_rewards)
        d_raw        = pool_embs_raw.shape[1]

        X_train, X_pool, _, _ = _fit_pca_on_train(eval_embs, pool_embs, self.n_pca)

        weights = self._sample_weights
        if self.use_recency_weighting and weights is None:
            decay   = 0.02
            weights = np.exp(-decay * np.arange(n_train - 1, -1, -1)).astype(np.float32)
            weights = weights / weights.sum() * n_train

        if self.mode == "ranknet":
            mean_pred, uncertainty = self._ranknet(
                X_train, eval_rewards, X_pool, weights
            )
        elif self.mode == "bt":
            mean_pred, uncertainty = self._bradley_terry(
                X_train, eval_rewards, X_pool, weights
            )
        else:
            raise ValueError(f"Unknown mode {self.mode!r}. Use 'ranknet' or 'bt'.")

        eval_raw  = eval_embs[:, :d_raw]
        eval_norm = eval_raw / (np.linalg.norm(eval_raw, axis=1, keepdims=True) + 1e-8)
        pool_norm = pool_embs_raw / (
            np.linalg.norm(pool_embs_raw, axis=1, keepdims=True) + 1e-8)
        max_sim   = (pool_norm @ eval_norm.T).max(axis=1)
        diversity = 1.0 - max_sim

        acq = (mean_pred
               + self.beta         * uncertainty
               + self.gamma        * diversity
               - self.step_penalty * max_sim)
        return acq, mean_pred, uncertainty

    def acquisition(self, pool_prompts: List[str]) -> np.ndarray:
        acq, _, _ = self.acquisition_with_uncertainty(pool_prompts)
        return acq


    def _ranknet(
        self,
        X_train:  np.ndarray,
        rewards:  np.ndarray,
        X_pool:   np.ndarray,
        weights:  Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ridge ensemble on rank-normalised targets.
        rankdata maps rewards → percentile in [0,1], removing absolute scale.
        Ensemble over logspace alphas gives uncertainty estimate.
        """
        from scipy.stats import rankdata
        from sklearn.linear_model import Ridge

        ranked = (rankdata(rewards, method="average") / len(rewards)).astype(np.float32)
        alphas = np.logspace(-1, 2, self.n_ensemble)
        preds  = np.stack([
            Ridge(alpha=a).fit(X_train, ranked,
                               sample_weight=weights).predict(X_pool)
            for a in alphas
        ])
        return preds.mean(0), preds.std(0)


    def _bradley_terry(
        self,
        X_train:  np.ndarray,
        rewards:  np.ndarray,
        X_pool:   np.ndarray,
        weights:  Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.linear_model import LogisticRegression

        n = len(rewards)
        ii, jj = np.triu_indices(n, k=1)
        diffs  = rewards[ii] - rewards[jj]

        nonzero = np.abs(diffs) > 1e-6
        ii, jj  = ii[nonzero], jj[nonzero]
        diffs   = diffs[nonzero]

        if len(ii) == 0:
            return self._ranknet(X_train, rewards, X_pool, weights)

        swap      = diffs < 0
        ii[swap], jj[swap] = jj[swap].copy(), ii[swap].copy()

        if len(ii) > self.max_pairs:
            abs_diffs = np.abs(rewards[ii] - rewards[jj])
            p         = abs_diffs / abs_diffs.sum()
            idx       = self._rng.choice(len(ii), size=self.max_pairs,
                                         replace=False, p=p)
            ii, jj = ii[idx], jj[idx]

        Z = X_train[ii] - X_train[jj] 
        y = np.ones(len(ii))

        pair_w = None
        if weights is not None:
            pair_w = 0.5 * (weights[ii] + weights[jj])

        C_values = np.logspace(-1, 1, self.n_ensemble)
        all_scores = []
        for C in C_values:
            try:
                clf = LogisticRegression(
                    C=C, fit_intercept=False, max_iter=1000,
                    solver="lbfgs", tol=1e-4,
                )
                clf.fit(Z, y, sample_weight=pair_w)
                all_scores.append(X_pool @ clf.coef_[0])
            except Exception:
                continue

        if not all_scores:
            return self._ranknet(X_train, rewards, X_pool, weights)

        preds = np.stack(all_scores)
        return preds.mean(0), preds.std(0)

    @property
    def eval_embs(self) -> Optional[np.ndarray]:
        return self._eval_embs

    @property
    def eval_rewards(self) -> Optional[np.ndarray]:
        return self._eval_rewards