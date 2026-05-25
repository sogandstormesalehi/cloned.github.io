from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


def _as_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "last_hidden_state"):
        if hasattr(x, attr):
            t = getattr(x, attr)
            if isinstance(t, torch.Tensor):
                return t
    raise TypeError(f"Unexpected CLIP output type: {type(x)}")


def _pool_response_1d(
    x: np.ndarray,
    *,
    pooling: str = "topk_mean",
    topk_frac: float = 0.1,
    eps: float = 1e-8,
) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    if pooling == "mean":
        return float(x.mean())
    if pooling == "topk_mean":
        k = max(1, int(round(topk_frac * x.size)))
        vals = np.partition(x, -k)[-k:]
        return float(vals.mean())
    if pooling == "l2":
        return float(np.linalg.norm(x) / (np.sqrt(x.size) + eps))
    raise ValueError(f"Unknown pooling={pooling}")


class _BERGImageMixin:
    def _pil_to_nchw_uint8(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        x = np.asarray(im, dtype=np.uint8)
        x = np.transpose(x, (2, 0, 1))
        return x[None, ...]

    def _batch_to_nchw_uint8(self, images) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, 3, self.image_size, self.image_size), dtype=np.uint8)
        return np.concatenate([self._pil_to_nchw_uint8(im) for im in images], axis=0)


class CLIPBlackboxReward:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        *,
        device: str | None = None,
        use_safetensors: bool = True,
        use_fast_processor: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=use_safetensors).to(device)
        self.model.eval()

        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_id,
                use_safetensors=use_safetensors,
                use_fast=use_fast_processor,
            )
        except TypeError:
            self.processor = CLIPProcessor.from_pretrained(
                model_id,
                use_safetensors=use_safetensors,
            )

    @torch.no_grad()
    def text_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), images=None, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.get_text_features(**inputs)
        feats = _as_tensor(out)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu()

    @torch.no_grad()
    def image_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(text=[" "], images=image, return_tensors="pt")
        inputs = {
            k: (v.to(self.device).float() if torch.is_floating_point(v) else v.to(self.device))
            for k, v in inputs.items()
        }
        out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = _as_tensor(out)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def reward(self, image: Image.Image, target_image: Image.Image) -> float:
        img_feat = self.image_embedding(image)
        tgt_feat = self.image_embedding(target_image)
        return float((img_feat @ tgt_feat.T).item())


class LowContrastLowLightReward:
    accepts_target = False
    requires_target = False

    def __init__(
        self,
        *,
        w_contrast: float = 0.5,
        w_light: float = 0.5,
        gamma: float = 1.0,
        eps: float = 1e-8,
        noise_prob: float = 0.1,
    ):
        self.w_contrast = float(w_contrast)
        self.w_light = float(w_light)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.noise_prob = float(noise_prob)

    def name(self) -> str:
        return "LowContrastLowLightReward"

    def _luminance01(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        if np.random.rand() < self.noise_prob:
            return float(np.random.rand())

        y = self._luminance01(image)
        mean_lum = float(y.mean())
        std_lum = float(y.std())

        std_norm = np.clip(std_lum / 0.5, 0.0, 1.0)
        low_contrast_score = 1.0 - std_norm
        low_light_score = 1.0 - mean_lum

        score = self.w_contrast * low_contrast_score + self.w_light * low_light_score

        if abs(self.gamma - 1.0) > self.eps:
            score = float(np.power(np.clip(score, 0.0, 1.0), self.gamma))

        return float(np.clip(score, 0.0, 1.0))


class PupilProxyReward:
    accepts_target = False
    requires_target = False

    _AROUSAL_PROMPTS = [
        "terrifying", "shocking", "creepy", "exciting",
        "alarming", "intense", "disturbing", "wild chaos",
    ]

    _CALM_PROMPTS = [
        "calm", "peaceful", "boring", "quiet", "serene", "still",
    ]

    def __init__(
        self,
        clip_reward: CLIPBlackboxReward,
        *,
        w_luminance: float = 0.45,
        w_contrast: float = 0.35,
        w_arousal: float = 0.20,
        noise_std: float = 0.04,
        gamma: float = 1.0,
        eps: float = 1e-8,
    ):
        assert abs(w_luminance + w_contrast + w_arousal - 1.0) < 1e-6, \
            "Weights must sum to 1"

        self.clip = clip_reward
        self.w_luminance = w_luminance
        self.w_contrast = w_contrast
        self.w_arousal = w_arousal
        self.noise_std = noise_std
        self.gamma = gamma
        self.eps = eps

        self._arousal_emb = self.clip.text_embeddings(self._AROUSAL_PROMPTS).mean(dim=0)
        self._arousal_emb = self._arousal_emb / self._arousal_emb.norm()

        self._calm_emb = self.clip.text_embeddings(self._CALM_PROMPTS).mean(dim=0)
        self._calm_emb = self._calm_emb / self._calm_emb.norm()

    def name(self) -> str:
        return "PupilProxyReward"

    def _luminance01(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _pixel_score(self, image: Image.Image) -> tuple[float, float]:
        y = self._luminance01(image)
        mean_lum = float(y.mean())
        std_lum = float(y.std())

        low_lum_score = 1.0 - mean_lum
        rms_contrast = std_lum / (mean_lum + self.eps)
        low_contrast_score = 1.0 - float(np.clip(rms_contrast / 2.0, 0.0, 1.0))

        return low_lum_score, low_contrast_score

    def _arousal_score(self, image: Image.Image) -> float:
        img_emb = self.clip.image_embedding(image).cpu()
        img_emb = img_emb.squeeze(0)

        sim_arousal = float((img_emb @ self._arousal_emb).item())
        sim_calm = float((img_emb @ self._calm_emb).item())

        raw = sim_arousal - sim_calm
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    @torch.no_grad()
    def score(self, images: list[Image.Image]) -> list[float]:
        scores = []
        for im in images:
            low_lum, low_contrast = self._pixel_score(im)
            arousal = self._arousal_score(im)

            s = (
                self.w_luminance * low_lum
                + self.w_contrast * low_contrast
                + self.w_arousal * arousal
            )
            if abs(self.gamma - 1.0) > self.eps:
                s = float(np.power(np.clip(s, 0.0, 1.0), self.gamma))
            if self.noise_std > 0:
                s += float(np.random.normal(0.0, self.noise_std))
            scores.append(float(np.clip(s, 0.0, 1.0)))
        return scores

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        return self.score([image])[0]


EMONET_CATEGORIES = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness",
    "adoration", "aesthetic_appreciation", "anxiety", "boredom",
    "calmness", "confusion", "craving", "empathic_pain",
    "entrancement", "horror", "interest", "nostalgia",
]

EMONET_AROUSAL_WEIGHTS = {
    "amusement":              0.55,
    "awe":                    0.65,
    "contentment":            0.20,
    "excitement":             0.90,
    "anger":                  0.85,
    "disgust":                0.70,
    "fear":                   0.88,
    "sadness":                0.35,
    "adoration":              0.40,
    "aesthetic_appreciation": 0.45,
    "anxiety":                0.80,
    "boredom":                0.05,
    "calmness":               0.05,
    "confusion":              0.55,
    "craving":                0.60,
    "empathic_pain":          0.65,
    "entrancement":           0.50,
    "horror":                 0.92,
    "interest":               0.55,
    "nostalgia":              0.30,
}

_AROUSAL_VECTOR = torch.tensor(
    [EMONET_AROUSAL_WEIGHTS[c] for c in EMONET_CATEGORIES],
    dtype=torch.float32,
)

_EMONET_TRANSFORM = T.Compose([
    T.Resize((227, 227)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_emonet(repo_path: str, device: str):
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        from models import EmoNet
    except ImportError as e:
        raise ImportError(
            f"Could not import EmoNet from {repo_path}/models.py\n"
            f"Clone https://github.com/ecco-laboratory/emonet-pytorch first.\n"
            f"Original error: {e}"
        )
    model = EmoNet()
    model.load_state_dict_from_web()
    model.eval()
    model.to(device)
    print(f"[PupilEmoNetReward] EmoNet loaded (Kragel 2019) on {device}")
    return model


class PupilEmoNetReward:
    accepts_target = False
    requires_target = False

    def __init__(
        self,
        emonet_repo_path: str,
        device: str = "cuda",
        w_luminance: float = 0.55,
        w_arousal: float = 0.30,
        w_contrast: float = 0.15,
        noise_std: float = 0.04,
        gamma: float = 1.0,
        eps: float = 1e-8,
    ):
        assert abs(w_luminance + w_arousal + w_contrast - 1.0) < 1e-5, \
            f"Weights must sum to 1, got {w_luminance + w_arousal + w_contrast:.4f}"

        self.device = device
        self.w_luminance = w_luminance
        self.w_arousal = w_arousal
        self.w_contrast = w_contrast
        self.noise_std = noise_std
        self.gamma = gamma
        self.eps = eps

        self._emonet = _load_emonet(emonet_repo_path, device)
        self._arousal_vec = _AROUSAL_VECTOR.to(device)

    def name(self) -> str:
        return "PupilEmoNetReward"

    def _pixel_scores(self, image: Image.Image) -> tuple[float, float]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        mean_lum = float(y.mean())
        rms_c = float(y.std()) / (mean_lum + self.eps)
        return (
            1.0 - mean_lum,
            1.0 - float(np.clip(rms_c / 2.0, 0.0, 1.0)),
        )

    @torch.no_grad()
    def _emonet_arousal_batch(self, images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack(
            [_EMONET_TRANSFORM(img.convert("RGB")) for img in images]
        ).to(self.device)
        logits = self._emonet(tensors)
        probs = torch.softmax(logits, dim=-1)
        arousal = (probs * self._arousal_vec).sum(dim=-1)
        return arousal.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def score(self, images: List[Image.Image]) -> List[float]:
        lum = np.zeros(len(images), dtype=np.float32)
        con = np.zeros(len(images), dtype=np.float32)
        for i, img in enumerate(images):
            lum[i], con[i] = self._pixel_scores(img)

        arous = self._emonet_arousal_batch(images)

        s = self.w_luminance * lum + self.w_arousal * arous + self.w_contrast * con
        if abs(self.gamma - 1.0) > self.eps:
            s = np.power(np.clip(s, 0.0, 1.0), self.gamma)
        if self.noise_std > 0:
            s = s + np.random.normal(0.0, self.noise_std, size=len(images))
        return np.clip(s, 0.0, 1.0).tolist()

    def reward(self, image: Image.Image, target_image: Optional[Image.Image] = None) -> float:
        return self.score([image])[0]


class DINOv2BlackboxReward:
    accepts_target = True
    requires_target = True

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        *,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def name(self) -> str:
        return f"DINOv2BlackboxReward({self.model.config._name_or_path})"

    @torch.no_grad()
    def image_embedding(self, image: Image.Image) -> torch.Tensor:
        im = image.convert("RGB")
        inputs = self.processor(images=im, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state[:, 0, :]
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        return feats

    @torch.no_grad()
    def reward(self, image: Image.Image, target_image: Image.Image) -> float:
        if target_image is None:
            raise ValueError("DINOv2BlackboxReward requires target_image (got None).")
        a = self.image_embedding(image)
        b = self.image_embedding(target_image)
        return float((a @ b.T).item())


class BERGBrainScoreVisionROIReward(_BERGImageMixin):
    accepts_target = False
    requires_target = False

    def __init__(
        self,
        *,
        berg_dir: str,
        model_id: str = "brainscore_vision-alexnet",
        roi: str = "IT",
        device: str = "auto",
        image_size: int = 224,
        pooling: str = "topk_mean",
        topk_frac: float = 0.1,
        eps: float = 1e-8,
    ):
        from berg import BERG

        self.berg_dir = os.path.expanduser(berg_dir)
        self.model_id = model_id
        self.roi = roi
        self.device = device
        self.image_size = int(image_size)
        self.pooling = pooling
        self.topk_frac = float(topk_frac)
        self.eps = float(eps)

        self.berg = BERG(self.berg_dir)
        self.model = self.berg.get_encoding_model(
            self.model_id,
            selection={"roi": self.roi},
            device=self.device,
        )

    def _score_resp(self, resp_bt: np.ndarray) -> list[float]:
        return [
            _pool_response_1d(r, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
            for r in resp_bt
        ]

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        stim = self._pil_to_nchw_uint8(image)
        resp = self.berg.encode(self.model, stim)
        return self._score_resp(resp)[0]

    def score(self, images):
        if len(images) == 0:
            return []
        stims = self._batch_to_nchw_uint8(images)
        resp = self.berg.encode(self.model, stims)
        return self._score_resp(resp)


class BERGUtahArrayROIReward(_BERGImageMixin):
    accepts_target = False
    requires_target = False

    ROI_DEFAULT_WINDOWS_MS = {
        "V1": (25, 125),
        "V4": (50, 150),
        "IT": (75, 175),
    }

    def __init__(
        self,
        *,
        berg_dir: str,
        subject: str = "N",
        roi: str = "V4",
        train_splits: str = "single",
        device: str = "auto",
        image_size: int = 224,
        pooling: str = "topk_mean",
        topk_frac: float = 0.1,
        time_window_ms: tuple[int, int] | None = None,
        eps: float = 1e-8,
    ):
        from berg import BERG

        self.berg_dir = os.path.expanduser(berg_dir)
        self.subject = subject
        self.roi = roi
        self.train_splits = train_splits
        self.device = device
        self.image_size = int(image_size)
        self.pooling = pooling
        self.topk_frac = float(topk_frac)
        self.eps = float(eps)

        if time_window_ms is None:
            time_window_ms = self.ROI_DEFAULT_WINDOWS_MS[self.roi]
        self.time_window_ms = time_window_ms

        self.berg = BERG(self.berg_dir)
        self.model = self.berg.get_encoding_model(
            "utah_array-tvsd-vit_b_32",
            subject=self.subject,
            train_splits=self.train_splits,
            selection={"roi": [self.roi]},
            device=self.device,
        )
        self.metadata = self.berg.get_model_metadata(
            "utah_array-tvsd-vit_b_32",
            subject=self.subject,
        )
        self.times = np.asarray(self.metadata["utah_array"]["times"], dtype=np.float32)

    def _time_mask(self) -> np.ndarray:
        t0, t1 = self.time_window_ms
        mask = (self.times >= t0) & (self.times <= t1)
        if not np.any(mask):
            raise ValueError(f"No Utah timepoints in window {self.time_window_ms}")
        return mask

    def _score_resp(self, resp) -> list[float]:
        tmask = self._time_mask()
        out = []
        if resp.ndim == 4:
            resp = resp.mean(axis=1)
        for b in range(resp.shape[0]):
            x = resp[b][:, tmask]
            x = x.mean(axis=-1)
            out.append(
                _pool_response_1d(x, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
            )
        return out

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        stim = self._pil_to_nchw_uint8(image)
        resp = self.berg.encode(self.model, stim)
        return self._score_resp(resp)[0]

    def score(self, images):
        if len(images) == 0:
            return []
        stims = self._batch_to_nchw_uint8(images)
        resp = self.berg.encode(self.model, stims)
        return self._score_resp(resp)


class BERGFMRINSDFWRFROIReward(_BERGImageMixin):
    accepts_target = False
    requires_target = False

    def __init__(
        self,
        *,
        berg_dir: str,
        subject: int = 1,
        roi: str = "hV4",
        device: str = "auto",
        image_size: int = 224,
        pooling: str = "topk_mean",
        topk_frac: float = 0.1,
        eps: float = 1e-8,
    ):
        from berg import BERG

        self.berg_dir = os.path.expanduser(berg_dir)
        self.subject = int(subject)
        self.roi = roi
        self.device = device
        self.image_size = int(image_size)
        self.pooling = pooling
        self.topk_frac = float(topk_frac)
        self.eps = float(eps)

        self.berg = BERG(self.berg_dir)
        self.model = self.berg.get_encoding_model(
            "fmri-nsd-fwrf",
            subject=self.subject,
            selection={"roi": self.roi},
            device=self.device,
        )

    def _score_resp(self, resp_bv: np.ndarray) -> list[float]:
        return [
            _pool_response_1d(r, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
            for r in resp_bv
        ]

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        stim = self._pil_to_nchw_uint8(image)
        resp = self.berg.encode(self.model, stim)
        return self._score_resp(resp)[0]

    def score(self, images):
        if len(images) == 0:
            return []
        stims = self._batch_to_nchw_uint8(images)
        resp = self.berg.encode(self.model, stims)
        return self._score_resp(resp)


class BERGFMRINSDFsaverageHuzeROIReward(_BERGImageMixin):
    accepts_target = False
    requires_target = False

    def __init__(
        self,
        *,
        berg_dir: str,
        subject: int = 1,
        roi: str = "hV4",
        device: str = "auto",
        image_size: int = 224,
        pooling: str = "topk_mean",
        topk_frac: float = 0.1,
        combine_hemispheres: str = "concat",
        eps: float = 1e-8,
    ):
        from berg import BERG

        self.berg_dir = os.path.expanduser(berg_dir)
        self.subject = int(subject)
        self.roi = roi
        self.device = device
        self.image_size = int(image_size)
        self.pooling = pooling
        self.topk_frac = float(topk_frac)
        self.combine_hemispheres = combine_hemispheres
        self.eps = float(eps)

        self.berg = BERG(self.berg_dir)
        self.model = self.berg.get_encoding_model(
            "fmri-nsd_fsaverage-huze",
            subject=self.subject,
            selection={"roi": self.roi},
            device=self.device,
        )

    def _score_resp(self, resp_tuple) -> list[float]:
        lh, rh = resp_tuple
        out = []
        for b in range(lh.shape[0]):
            l = np.asarray(lh[b], dtype=np.float32).reshape(-1)
            r = np.asarray(rh[b], dtype=np.float32).reshape(-1)

            if self.combine_hemispheres == "concat":
                x = np.concatenate([l, r], axis=0)
                s = _pool_response_1d(x, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
            elif self.combine_hemispheres == "mean":
                sl = _pool_response_1d(l, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
                sr = _pool_response_1d(r, pooling=self.pooling, topk_frac=self.topk_frac, eps=self.eps)
                s = 0.5 * (sl + sr)
            else:
                raise ValueError(f"Unknown combine_hemispheres={self.combine_hemispheres}")

            out.append(float(s))
        return out

    def reward(self, image: Image.Image, target_image: Image.Image | None = None) -> float:
        stim = self._pil_to_nchw_uint8(image)
        resp = self.berg.encode(self.model, stim)
        return self._score_resp(resp)[0]

    def score(self, images):
        if len(images) == 0:
            return []
        stims = self._batch_to_nchw_uint8(images)
        resp = self.berg.encode(self.model, stims)
        return self._score_resp(resp)


class BERGTHINGSEEG2Reward(_BERGImageMixin):
    accepts_target = False
    requires_target = False

    CHANNEL_GROUPS = {
        "occipital":         ["O1", "Oz", "O2"],
        "parietal":          ["P7", "P3", "Pz", "P4", "P8", "P1", "P2", "P5", "P6"],
        "parieto_occipital": ["PO7", "PO3", "POz", "PO4", "PO8"],
        "central":           ["C3", "Cz", "C4", "C1", "C2", "C5", "C6"],
        "frontal":           ["Fp1", "Fp2", "F3", "F4", "F7", "F8",
                              "F1", "F2", "F5", "F6", "AF3", "AF4", "AF7", "AF8", "AFz"],
        "frontocentral":     ["FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz"],
        "temporoparietal":   ["T7", "T8", "TP7", "TP8", "TP9", "TP10"],
        "visual_default":    ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"],
        "oz_only":           ["Oz"],
        "right_parietal":    ["P6", "P8", "P4", "PO8"],
        "visual_peak":       ["Oz", "P6", "P8", "PO8", "POz", "O2", "PO7"],
    }

    def __init__(
        self,
        *,
        berg_dir: str,
        subject: int = 1,
        device: str = "auto",
        image_size: int = 224,
        channels: list = None,
        channel_group: str = "visual_default",
        time_window_ms: tuple = (0.075, 0.175),
        pooling: str = "topk_mean",
        topk_frac: float = 0.3,
        average_repeats: bool = True,
        use_abs: bool = True,
        eps: float = 1e-8,
    ):
        from berg import BERG

        if channels is not None and channel_group is not None:
            raise ValueError("Provide either channels or channel_group, not both.")

        self.berg_dir = os.path.expanduser(berg_dir)
        self.subject = int(subject)
        self.device = device
        self.image_size = int(image_size)
        self.pooling = pooling
        self.topk_frac = float(topk_frac)
        self.average_repeats = bool(average_repeats)
        self.use_abs = bool(use_abs)
        self.eps = float(eps)
        self.time_window_ms = time_window_ms
        self.channel_group = channel_group

        if channels is None and channel_group is not None:
            if channel_group not in self.CHANNEL_GROUPS:
                raise ValueError(
                    f"Unknown channel_group={channel_group!r}. "
                    f"Valid: {sorted(self.CHANNEL_GROUPS.keys())}"
                )
            channels = list(self.CHANNEL_GROUPS[channel_group])
        self.channels = channels

        self.berg = BERG(self.berg_dir)
        self.model = self.berg.get_encoding_model(
            "eeg-things_eeg_2-vit_b_32",
            subject=self.subject,
            selection={"channels": self.channels} if self.channels else None,
            device=self.device,
        )
        self.metadata = self.berg.get_model_metadata(
            "eeg-things_eeg_2-vit_b_32",
            subject=self.subject,
        )
        self.times = np.asarray(self.metadata["eeg"]["times"], dtype=np.float32)

    def name(self) -> str:
        ch = self.channel_group or "custom"
        tw = (
            f"{self.time_window_ms[0] * 1000:.0f}_{self.time_window_ms[1] * 1000:.0f}ms"
            if self.time_window_ms else "all"
        )
        ab = "abs" if self.use_abs else "signed"
        return f"BERGTHINGSEEG2({ch},{tw},{self.pooling},{ab},sub={self.subject})"

    def _time_mask(self) -> np.ndarray:
        if self.time_window_ms is None:
            return np.ones_like(self.times, dtype=bool)
        t0, t1 = self.time_window_ms
        mask = (self.times >= t0) & (self.times <= t1)
        if not np.any(mask):
            raise ValueError(
                f"No timepoints in window {self.time_window_ms} (seconds). "
                f"Available range: [{self.times.min():.3f}, {self.times.max():.3f}]"
            )
        return mask

    def _pool_1d(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0
        if self.use_abs:
            x = np.abs(x)
        if self.pooling == "mean":
            return float(x.mean())
        if self.pooling == "topk_mean":
            k = max(1, int(round(self.topk_frac * x.size)))
            return float(np.partition(x, -k)[-k:].mean())
        if self.pooling == "l2":
            return float(np.linalg.norm(x) / (np.sqrt(x.size) + self.eps))
        raise ValueError(f"Unknown pooling={self.pooling!r}")

    def _score_resp(self, resp: np.ndarray) -> list:
        resp = np.asarray(resp, dtype=np.float32)
        assert resp.ndim == 4, f"Expected (B,R,C,T), got {resp.shape}"
        tmask = self._time_mask()
        if self.average_repeats:
            resp = resp.mean(axis=1)
            return [self._pool_1d(resp[b][:, tmask]) for b in range(resp.shape[0])]
        out = []
        for b in range(resp.shape[0]):
            xb = resp[b][:, :, tmask]
            out.append(float(np.mean([self._pool_1d(xb[r]) for r in range(xb.shape[0])])))
        return out

    def reward(self, image, target_image=None) -> float:
        return self.score([image])[0]

    def score(self, images: list) -> list:
        if not images:
            return []
        stims = self._batch_to_nchw_uint8(images)
        resp = self.berg.encode(self.model, stims)
        return self._score_resp(resp)