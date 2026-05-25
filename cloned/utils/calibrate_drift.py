from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from cloned.algorithms.embedder import CLIPTextEmbedder
from cloned.generators.sdxl_turbo import SDXLTurboGenerator
from cloned.rewards.blackbox_rewards import BERGFMRINSDFsaverageHuzeROIReward
from cloned.spaces.base import Candidate
from cloned.spaces.structured_art_space import StructuredArtPromptSpace, art_data, flatten_art_data

BERG_FMRI_DIR = "./berg_models"
OUT_DIR       = "./search/clean_search/calibration"
DEVICE        = "cuda"

FMRI_ROIS = [
    "early", "lateral", "EBA", "FBA-1", "FFA-1", "OFA", "OPA",
    "PPA", "RSC", "V1v", "VWFA-1", "aTL-faces",
]

FMRI_SUBJECTS = [1]

N_CALIBRATION = 1000
BATCH_SIZE    = 16
SEED          = 0


def sample_prompts(space, n: int, seed: int) -> list[str]:
    random.seed(seed)
    seen: set[str] = set()
    prompts: list[str] = []
    for _ in range(n * 20):
        if len(prompts) >= n:
            break
        ind = list(space.random_candidate().genes)
        p = space.decode(Candidate(tuple(ind)))
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    if len(prompts) < n:
        print(f"  [warn] only {len(prompts)} unique prompts (wanted {n})")
    return prompts


def generate_all_images(generator, prompts, batch_size, seed):
    images = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = prompts[i * batch_size: (i + 1) * batch_size]
        imgs = generator.generate(batch, seed=seed + i)
        images.extend(imgs)
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"    generated [{i+1}/{n_batches} batches]  ({len(images)} images)")
    return images


def score_all(reward, images, batch_size) -> np.ndarray:
    scores = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = images[i * batch_size: (i + 1) * batch_size]
        scores.extend(reward.score(batch))
    return np.array(scores, dtype=np.float32)


def stats(arr: np.ndarray) -> dict:
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")


def main() -> None:
    device = DEVICE if torch.cuda.is_available() else "cpu"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  calibrate_drift_all.py")
    print("=" * 65)
    print(f"  device          : {device}")
    print(f"  N calibration   : {N_CALIBRATION}")
    print(f"  fMRI subjects   : {FMRI_SUBJECTS}")
    print("=" * 65)

    print("\n[1/3] Building prompt space...")
    _, options = flatten_art_data(art_data)
    text_embedder = CLIPTextEmbedder(device=device)
    option_embs = [torch.from_numpy(text_embedder.batch_numpy(opts)) for opts in options]
    space = StructuredArtPromptSpace(art_data=art_data, option_embeddings=option_embs)

    print("\n[2/3] Sampling calibration prompts...")
    prompts = sample_prompts(space, N_CALIBRATION, seed=SEED)
    print(f"  {len(prompts)} unique prompts sampled")

    print("\n[3/3] Generating images (shared pool)...")
    generator = SDXLTurboGenerator(device=device, batch_size=BATCH_SIZE)
    images = generate_all_images(generator, prompts, BATCH_SIZE, seed=SEED)
    print(f"  {len(images)} images ready")

    print("\n[4/4] fMRI calibration...")
    for subject in FMRI_SUBJECTS:
        out_path = out_dir / f"fmri_calibration_subj{subject}.json"
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            roi_stats = existing.get("rois", {})
            print(f"\n  Subject {subject} — resuming ({len(roi_stats)} ROIs already done)")
        else:
            roi_stats = {}
            print(f"\n  Subject {subject}")

        for roi in FMRI_ROIS:
            if roi in roi_stats:
                print(f"    skip ROI={roi}")
                continue
            print(f"    scoring ROI={roi} ...", end=" ", flush=True)
            reward = BERGFMRINSDFsaverageHuzeROIReward(
                berg_dir=BERG_FMRI_DIR, subject=subject, roi=roi,
                device="auto", pooling="topk_mean", topk_frac=0.1,
                combine_hemispheres="concat",
            )
            sc = score_all(reward, images, BATCH_SIZE)
            roi_stats[roi] = stats(sc)
            print(f"mean={sc.mean():.4f}  std={sc.std():.4f}")
            save_json(out_path, {
                "subject": subject, "rois": roi_stats,
                "n_samples": len(prompts), "seed": SEED,
            })

    print("\n" + "=" * 65)
    print("  Done. Files written:")
    for p in sorted(out_dir.glob("*.json")):
        print(f"    {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()