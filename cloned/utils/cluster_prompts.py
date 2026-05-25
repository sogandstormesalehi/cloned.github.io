from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from cloned.algorithms.embedder import CLIPTextEmbedder
from cloned.spaces.base import Candidate
from cloned.spaces.structured_art_space import StructuredArtPromptSpace, art_data, flatten_art_data

CLUSTER_OUT_PATH = "./search/real_time_search/AdaptivePersonalization/prompt_clusters.npz"
DEVICE           = "cuda"

N_PROMPTS        = 500_000
N_PCA_COMPONENTS = 50
BATCH_SIZE       = 512
SEED             = 0

N_COARSE         = 20
N_MID_PER_COARSE = 5
N_LEAF_PER_MID   = 5


def sample_prompts(space, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    prompts: list[str] = []
    for _ in range(n * 5):
        if len(prompts) >= n:
            break
        ind = [rng.randrange(len(opts)) for opts in space.options]
        p = space.decode(Candidate(tuple(ind)))
        if p not in seen:
            seen.add(p)
            prompts.append(p)
            if len(prompts) % 50_000 == 0:
                print(f"  sampled {len(prompts):,}")
    if len(prompts) < n:
        print(f"  warning: only {len(prompts):,} unique prompts (space may be smaller)")
    return prompts


def embed_prompts(embedder, prompts: list[str], batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(prompts), batch_size):
        out.append(embedder.batch_numpy(prompts[i:i + batch_size]).astype(np.float32))
        if (i // batch_size + 1) % 50 == 0:
            print(f"  embedded {min(i + batch_size, len(prompts)):,} / {len(prompts):,}")
    return np.concatenate(out, axis=0)


def kmeans(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(X))
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed,
        batch_size=min(4096, len(X)), n_init=10, max_iter=300,
    )
    labels = km.fit_predict(X).astype(np.int32)
    centroids = km.cluster_centers_.astype(np.float32)
    return labels, centroids


def build_tree(
    embs_pca: np.ndarray,
    n_coarse: int,
    n_mid_per_coarse: int,
    n_leaf_per_mid: int,
    seed: int,
) -> dict:
    N = len(embs_pca)
    print(f"\nBuilding tree: {n_coarse} coarse × {n_mid_per_coarse} mid "
          f"× {n_leaf_per_mid} leaf = "
          f"{n_coarse * n_mid_per_coarse * n_leaf_per_mid} leaves")

    node_parent: list[int] = []
    node_level: list[int] = []
    node_centroid: list[np.ndarray] = []
    node_children: list[list[int]] = []
    node_prompt_indices: list[np.ndarray] = []

    prompt_leaf = np.full(N, -1, dtype=np.int32)

    def add_node(parent_idx: int, level: int, centroid: np.ndarray) -> int:
        idx = len(node_parent)
        node_parent.append(parent_idx)
        node_level.append(level)
        node_centroid.append(centroid)
        node_children.append([])
        node_prompt_indices.append(np.empty(0, dtype=np.int32))
        if parent_idx >= 0:
            node_children[parent_idx].append(idx)
        return idx

    root_centroid = embs_pca.mean(axis=0).astype(np.float32)
    root_idx = add_node(-1, 0, root_centroid)

    t0 = time.time()
    coarse_labels, coarse_cents = kmeans(embs_pca, n_coarse, seed)
    print(f"  level 1 done ({time.time() - t0:.1f}s) — "
          f"sizes {np.bincount(coarse_labels).tolist()}")

    coarse_node_ids = []
    for c in range(n_coarse):
        nid = add_node(root_idx, 1, coarse_cents[c])
        coarse_node_ids.append(nid)

    t0 = time.time()
    mid_labels_global = np.full(N, -1, dtype=np.int32)
    global_mid = 0
    for c in range(n_coarse):
        mask = coarse_labels == c
        idxs = np.where(mask)[0]
        X_sub = embs_pca[idxs]
        local_labels, mid_cents = kmeans(X_sub, n_mid_per_coarse, seed + c)
        for m in range(len(mid_cents)):
            add_node(coarse_node_ids[c], 2, mid_cents[m])
            sub_mask = local_labels == m
            mid_labels_global[idxs[sub_mask]] = global_mid
            global_mid += 1
    print(f"  level 2 done ({time.time() - t0:.1f}s) — {global_mid} mid nodes")

    t0 = time.time()
    n_mid_total = global_mid
    leaf_centroids_list = []
    leaf_node_ids: list[int] = []
    global_leaf = 0

    for mid_idx in range(n_mid_total):
        mask = mid_labels_global == mid_idx
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        X_sub = embs_pca[idxs]
        local_labels, leaf_cents = kmeans(X_sub, n_leaf_per_mid, seed + mid_idx)
        parent_nid = 1 + n_coarse + mid_idx
        for lf in range(len(leaf_cents)):
            nid = add_node(parent_nid, 3, leaf_cents[lf])
            leaf_node_ids.append(nid)
            leaf_centroids_list.append(leaf_cents[lf])
            sub_mask = local_labels == lf
            prompt_idxs = idxs[sub_mask].astype(np.int32)
            node_prompt_indices[nid] = prompt_idxs
            prompt_leaf[prompt_idxs] = global_leaf
            global_leaf += 1

    print(f"  level 3 done ({time.time() - t0:.1f}s) — {global_leaf} leaf nodes")
    assert (prompt_leaf >= 0).all(), "Some prompts were not assigned to a leaf"

    M = len(node_parent)
    tree_parent = np.array(node_parent, dtype=np.int32)
    tree_level = np.array(node_level, dtype=np.int32)
    tree_centroid = np.stack(node_centroid).astype(np.float32)
    tree_children = np.empty(M, dtype=object)
    tree_leaf_prompts = np.empty(M, dtype=object)
    for i in range(M):
        tree_children[i] = np.array(node_children[i], dtype=np.int32)
        tree_leaf_prompts[i] = node_prompt_indices[i]

    centroids_pca = np.stack(leaf_centroids_list).astype(np.float32)

    print(f"\nTree summary:")
    print(f"  total nodes : {M}")
    for lvl, name in [(0, "root"), (1, "coarse"), (2, "mid"), (3, "leaf")]:
        count = int((tree_level == lvl).sum())
        print(f"  level {lvl} ({name:6s}) : {count} nodes")
    sizes = np.array([len(tree_leaf_prompts[nid]) for nid in leaf_node_ids])
    print(f"  leaf sizes  : min={sizes.min()}  max={sizes.max()}  mean={sizes.mean():.0f}")

    return {
        "prompt_leaf":       prompt_leaf,
        "centroids_pca":     centroids_pca,
        "tree_node_count":   np.int32(M),
        "tree_parent":       tree_parent,
        "tree_level":        tree_level,
        "tree_centroid":     tree_centroid,
        "tree_children":     tree_children,
        "tree_leaf_prompts": tree_leaf_prompts,
    }


def main():
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device         : {device}")
    print(f"N prompts      : {N_PROMPTS:,}")
    print(f"PCA components : {N_PCA_COMPONENTS}")
    print(f"Tree shape     : {N_COARSE} × {N_MID_PER_COARSE} × {N_LEAF_PER_MID}")
    print(f"Output         : {CLUSTER_OUT_PATH}\n")

    _, options = flatten_art_data(art_data)
    embedder = CLIPTextEmbedder(device=device)
    option_embs = [torch.from_numpy(embedder.batch_numpy(opts)) for opts in options]
    space = StructuredArtPromptSpace(art_data=art_data, option_embeddings=option_embs)

    print("Sampling prompts...")
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    prompts = sample_prompts(space, N_PROMPTS, seed=SEED)
    print(f"  {len(prompts):,} prompts in {time.time() - t0:.1f}s\n")

    print("Embedding with CLIP...")
    t0 = time.time()
    embs_raw = embed_prompts(embedder, prompts, BATCH_SIZE)
    print(f"  shape {embs_raw.shape}  in {time.time() - t0:.1f}s\n")

    embs_norm = normalize(embs_raw, norm="l2")

    print(f"PCA → {N_PCA_COMPONENTS} components...")
    t0 = time.time()
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
    embs_pca = pca.fit_transform(embs_norm).astype(np.float32)
    print(f"  explained variance {pca.explained_variance_ratio_.sum():.3f}  "
          f"in {time.time() - t0:.1f}s\n")

    tree = build_tree(embs_pca, N_COARSE, N_MID_PER_COARSE, N_LEAF_PER_MID, SEED)

    Path(CLUSTER_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CLUSTER_OUT_PATH,
        prompts          = np.array(prompts, dtype=object),
        embeddings_raw   = embs_raw,
        embeddings_pca   = embs_pca,
        labels           = tree["prompt_leaf"],
        centroids_pca    = tree["centroids_pca"],
        pca_mean         = pca.mean_.astype(np.float32),
        pca_components   = pca.components_.astype(np.float32),
        tree_node_count  = tree["tree_node_count"],
        tree_parent      = tree["tree_parent"],
        tree_level       = tree["tree_level"],
        tree_centroid    = tree["tree_centroid"],
        tree_children    = tree["tree_children"],
        tree_leaf_prompts = tree["tree_leaf_prompts"],
    )
    print(f"\nSaved → {CLUSTER_OUT_PATH}")

    data = np.load(CLUSTER_OUT_PATH, allow_pickle=True)
    rng = np.random.default_rng(SEED)
    labels = data["labels"]
    print("\nSample prompts from 3 random leaf clusters:")
    for c in rng.choice(int(data["tree_level"].sum() > 0), size=3, replace=False):
        idxs = np.where(labels == c)[0]
        for idx in rng.choice(idxs, size=min(2, len(idxs)), replace=False):
            print(f"  leaf {c:03d}: {prompts[idx]}")


if __name__ == "__main__":
    main()