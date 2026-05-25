from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(self, device: str, clip_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.model  = CLIPModel.from_pretrained(clip_name, use_safetensors=True).to(device).eval()
        self.proc   = CLIPProcessor.from_pretrained(clip_name, use_safetensors=True)


    @torch.no_grad()
    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.proc(
            text=prompts, return_tensors="pt", padding=True, truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb    = self.model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        return torch.nn.functional.normalize(emb.float(), p=2, dim=-1)

    def text_numpy(self, prompts: List[str], batch_size: int = 256) -> np.ndarray:
        all_embs = []
        for i in range(0, len(prompts), batch_size):
            e = self.text_embeddings(prompts[i:i+batch_size]).cpu().numpy().astype(np.float32)
            all_embs.append(e)
        return np.vstack(all_embs)

    def batch_numpy(self, prompts: List[str], batch_size: int = 256) -> np.ndarray:
        return self.text_numpy(prompts, batch_size)


    @torch.no_grad()
    def image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.proc(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb    = self.model.get_image_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        return torch.nn.functional.normalize(emb.float(), p=2, dim=-1)

    def image_numpy(self, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        for i in range(0, len(images), batch_size):
            e = self.image_embeddings(images[i:i+batch_size]).cpu().numpy().astype(np.float32)
            all_embs.append(e)
        return np.vstack(all_embs)


CLIPTextEmbedder = CLIPEmbedder