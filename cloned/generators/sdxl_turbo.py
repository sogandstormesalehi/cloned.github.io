from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

from .base import ImageGenerator


class SDXLTurboGenerator(ImageGenerator):
    def __init__(
        self,
        device: str,
        batch_size: int = 32,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        height: int = 512,
        width: int = 512,
    ):
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        self.device              = device
        self.batch_size          = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale      = guidance_scale
        self.height              = height
        self.width               = width

    def generate(
        self,
        prompts: List[str],
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
    
        images: List[Image.Image] = []
        for chunk_idx, start in enumerate(range(0, len(prompts), self.batch_size)):
            chunk      = prompts[start : start + self.batch_size]
            chunk_seed = (seed + chunk_idx) if seed is not None else None
            self._set_seed(chunk_seed)
            with torch.no_grad():
                out = self.pipe(
                    chunk,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=self.height,
                    width=self.width,
                ).images
            images.extend(out)
        return images

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)