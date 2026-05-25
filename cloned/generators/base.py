from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any
from PIL import Image

class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], seed: int | None = None) -> List[Image.Image]:
        ...

class VideoGenerator(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], seed: int | None = None) -> List[str]:
        ...