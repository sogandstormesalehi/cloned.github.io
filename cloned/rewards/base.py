from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class Reward(ABC):
    @abstractmethod
    def score(self, images: List[Image.Image]) -> List[float]:
        ...