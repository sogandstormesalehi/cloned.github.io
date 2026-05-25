from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image

from cloned.rewards.base import Reward

@dataclass(frozen=True)
class RewardCallSpec:
    accepts_target: bool
    requires_target: bool

def _reward_call_spec(rewarder) -> RewardCallSpec:
    if hasattr(rewarder, "accepts_target") or hasattr(rewarder, "requires_target"):
        accepts = bool(getattr(rewarder, "accepts_target", False))
        requires = bool(getattr(rewarder, "requires_target", False))
        return RewardCallSpec(accepts_target=accepts, requires_target=requires)

    fn = getattr(rewarder, "reward", None)
    if fn is None or not callable(fn):
        raise TypeError(f"Reward model {rewarder!r} has no callable .reward(...)")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]

    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) <= 1:
        return RewardCallSpec(accepts_target=False, requires_target=False)

    target_param = positional[1]
    requires_target = target_param.default is inspect._empty
    return RewardCallSpec(accepts_target=True, requires_target=requires_target)

def call_reward(rewarder, image: Image.Image, target_image: Optional[Image.Image]) -> float:
    spec = _reward_call_spec(rewarder)
    if spec.requires_target and target_image is None:
        raise ValueError(f"{rewarder.__class__.__name__} requires a target_image but target_image=None.")
    if target_image is not None and spec.accepts_target:
        return float(rewarder.reward(image, target_image))
    return float(rewarder.reward(image))

class BlackboxRewardAdapter(Reward):
    def __init__(self, rewarder, target_image: Optional[Image.Image] = None):
        self.rewarder = rewarder
        self.target_image = target_image

        spec = _reward_call_spec(rewarder)
        if spec.requires_target and self.target_image is None:
            raise ValueError(f"{rewarder.__class__.__name__} requires target_image, but none provided.")

    def score(self, images: List[Image.Image]) -> List[float]:
        return [call_reward(self.rewarder, im, self.target_image) for im in images]