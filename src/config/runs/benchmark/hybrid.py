from __future__ import annotations
from .constant import CONFIG_PREFIX
from ...base import Settings


class HybridConfig(Settings):

    # calculate metrics
    minimum_threshold: float = 0.0

    class Config():
        env_prefix = f"{CONFIG_PREFIX}hybrid_"
