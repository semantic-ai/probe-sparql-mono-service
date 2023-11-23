from __future__ import annotations
from .constant import CONFIG_PREFIX
from ...base import Settings


class MetricsConfig(Settings):

    # calculate metrics
    f1: bool = True
    precision: bool = True
    recall: bool = True
    hamming: bool = False

    # extra metric configuration
    zero_division_default: float = 0.0

    class Config():
        env_prefix = f"{CONFIG_PREFIX}metrics_"
