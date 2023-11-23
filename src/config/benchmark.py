from __future__ import annotations
from .base import Settings


class BenchmarkConfig(Settings):
    evaluation_type: str = "multi"
    probe_model_class_type: str = ""
    dataset_type: str = ""
    zero_devision_default: float = 0.0

    class Config():
        env_prefix = "benchmark_"
