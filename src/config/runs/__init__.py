from __future__ import annotations
from typing import NamedTuple

from .benchmark import BenchmarkConfig
from .dataset import DatasetConfig
from .training import TrainingConfig
from .model import ModelConfig


class RunConfig(NamedTuple):
    benchmark = BenchmarkConfig()
    dataset = DatasetConfig()
    training = TrainingConfig()
    model = ModelConfig()

    def to_dict(self):
        return dict(
            benchmark=self.benchmark.to_dict(),
            dataset=self.dataset.model_dump(),
            training=self.training.to_dict(),
            model=self.model.model_dump()

        )
