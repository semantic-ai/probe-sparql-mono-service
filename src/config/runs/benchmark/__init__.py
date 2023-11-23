from typing import NamedTuple

from .evaluation import EvaluationConfig
from .hybrid import HybridConfig
from .metrics import MetricsConfig



class BenchmarkConfig(NamedTuple):
    evaluation = EvaluationConfig()
    metrics = MetricsConfig()
    hybrid = HybridConfig()

    def to_dict(self):
        return dict(
            evaluation=self.evaluation.model_dump(),
            metrics=self.metrics.model_dump(),
            hybrid=self.hybrid.model_dump()
        )
