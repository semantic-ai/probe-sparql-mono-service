from typing import NamedTuple

from .default import DefaultConfig
from .arguments import TrainingConfig


class TrainingConfig(NamedTuple):
    default = DefaultConfig()
    arguments = TrainingConfig()

    def to_dict(self):
        return dict(
            default=self.default.model_dump(),
            arguments=self.arguments.model_dump()
        )
