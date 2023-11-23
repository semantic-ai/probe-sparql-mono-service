from __future__ import annotations

from .constant import CONFIG_PREFIX
from ..base import Settings
from ...enums import ModelType, TrainingFlavours


class ModelConfig(Settings):
    type: ModelType = ModelType.EMBEDDING_REGULAR.value
    flavour: TrainingFlavours = TrainingFlavours.DISTIL_BERT.value
    pull_token: str = ""

    class Config():
        env_prefix = f"{CONFIG_PREFIX}model_"
