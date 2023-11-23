from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

import enum


class TrainingFlavours(str, enum.Enum):
    """
    This enum is used to identify what type of training flavour you want to use.
    """
    BERT: str = "bert"
    DISTIL_BERT: str = "distil_bert"
    SETFIT: str = "setfit"

    @staticmethod
    def get_default_model_for_type(config: Config, model_flavour: TrainingFlavours) -> str:
        """
        this function checks what the chosen flavour is and returns the defaulted value from the config for the model_id

        :param config: the global configuration object
        :param model_flavour: the enum value that is used.
        :return: format-able query in string format
        """
        match model_flavour:

            case TrainingFlavours.BERT | TrainingFlavours.BERT.value:
                return config.run.training.default.bert_model_id

            case TrainingFlavours.DISTIL_BERT | TrainingFlavours.DISTIL_BERT.value:
                return config.run.training.default.distil_bert_model_id

            case TrainingFlavours.SETFIT | TrainingFlavours.SETFIT.value:
                return config.run.training.default.setfit_model_id
