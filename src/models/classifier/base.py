from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy

    from logging import Logger

from ..base import Model

from setfit import SetFitModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ClassifierModel(Model):
    """
    Custom classifier base model

    This model implements the custom base for classifier models, it has minor adaptations compared to the original base model
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model_id: str,
            taxonomy: Taxonomy
    ) -> None:

        super().__init__(
            config=config,
            logger=logger,
            model_id=model_id
        )

        self._prep_labels(taxonomy)
        self._load_model(model_id)

    def _load_model(self, model_id: str) -> None:
        if "mlflow" in self.model_id:
            pass
        else:
            self.model = SetFitModel.from_pretrained(model_id)

    _load_model.__doc__ = Model._load_model.__doc__

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:
        pass

    classify.__doc__ = Model.classify.__doc__