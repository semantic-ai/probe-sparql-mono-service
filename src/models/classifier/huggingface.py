from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

from ..base import Model

import mlflow
from transformers import pipeline, AutoModelForSequenceClassification


class HuggingfaceModel(Model):

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model_id: str,
            taxonomy: Taxonomy,
            stage: str = "Production"
    ) -> None:

        super().__init__(
            config=config,
            logger=logger,
            model_id=model_id
        )
        self._prep_labels(taxonomy)
        self._load_model(
            model_id=model_id,
            stage=stage
        )

    def _load_model(
            self,
            model_id: str,
            stage: str
    ) -> None:

        self.logger.debug(f"model id {model_id}")

        if "mlflow" in self.model_id:
            self.logger.debug("SELECTING MLFLOW MODEL")
            components = mlflow.transformers.load_model(
                f"models:/{model_id.split(':/')[-1]}/{stage}",
                return_type="components"
            )

            tokenizer = components.get("tokenizer")
            tokenizer.model_max_len = 512  # hardcode max length of input to prevent further issues

            self.model = pipeline(
                "text-classification",
                model=components.get("model"),
                tokenizer=tokenizer
            )

        else:
            self.logger.debug("SELECTING HUGGINGFACE MODEL")
            self.model = pipeline(
                model_id=model_id,
                task="text-classification",
            )

    def classify(self, text: str, multi_label: bool = True, **kwargs) -> dict[str, float]:

        self.logger.debug(f"pipeline: {type(self.model)} {self.model}")
        pipeline_output = self.model(
            text,
            return_all_scores=True,
            # padding=True,
            truncation=True
        )[0]

        reformatted_output = dict()
        for record in pipeline_output:
            reformatted_output[record.get("label")] = record.get("score")

        return reformatted_output

    classify.__doc__ = Model.classify.__doc__
