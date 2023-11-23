from __future__ import annotations

from abc import ABC
# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

from ..base import Model

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


class ZeroshotModel(Model):
    """
    Base model for zeroshot approaches
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model_id: str,
            taxonomy: Taxonomy
    ):
        super().__init__(
            config=config,
            logger=logger,
            model_id=model_id
        )

        self._load_model(model_id)
        self._prep_labels(taxonomy)

    def _load_model(self, model_id: str) -> None:
        """
        [Adaptation] Logic for loading zeroshot models
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.config.run.model.pull_token)

        # force tokenizer to max length some models don't have this for some reason??
        # should we try to do this only for the models that don't possess the max length information?
        self.tokenizer.model_max_length = 512

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            token=self.config.run.model.pull_token
        ).to(self.device)

        self.pipe = pipeline(
            task="zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer, # https://github.com/huggingface/transformers/issues/4501
            device=self.device,
        )

    _load_model.__doc__ += Model._load_model.__doc__

    @torch.inference_mode()
    def nli_infer(self, premise: str, hypothesis: str):
        """
        Low level implementation on how zeroshot models can be used aswel

        :param premise: input text
        :param hypothesis: parsed label text
        :return: score for prediction to be one of 3 hardcoded laabels
        """
        input = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        output = self.model(input["input_ids"])

        prediction = torch.softmax(output["logits"][0], -1).tolist()
        prediction = {name: round(float(pred) * 100, 1) for pred, name in
                      zip(prediction, ["entailment", "neutral", "contradiction"])}
        return prediction

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:
        labels = kwargs.get("labels", self.labels)

        result = self.pipe(
            text,
            labels,
            multi_label=multi_label
        )

        return {k: v for k, v in zip(result.get("labels"), result.get("scores"))}

    classify.__doc__ = Model.classify.__doc__
