from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from logging import Logger

from ..base import Model

from .base import ZeroshotModel
from nltk.tokenize import sent_tokenize
import numpy as np
import torch


class SentenceZeroshotModel(ZeroshotModel):
    """
    Zeroshot model that has the sentence based approach (predict for each sentence separately)
    """

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:
        zeroshot_scores = []
        result = None
        labels = kwargs.get("labels", self.labels)

        self.logger.debug(f"Input text: {text}")
        self.logger.debug(f"predicting for: {labels}")

        for sentence in sent_tokenize(text, language="dutch"):

            self.logger.debug(f"predicting for: '{sentence}'")
            result = self.pipe(sentence, labels, multi_label=multi_label)
            zeroshot_scores.append(result.get("scores"))

        scores = np.asarray(zeroshot_scores).mean(axis=0)

        return {k: v for k, v in zip(result.get("labels"), scores.tolist())}

    classify.__doc__ = ZeroshotModel.classify.__doc__
