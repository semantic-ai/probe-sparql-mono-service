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


class ChunkedZeroshotModel(ZeroshotModel):
    """
    Chunked zeroshot implementation, based on the regular approach but text is chunked based on its maximum length
    """

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:
        """
        [Adaptation] Text length can be predefined with kwargs (using max_length)
        """

        zeroshot_scores = []
        result = None
        labels = kwargs.get("labels", self.labels)

        self.logger.debug(f"Input text: {text}")
        self.logger.debug(f"predicting for: {labels}")

        # logic that checks tokenized length
        text_chunks = []
        text_buffer = []
        cur_length = 0

        for sentence in sent_tokenize(text):
            tokenized_length = len(self.tokenizer.tokenize(sentence))
            self.logger.debug(f"Current_length {cur_length}, new_slice_length: {tokenized_length}")

            if (cur_length + tokenized_length) < kwargs.get("max_length", 512): # hardcoded max length for now
                cur_length += tokenized_length
                text_buffer.append(sentence)
            else:
                text_chunks.append(". ".join(text_buffer))

                text_buffer = []
                cur_length = 0
        else:
            text_chunks.append(". ".join(text_buffer))

        self.logger.info(f"Chunked data: {text_chunks}")

        for sentence in [c for c in text_chunks if len(c.replace(" ", "")) >= 2]:

            self.logger.debug(f"predicting for sentence: '{sentence}'")

            result = self.pipe(sentence, labels, multi_label=multi_label)
            zeroshot_scores.append(result.get("scores"))

        scores = np.asarray(zeroshot_scores).mean(axis=0)

        return {k: v for k, v in zip(result.get("labels"), scores.tolist())}

    classify.__doc__ += ZeroshotModel.classify.__doc__
