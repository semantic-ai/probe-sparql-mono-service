from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, AutoModel

from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import EmbeddingModel


class ChunkedEmbeddingModel(EmbeddingModel):
    """
    Embedding implementation that chunks the text into slices of a certain length
    """

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:
        """
        [Adaptation] you can suply custom length with the kwargs (max_length)

        """
        similarity_scores = []

        # logic that checks tokenized length and optimizes
        text_chunks = []
        text_buffer = []
        cur_length = 0

        if labels := kwargs.get("labels", None):
            self._prep_labels(labels)

        for sentence in sent_tokenize(text, language="dutch"):
            tokenized_length = len(self.model.tokenize(sentence))
            self.logger.debug(f"Current_lenght {cur_length}, new_slice_length: {tokenized_length}")

            if (cur_length + tokenized_length) < kwargs.get("max_length", 512):
                cur_length += tokenized_length
                text_buffer.append(sentence)
            else:
                text_chunks.append(". ".join(text_buffer))

                text_buffer = []
                cur_length = 0
        else:
            text_chunks.append(". ".join(text_buffer))

        self.logger.debug(f"Chuncked data: {text_chunks}")

        for sentence in text_chunks:

            if len(sentence) < 5:
                # skipping sentences that are extremely short
                continue

            sentence_embedding = np.asarray(self._embed(sentence))

            similarity_scores.append(
                cosine_similarity(
                    np.asarray([sentence_embedding]),
                    self.embedding_matrix
                ).tolist()[0]
            )

        similarity: np.array = np.asarray(similarity_scores).mean(axis=0)
        return {k: v for k, v in zip(self.labels, similarity.tolist())}

    classify.__doc__ = EmbeddingModel.classify.__doc__
