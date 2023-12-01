from __future__ import annotations

from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import EmbeddingModel


class SentenceEmbeddingModel(EmbeddingModel):

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:

        similarity_scores = []

        if labels := kwargs.get("labels", None):
            self._prep_labels(labels)

        for sentence in sent_tokenize(text, language="dutch"):

            # if len(sentence) < 5:
            #     # skipping sentences that are extremely short
            #     continue

            sentence_embedding = np.asarray(self._embed(sentence))
            similarity_scores.append(
                cosine_similarity(
                    np.asarray([sentence_embedding]),
                    self.embedding_matrix
                ).tolist()[0]
            )

        similarity = np.asarray(similarity_scores).mean(axis=0)
        return {k: v for k, v in zip(self.labels, similarity.tolist())}

    classify.__doc__ = EmbeddingModel.classify.__doc__
