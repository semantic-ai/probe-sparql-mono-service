from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy

    from logging import Logger

from .base import EmbeddingModel

import numpy as np
import pandas as pd
from ...data_models import Taxonomy
from sklearn.metrics.pairwise import cosine_similarity


class GroundUpRegularEmbeddingModel(EmbeddingModel):
    """
    This embedding model builds the tree from bottom to top based on confidences
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
            model_id=model_id,
            taxonomy=taxonomy
        )
        # todo further implement the label logic on baseclass etc
        self._prep_labels(taxonomy)

    def _prep_labels(self, taxonomy: Taxonomy | list[str]) -> None:

        if not isinstance(taxonomy, list):
            labels = taxonomy.get_labels(
                max_depth=10,
                include_tree_indication=True
            )

            self.indexes, self.labels = zip(*labels)
            self.embedding_matrix = self._embed(self.labels)

        else:
            self.labels = taxonomy
            self.embedding_matrix = self._embed(taxonomy)

    _prep_labels.__doc__ = EmbeddingModel._prep_labels.__doc__

    def _embed(self, text: str | list[str]) -> np.array:
        if isinstance(text, list):
            return np.asarray([np.asarray(self.model.encode(t)) for t in text])
        else:
            return np.asarray(self.model.encode(text))

    _embed.__doc__ = EmbeddingModel._embed.__doc__

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:
        text_embedding = np.asarray(self._embed(text))

        if labels := kwargs.get("labels", None):
            self._prep_labels(labels)

        similarity = cosine_similarity(np.asarray([text_embedding]), self.embedding_matrix)[0]

        self.logger.debug(f"Indexes: {self.indexes}")

        # create dataframe for easy builtin functions
        index_index_mapper = pd.DataFrame(
            list(
                zip(
                    self.indexes,
                    similarity
                )
            ),
            columns=["indexes", "scores"]
        )

        self.logger.debug(f"Index_mapper: {index_index_mapper}")

        # remap average score per subset and compute
        level_score_dict = {
            record.get("indexes"): record.get("scores")
            for record in index_index_mapper.groupby("indexes").agg(
                {
                    "indexes": "first",
                    "scores": "mean"
                }
            ).to_dict(orient="records")
        }
        self.logger.debug(f"level_Score_dict: {level_score_dict}")

        master_keys = [k for k in level_score_dict.keys() if "." not in k]
        self.logger.debug(f"Master keys: {master_keys}")

        average = dict()
        for master_key in master_keys:
            average[int(master_key)] = np.mean([
                v for k, v in level_score_dict.items()
                if k.startswith(master_key)
            ])

        return_scores = {k: v for k, v in zip(self.labels, list(average.values()))}
        self.logger.debug(f"Return scores: {return_scores}")

        return return_scores

    classify.__doc__ = EmbeddingModel.classify.__doc__
