from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy

    from logging import Logger

from ..base import Model
from ...data_models import Taxonomy

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel(Model):
    """
    Custom class implementation for model.

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

        self.embedding_matrix = None
        self._load_model(model_id)
        self._prep_labels(taxonomy)

    def _load_model(self, model_uri: str) -> None:
        self.model = SentenceTransformer(self.model_id)

    _load_model.__doc__ = Model._load_model.__doc__

    def _prep_labels(self, taxonomy: Taxonomy | list[str]) -> None:

        if not isinstance(taxonomy, list):
            self.labels = taxonomy.get_labels(max_depth=1)
            self.embedding_matrix = self._embed(self.labels)
        else:
            self.labels= taxonomy
            self.embedding_matrix = self._embed(taxonomy)

    _prep_labels.__doc__ = Model._prep_labels.__doc__

    def add_labels(self, labels: list[str]) -> None:
        self.embedding_matrix = self._embed(labels)

    add_labels.__doc__ = Model.add_labels.__doc__

    def _embed(self, text: str | list[str]) -> np.array:
        """
        This internal function is used to create embeddings.

        :param text: the text to embed
        :return:
        """
        if isinstance(text, list):
            return np.asarray([np.asarray(self.model.encode(t)) for t in text])
        else:
            return np.asarray(self.model.encode(text))

    def classify(self, text: str, multi_label, **kwargs) -> dict[str, float]:
        """
        [Adaptation] customized for embedding similarity predictions
        """
        text_embedding = np.asarray(self._embed(text))

        if labels := kwargs.get("labels", None):
            self._prep_labels(labels)

        similarity = cosine_similarity(np.asarray([text_embedding]), self.embedding_matrix)

        return {k: v for k, v in zip(self.labels, similarity.tolist()[0])}

    classify.__doc__ += Model.classify.__doc__
