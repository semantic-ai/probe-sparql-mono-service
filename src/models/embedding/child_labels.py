from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy

    from logging import Logger

from .base import EmbeddingModel

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChildLabelsEmbeddingModel(EmbeddingModel):
    """
    Child label class implementation for embedding model
    """

    def _text_formatting(self, taxonomy_node: Taxonomy) -> str:
        parent_label = taxonomy_node.label
        sub_labels = [label for label in taxonomy_node.all_linked_labels if label != parent_label]

        custom_text = f"Tekst over {parent_label} of meer specifiek {' of'.join(sub_labels)}"
        self.logger.debug(f"Created custom text: {custom_text}")

        return custom_text

    def _prep_labels(self, taxonomy: Taxonomy | list[str]) -> None:

        if not isinstance(taxonomy, list):
            self.labels = taxonomy.get_labels(max_depth=1)

            label_string = [
                self._text_formatting(taxonomy_node=parent_node)
                for parent_node in taxonomy.children
            ]

            self.embedding_matrix = self._embed(label_string)

        else:
            self.labels = taxonomy
            self.embedding_matrix = self._embed(taxonomy)

