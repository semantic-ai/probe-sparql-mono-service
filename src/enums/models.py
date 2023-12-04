from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

import enum


class ModelType(str, enum.Enum):
    """
    This enum is used to identify what model type/flavour you are running.
    """
    # zeroshot
    ZEROSHOT_REGULAR: str = "zeroshot_regular"
    ZEROSHOT_SENTENCE: str = "zeroshot_sentence"
    ZEROSHOT_CHUNKED: str = "zeroshot_chunked"
    ZEROSHOT_CHILD_LABELS: str = "_zeroshot_child_labels"

    # embeddings
    EMBEDDING_CHILD_LABELS: str = "_embedding_child_labels"
    EMBEDDING_REGULAR: str = "embedding_regular"
    EMBEDDING_SENTENCE: str = "embedding_sentence"
    EMBEDDING_CHUNKED: str = "embedding_chunked"
    EMBEDDING_GROUND_UP: str = "_embedding_ground_up"
    EMBEDDING_GROUND_UP_GREEDY: str = "_embedding_ground_up_greedy"

    # setfit

    # classifier
    HUGGINGFACE_MODEL: str = "huggingface_model"

    # other?
    HYBRID_BASE_MODEL: str = "hybrid_base_model"
    HYBRID_SELECTIVE_MODEL: str = "hybrid_selective_model"

    # topic modeling
    REGULAR_TOPIC_MODEL: str = "topic_model_regular"
    HIERARCHIC_TOPIC_MODEL: str = "topic_model_hierarchic"
    DYNAMIC_TOPIC_MODEL: str = "topic_model_dynamic"

    @classmethod
    def _list(cls):
        """
        internal classmethod that allows us to retrieve all possible datasets
        :return:
        """
        return list(map(lambda c: c.value, cls))

    @staticmethod
    def get_models_for_type(model_type: str):
        """
        this function allows us to retrieve only the models compliant with the prefix filter

        :param model_type: the string prefix to filter the models with
        :return: a list with models that comply with the filter
        """
        return [v for v in ModelType._list() if v.split("_")[0] == model_type]

    @staticmethod
    def get_from_prefix(model_type: str):
        """
        this function allows us to retrieve only the models compliant with the prefix filter

        :param model_type: the string prefix to filter the models with
        :return: a list with models that comply with the filter
        """
        return [v for v in ModelType._list() if v.startswith(model_type)]
