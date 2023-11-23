from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.data_models import DataModelConfig

import enum


class GraphType(str, enum.Enum):
    """
    This enum contains the predefined graph suffixes that can be used to save to a certain sparql graph
    """

    USER_ANNOTATION = "user_annotation"
    MODEL_ANNOTATION = "model_annotation"
    TESTING = "testing_annotation"

    @staticmethod
    def match(config: DataModelConfig, value: GraphType):
        """
        this function returns the config value for the chosen enum type
        :param config:
        :param value:
        :return:
        """
        match value:
            case GraphType.USER_ANNOTATION:
                return config.sparql.user_annotations_graph
            case GraphType.MODEL_ANNOTATION:
                return config.sparql.probe_model_annotations_graph
            case GraphType.TESTING:
                return config.sparql.testing_graph
