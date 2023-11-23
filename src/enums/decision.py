from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.data_models import DataModelConfig

import enum


class DecisionQuery(str, enum.Enum):
    """
    This enum is used to identify what type of query to run
    """
    ANNOTATED = "annotated"
    ALL = "all"

    @staticmethod
    def match(config: DataModelConfig, value: DecisionQuery):
        """
        this function allows us to verify the provided input and return the correct query

        :param config: the global configuration object
        :param value: the enum value that is used.
        :return: format-able query in string format
        """
        match value:
            case DecisionQuery.ANNOTATED:
                return config.decision.query_all_decisions_with_specified_taxonomy
            case DecisionQuery.ALL:
                return config.decision.query_all_decisions
