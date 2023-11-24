from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

import enum


class EndpointType(str, enum.Enum):
    """
    This enum is used to specify what type of query you want to execute
    """
    DECISION = "decision"
    TAXONOMY = "taxonomy"

    @staticmethod
    def match(config: Config, value: EndpointType):
        """
        this function allows us to verify the provided input and return the correct query

        :param config: the global configuration object
        :param value: the enum value that is used.
        :return: the chosen endpoint url
        """
        match value:
            case EndpointType.DECISION:
                return config.request.endpoint_decision
            case EndpointType.TAXONOMY:
                return config.request.endpoint_taxonomy


class AuthType(enum.Enum):
    NONE = "none"
    BASIC = "basic"
    DIGEST = "digest"
