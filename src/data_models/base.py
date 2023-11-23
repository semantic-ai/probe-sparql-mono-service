from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from logging import Logger

import uuid
import os


class Base:
    """
    This class implements the basic functionality that is re-used in all data models.

    These are:
        1. set commonly used variables (config & logger)
        2. generate_uri
    """

    def __init__(self, config: DataModelConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

    @staticmethod
    def generate_uri(base_uri: str) -> str:
        """
        This function generates new randomized uri's based on a prefix and uuid4 combo.
        These uri's are required when writing new objects to the sparql endpoint.

        (During testing, base_uri is overwritten by TESTING environment value)

        :param base_uri: the base_uri to use for the freshly generated uri
        :return: The string formatted randomized uri
        """
        return f"<{base_uri}{os.getenv('TESTING', str(uuid.uuid4()))}>"

    def __repr__(self):
        """
        This function forces the objects to be representable with the __dict__ function cast to string

        :return: The string cast output of the self.__init__ function
        """
        return str(self.__dict__)

    def _ensure_encapsulation(self, str_input):

        corrected_str = str_input
        if not corrected_str.startswith("<"):
            corrected_str = "<" + corrected_str

        if not corrected_str.endswith(">"):
            corrected_str = corrected_str + ">"

        self.logger.debug(f"Corrected {str_input} -> {corrected_str}")
        return corrected_str
