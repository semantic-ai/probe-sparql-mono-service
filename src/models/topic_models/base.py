from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...dataset import DatasetBuilder
    from logging import Logger

import abc


class TopicModel:
    """
    ABC class for topic modeling

    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            dataset_builder: DatasetBuilder
    ) -> None:
        self.config = config
        self.logger = logger
        self.dataset_builder = dataset_builder

    @abc.abstractmethod
    def _create_dataset(self):
        """
        This function creates the dataset from the provided dataset object, the dataset is transformed into the required
        shape for bertopic modeling

        :return:
        """
        pass

    @abc.abstractmethod
    def analyse(self):
        """
        This function is called to start the topic modeling algorithm

        :return: Nothing, everything is logged to mlflow artifacts
        """
        pass
