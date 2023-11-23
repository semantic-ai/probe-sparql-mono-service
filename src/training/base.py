from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..dataset import DatasetBuilder

from ..dataset import create_dataset
from datasets import Dataset
import abc


class Training:
    """
    Base class for training scripts
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            base_model_id: str,
            dataset_builder: DatasetBuilder
    ) -> None:
        self.config = config
        self.logger = logger
        self.base_model_id = base_model_id
        self.dataset_builder = dataset_builder

    @abc.abstractmethod
    def compute_metrics(self, pred):
        """
        This function computes the custom metrics during the training process

        :param pred: input values from training
        :return:
        """
        pass

    @abc.abstractmethod
    def _create_model(self):
        """
        Function that instantiates the model for training useage
        :return:
        """
        pass

    @abc.abstractmethod
    def _create_dataset(self):
        """
        This internal function is used to create datasets from the provided dataset builder
        :return:
        """
        train_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy
        )

        self.train_ds = Dataset.from_list(
            train_dataset
        )

        eval_dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.test_dataset,
            taxonomy=self.dataset_builder.taxonomy
        )

        self.eval_ds = Dataset.from_list(
            eval_dataset
        )

    @abc.abstractmethod
    def train(self):
        """
        This function executes the training code.

        :return: Nothing
        """
        pass

    @abc.abstractmethod
    def __call__(self):
        """
        Function wrapper for train func
        :return:
        """
        return self.train()
