from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

from torch import Tensor
from transformers import AutoTokenizer

import abc
from typing import Any
from ..base import TrainDataset


class TrainingDataset(TrainDataset):
    """
    This class is the basic implementation for the single label dataset concept.
    This class is based from 'TrainDataset', which on its own is derived from the torch.utils.Dataset.

    Typical usage:
        >>> from src.dataset.builder import DatasetBuilder
        >>> dataset_builder = DatasetBuilder(...)
        >>> ds = TrainingDataset(
                config=Config(),
                logger=logging.logger,
                taxonomy=Taxonomy(...),
                dataset=dataset_builder.train_dataset
            )
        >>> # getting the first custom formatted data example
        >>> print(ds[0])
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            taxonomy: Taxonomy,
            data: Any,
            tokenizer: AutoTokenizer = None,
            device: str = "cpu"
    ):
        self.logger = logger
        self.config = config
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.dataset = data
        self.device = device

    @abc.abstractmethod
    def __get_label(self, idx: int) -> Tensor | str:
        pass

    @abc.abstractmethod
    def __get_text(self, idx: int) -> dict[str, Tensor] | str:
        pass

    def __getitem__(self, idx) -> dict[str, Tensor | str]:
        """
        This function provides the functionality to retrieve a training/inference record based on the provided idx.
        The function also provides extra functionality like tokenization and moving tensors from one device to another.

        Example usage:
            >>> dataset = TrainingDataset(...)
            >>> record = dataset[10]
            >>> print(record)

        :param idx: the integer value of the input index
        :return: a dictionary containing the train/inference samples (adaptable with a config)
        """
        label = self.__get_label(idx)
        text = self.__get_text(idx)

        tokenized_text = self.tokenizer.tokenize(text).to(self.device)

        return dict(
            input_ids=tokenized_text["input_ids"].to(self.device),
            attention_mask=tokenized_text["attention_mask"].to(self.device),
            label=label.to(self.device)
        )

    def __len__(self) -> int:
        """
        This method responds with the current length of the dataset

        :return: integer value representing the length of the internal dataset.
        """
        return len(self.dataset)

    def iter_records(self):
        """
        Iterator function that iterates over all the items in the dataset

        :return:
        """
        for i in range(len(self)):
            yield self[i]