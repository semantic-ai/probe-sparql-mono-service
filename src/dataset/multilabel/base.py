from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

from transformers import AutoTokenizer
import torch
from torch import device
import abc

from ..base import TrainDataset


class MultilabelTrainingDataset(TrainDataset):
    """
    This class is the basic implementation for the multilabel dataset concept, as the name suggest these are datasets that support
    functionality to process multilabel data.

    It has extended functionality compared to the single label dataset in order to manage the multible labels.

    This class is based from 'TrainDataset', which on its own is derived from the torch.utils.Dataset.

    Typical usage:
        >>> from src.dataset.builder import DatasetBuilder
        >>> dataset_builder = DatasetBuilder(...)
        >>> ds = MultilabelTrainingDataset(
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
            dataset: list[dict[str, str]],
            tokenizer: AutoTokenizer = None,
            _device: device = device("cpu"),
            sub_node: str = None
    ):

        self._labels: list[str] | None = None
        self._max_label_depth = 1

        self._taxonomy = None

        self.config = config
        self.logger = logger
        self.taxonomy = taxonomy
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = _device
        self.max_length = 512
        self.max_depth = 1
        self.sub_node = sub_node

    def _get_label(self, idx: int) -> list[int]:
        """
        This function implements the abstract logic for building the labels, it can be overwritten and adapted without
        problems (as long as the default input output signature is kept).

        :param idx: the integer value of the input index
        :return: a list of integer values where the labels should be predicted
        """

        labels = []

        self.logger.debug(f"len {len(self.dataset)}")
        selected_record = self.dataset[idx]
        self.logger.debug(f"selected_record: {selected_record}")

        input_labels = selected_record.get("labels", None)

        if input_labels is None:
            input_labels = []

        self.logger.debug(f"input labels: {input_labels}")

        for label in input_labels:
            label_in_tree = self.taxonomy.find(label)

            self.logger.debug(f"Label ({label}) found in tree {label_in_tree}")
            labels.append(label_in_tree.get(1).get("label"))

        if self.config.run.dataset.apply_mlb:
            self.logger.debug("mlb")
            _labels = self.binarized_label_dictionary
            for label in labels:
                _labels[label] = 1

            self.logger.debug(f"Current mlb labels: {_labels}")
            return list(_labels.values())

        assert not labels == [], Exception("Labels can't be empty when pulling training records")
        return labels

    def _get_text(self, idx: int) -> str:
        """
        This function implements the abstract logic in order to retrieve the text from the provided dataset.
        It is possible to overwrite this function and define custom behaviour to create the text input for the model.

        :param idx: the integer value of the input index
        :return: a list of integer values where the labels should be predicted
        """
        data_record = self.dataset[idx]
        motivation = data_record.get("motivation", "")

        return motivation

    def __getitem__(self, idx) -> dict[str, str]:
        """
        This function provides the functionality to retrieve a training/inference record based on the provided idx.
        The function also provides extra functionality like tokenization and moving tensors from one device to another.

        Example usage:
            >>> dataset = MultilabelTrainingDataset(...)
            >>> record = dataset[10]
            >>> print(record)

        :param idx: the integer value of the input index
        :return: a dictionary containing the train/inference samples (adaptable with a config)
        """

        labels = self._get_label(idx) if self.config.run.dataset.get_label else []
        text = self._get_text(idx)
        decision_uri = self.dataset[idx].get("uri")

        if self.config.run.dataset.tokenize:
            self.logger.debug("tokenize")
            tokenized_text = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            ).to(self.device)

            return dict(
                input_ids=tokenized_text.input_ids[0],
                attention_mask=tokenized_text.attention_mask[0],
                labels=torch.tensor(labels, dtype=torch.float)
            )

        else:
            return dict(
                text=text,
                decision_uri=decision_uri,
                labels=labels
            )

    @property
    def candid_labels(self) -> list[str]:
        """
        This property provides the functionality to set and retrieve the candid_labels.
        The candid_labels are the labels that are used for model prediction.

        :return: list of strings representing the current labels
        """
        self._labels = self.taxonomy.get_labels(max_depth=self.max_depth)

        self.logger.debug(f"Current candid labels: {self._labels}")

        return self._labels

    @candid_labels.setter
    def candid_labels(self, value: list[str]) -> None:
        if isinstance(value, str):
            value = [value]

        self._labels = value

    @property
    def binarized_label_dictionary(self) -> dict[str, int]:
        """
        This property allows users to retrieve the blank dictionary that can be used for well formatted label binarization

        :return: a dictionary containing the binarized candid labels
        """

        binarized_candid_labels = {l: 0 for l in self.candid_labels}
        self.logger.debug(f"Current binirized labels: {binarized_candid_labels}")
        return binarized_candid_labels

    @property
    def max_label_depth(self) -> int:
        """
        This property can be used to set or retrieve the max depth (for labels handling logic)

        :return: the integer value for the max depth
        """
        return self._max_label_depth

    @max_label_depth.setter
    def max_label_depth(self, value: int) -> None:
        if not isinstance(value, int):
            raise Exception("Provided max depth value is not of type integer.")
        if value > 0:
            raise Exception("Cannot set max depth to zero.")
        self._max_label_depth = value

    @property
    def taxonomy(self) -> Taxonomy:
        """
        This property can be used to set or retrieve the taxonomy that is/will be used for processing and finding of labels.

        :return: an instance of Taxonomy
        """
        return self._taxonomy

    @taxonomy.setter
    def taxonomy(self, value: Taxonomy) -> None:
        self._taxonomy = value

    def __len__(self):
        """
        This method responds with the current length of the dataset

        :return: integer value representing the length of the internal dataset.
        """
        return len(self.dataset)

    def get_specific_record(self, idx: int, label_level: int) -> dict[str, str | list]:
        pass
