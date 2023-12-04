from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from logging import Logger
    from typing import Any

from ...data_models import Taxonomy
from transformers import AutoTokenizer
import torch
from torch import device
from ...enums import TaxonomyFindTypes
import abc

from ..base import TrainDataset


class DynamicMultilabelTrainingDataset(TrainDataset):
    """
    This class is the dynamic implementation for the multilabel dataset concept, as the name suggest these are datasets that support
    functionality to process multilabel data.

    Based on the provided taxonomy sub node it will create its own label set etc.


    It has extended functionality compared to the single label dataset in order to manage the multible labels.

    This class is based from 'TrainDataset', which on its own is derived from the torch.utils.Dataset.

    Typical usage:
        >>> from src.dataset.builder import DatasetBuilder
        >>> dataset_builder = DatasetBuilder(...)
        >>> ds = DynamicMultilabelTrainingDataset(
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

        self._sub_select_dataset()

    def _sub_select_dataset(self):
        """
        This function is responsible for the remapping of the dataset from the general input data
        to the input required to train for the provided sub node.

        Besides the general concept of creating the subselected dataset, there is also an option to train with the negative
        examples. These are the decisions that do no match to the taxonomy node, however they can provide more context to
        the model.
        Below you can find some more information about the usage and impact:
            Providing the model with examples that do not match any of the classes you are training for can potentially help improve its performance in multilabel document classification. This is because it can help the model learn to distinguish between the positive and negative examples more effectively.

            When training a multilabel classifier, it is important to provide the model with a balanced dataset that includes both positive and negative examples for each class. However, it can also be helpful to provide the model with examples that do not belong to any of the classes. This can help the model learn to distinguish between the positive and negative examples more effectively.

            Here are some specific benefits of providing the model with examples that do not match any of the classes:

            It can help prevent the model from overfitting to the training data. Overfitting occurs when the model learns the training data too well and is unable to generalize to new data. Providing the model with examples that do not belong to any of the classes can help prevent this by forcing the model to learn more general features.
            It can help the model learn to identify patterns in the data that are not associated with any of the classes. These patterns can then be used to identify new classes that the model has not been trained on.
            It can help the model learn to ignore irrelevant information. This can improve the model's performance on tasks that require it to focus on the most important information in the data.
            Of course, there are also some potential drawbacks to providing the model with examples that do not match any of the classes. For example, it can increase the training time and make it more difficult for the model to converge on a good solution. Additionally, if the model is not carefully tuned, it can learn to associate these examples with one of the classes, which can lead to errors.

            Overall, whether or not to provide the model with examples that do not match any of the classes is a trade-off that depends on the specific problem you are trying to solve. However, in general, it can be a helpful technique for improving the performance of multilabel document classifiers.

        """
        self.logger.info(f"current dataset length: {len(self.dataset)}")

        # if no sub_node is specified, there shouldn't be any adaptations to the dataset
        if self.sub_node is None:
            self.logger.debug("no triggered for sub node")

            # setting some required variables
            self.candid_labels = self.taxonomy.get_labels(max_depth=1)
            self.node_depth = 0
            self.label_distribution = self.binarized_label_dictionary
            self.sub_node_taxo = self.taxonomy
            return

        self.logger.debug("Triggerd subselection of dataset")
        self.logger.debug(f"searching for {self.sub_node}")

        # selecting the specific node from the taxonomy
        sub_node_information: dict = self.taxonomy.find(
            search_term=self.sub_node.lower(),
            search_kind=TaxonomyFindTypes.URI if self.sub_node.startswith("http") else TaxonomyFindTypes.LABEL,
            with_children=True,
        )
        self.node_depth: int = len(sub_node_information)  # getting taxonomy depth

        self.logger.debug(f"Sub node information depth: {self.node_depth} full:{sub_node_information}")

        self.sub_node_taxo = Taxonomy.from_dict(
            config=self.config.data_models,
            logger=self.logger,
            dictionary=sub_node_information.get(self.node_depth)
        )
        self.logger.info(f"rebuild taxonomy from sub_node: {self.sub_node}")
        self.candid_labels = [node.label for node in self.sub_node_taxo.children]
        self.label_distribution = self.binarized_label_dictionary

        new_dataset = []
        for row in self.dataset:
            row: dict[str, Any]

            valid_labels: list[str] = []
            labels_in_row: list[str] = row.get("labels", [])
            self.logger.debug(f"Labels in row: {labels_in_row}")

            for label in labels_in_row:
                label_information = self.taxonomy.find(label)

                self.logger.debug(f"requested sub_node: {self.sub_node_taxo.todict()}")
                self.logger.debug(f"label information: {label_information}")

                # check if the label length is at-least deeper than the master node (otherwise irrelevant for training)
                if len(label_information) > len(sub_node_information):

                    # retrieve the correct depth to compare if it at-least matches the parent label
                    specific_depth_label_uri: str = label_information.get(self.node_depth).get("uri")
                    specific_depth_node_uri: str = sub_node_information.get(self.node_depth).get("uri")

                    self.logger.info(
                        f"sub label in candid_labels: {(label_information.get(self.node_depth + 1).get('label') in self.candid_labels)}")
                    # check whether parent labels match and child labels are in candid_labels
                    if (specific_depth_node_uri == specific_depth_label_uri) \
                            & (label_information.get(self.node_depth + 1).get("label") in self.candid_labels):
                        valid_labels.append(label)

            row["labels"] = valid_labels

            # check if negative labels keep is true or if length is more than 1
            if self.config.run.training.default.keep_negative_examples or (len(valid_labels) > 0):
                new_dataset.append(row)

        self.logger.info(f"New dataset length {len(new_dataset)}")
        self.dataset = new_dataset

    def _get_label(self, idx: int) -> list[int]:
        """
        This function implements the abstract logic for building the labels, it can be overwritten and adapted without
        problems (as long as the default input output signature is kept).

        :param idx: the integer value of the input index
        :return: a list of integer values where the labels should be predicted
        """

        labels = []

        selected_record = self.dataset[idx]
        for label in selected_record.get("labels", []):
            label_in_tree = self.taxonomy.find(label)

            self.logger.debug(f"Label ({label}) found in tree {label_in_tree}")
            labels.append(label_in_tree.get(self.node_depth + 1).get("label"))

        if self.config.run.dataset.apply_mlb:
            self.logger.debug("mlb")
            _labels = self.binarized_label_dictionary
            for label in list(set(labels)):
                _labels[label] = 1
                self.label_distribution[label] += 1

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
        self.logger.debug(f"label: len({len(labels)}) {labels}")
        text = self._get_text(idx)
        decision_uri = self.dataset[idx].get("uri")

        # workaround for missing date, use short_title year reference if no date suplied
        publication_date = self.dataset[idx].get("", None)

        if publication_date is not None:
            date = publication_date
        else:
            try:
                date = int(self.dataset[idx].get("short_title").split("_")[0])
            except:
                date = -1

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
                text=" ".join(text.split()),
                decision_uri=decision_uri,
                labels=labels,
                date=date
            )

    @property
    def candid_labels(self) -> list[str]:
        """
        This property provides the functionality to set and retrieve the candid_labels.
        The candid_labels are the labels that are used for model prediction.

        :return: list of strings representing the current labels
        """
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
