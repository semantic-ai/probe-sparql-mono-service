from __future__ import annotations

from abc import ABC
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

from .base import MultilabelTrainingDataset


class SummaryStatisticDataset(MultilabelTrainingDataset, ABC):
    """
    [Adapated from baseclase] This implementation uses lvl2 labels

    PARENT DOCS:
    ---
    """
    __doc__ += MultilabelTrainingDataset.__doc__

    def _get_label(self, idx: int, label_level: int) -> list[int]:
        """
        [Adapted implementation] overwritten from baseclase, label responds with string value instead
        """
        __doc__ = MultilabelTrainingDataset._get_text.__doc__

        labels = []

        selected_record = self.dataset[idx]

        record_labels = selected_record.get("labels")
        if record_labels is None:
            record_labels = []

        for label in record_labels:
            label_in_tree = self.taxonomy.find(label)

            self.logger.debug(f"Label ({label}) found in tree {label_in_tree}")

            if selected_label := label_in_tree.get(label_level, None):
                labels.append(selected_label.get("label"))

        return list(set(labels))

    _get_label.__doc__ += MultilabelTrainingDataset._get_label.__doc__

    def get_specific_record(self, idx: int, label_level: int) -> dict[str, str | list]:
        """
        This function implements the functionality to retrieve what label is available at what level.

        :param idx: the index to take as integer value
        :param label_level: the label level as integer value
        :return:
        """
        labels = self._get_label(
            idx=idx,
            label_level=label_level
        )

        return dict(labels=labels)
