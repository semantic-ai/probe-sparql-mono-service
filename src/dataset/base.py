from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_models import Taxonomy

import abc

from torch.utils.data import Dataset


class TrainDataset(Dataset, abc.ABC):
    """
    Abstract training dataset class.



    """
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def _get_label(self, idx: int) -> list[int]:
        pass

    @abc.abstractmethod
    def _get_text(self, idx: int) -> str:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx) -> dict[str, str]:
        pass

    @property
    @abc.abstractmethod
    def candid_labels(self) -> list[str]:
        pass

    @candid_labels.setter
    @abc.abstractmethod
    def candid_labels(self, value: list[str]) -> None:
        pass

    @property
    @abc.abstractmethod
    def binarized_label_dictionary(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def max_label_depth(self) -> int:
        pass

    @max_label_depth.setter
    @abc.abstractmethod
    def max_label_depth(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def taxonomy(self) -> Taxonomy:
        pass

    @taxonomy.setter
    @abc.abstractmethod
    def taxonomy(self, value: Taxonomy):
        pass

    @abc.abstractmethod
    def get_specific_record(self, idx: int, label_level: int) -> dict[str, str | list]:
        pass
