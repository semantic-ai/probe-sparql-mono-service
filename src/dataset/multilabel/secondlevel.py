from __future__ import annotations
from abc import ABC

from .base import MultilabelTrainingDataset

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

# other imports
from transformers import AutoTokenizer
from torch import device

from .toplevel_general import MultiLabelTopLevelFullText


class MultiLabelSecondLevelFullText(MultilabelTrainingDataset):
    """
    [Adaptation from base class] This implementation uses lvl2 labels

    PARENT DOCS:
    ---
    """
    __doc__ += str(MultilabelTrainingDataset.__doc__)

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

        super().__init__(
            config=config,
            logger=logger,
            taxonomy=taxonomy,
            tokenizer=tokenizer,
            dataset=dataset,
            _device=device(_device),
            sub_node=sub_node
        )

        self.max_depth = 2

    def _get_text(self, idx: int) -> str:
        """
        [Adapted implementation] get text for second level dataset.
        """
        __doc__ = MultiLabelTopLevelFullText._get_text.__doc__

        data_record = self.dataset[idx]

        short_title = data_record.get("short_title", "")
        motivation = data_record.get("motivation", "")
        description = data_record.get("description", "")

        if isinstance(articles := data_record.get("articles", []), list):
            articles = "\n\n".join(articles)

        return f"""\
        Een besluit over: 
        {short_title}: {motivation} 
        Artikels: {articles}
        description: {description}
        """

    def _get_label(self, idx: int) -> list[int]:
        """
        [Adapted implementation] get label returns only second level labels
        """
        __doc__ = MultilabelTrainingDataset._get_label.__doc__

        labels = []

        selected_record = self.dataset[idx]
        for label in selected_record.get("labels", []):
            label_in_tree = self.taxonomy.find(label)

            self.logger.debug(f"Label ({label}) found in tree {label_in_tree}")

            if second := label_in_tree.get(2):
                labels.append(second.get("label"))

        if self.config.run.dataset.apply_mlb:
            self.logger.debug("mlb")
            _labels = self.binarized_label_dictionary
            for label in labels:
                _labels[label] = 1

            self.logger.debug(f"Current mlb labels: {_labels}")
            return list(_labels.values())

        return labels

    _get_label.__doc__ += MultilabelTrainingDataset._get_label.__doc__

