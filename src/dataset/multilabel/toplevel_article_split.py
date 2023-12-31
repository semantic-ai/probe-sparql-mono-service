from __future__ import annotations
from abc import ABC

from .base import MultilabelTrainingDataset

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger
    from transformers import AutoTokenizer

# other imports
from .toplevel_general import MultiLabelTopLevelFullText
from torch import device


class MultiLabelTopLevelArticleSplit(MultiLabelTopLevelFullText):
    """
    [Adaptation from base class] This adaptation splits decisions based on the articles

    PARENT DOCS:
    ---
    """
    __doc__ += MultilabelTrainingDataset.__doc__

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

        self._remap_dataset()

    def _remap_dataset(self) -> None:
        """
        This function remaps the input dataset to an article based dataset, splitting documents on separate articles.
        :return:
        """
        def article_dataset_generator():

            for record in self.dataset:
                articles = record.get("articles")
                if articles is not None:

                    for article in articles:
                        yield dict(
                            uri=record.get("uri"),
                            article=article,
                            labels=record.get("labels")
                        )

        self.dataset = list(article_dataset_generator())

    def _get_text(self, idx: int) -> str:
        """
        [Adapted implementation] get text returns only returns articles
        """

        data_record = self.dataset[idx]
        article = data_record.get("article", "")

        return article

    _get_text.__doc__ += MultilabelTrainingDataset._get_text.__doc__


