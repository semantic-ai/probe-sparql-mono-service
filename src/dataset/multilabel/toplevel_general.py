from __future__ import annotations
from abc import ABC

from .base import MultilabelTrainingDataset

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.data_models import Taxonomy
    from logging import Logger

# other imports
from transformers import AutoTokenizer
from torch import device


class MultiLabelTopLevelFullText(MultilabelTrainingDataset):
    """
    [Adaptation from base class] This implementation responds with the full text

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

    def _get_text(self, idx: int) -> str:
        """
        [Adapted implementation] get text returns full text
        """
        __doc__ = MultilabelTrainingDataset._get_text.__doc__

        data_record = self.dataset[idx]

        short_title = data_record.get("short_title", "")
        motivation = data_record.get("motivation", "")
        description = data_record.get("motivation", "")

        if isinstance(articles := data_record.get("articles", []), list):
            articles = "\n\n".join(articles)

        return f"""\
        {short_title}: 

        {motivation} 
        Artikels:
        {articles}

        {description}
        """

    _get_text.__doc__ += MultilabelTrainingDataset._get_text.__doc__

