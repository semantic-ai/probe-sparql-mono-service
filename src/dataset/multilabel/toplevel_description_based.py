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


class MultiLabelTopLevelDescriptionBased(MultiLabelTopLevelFullText):
    """
    [Adaptation from base class] This implementation responds with description only

    PARENT DOCS:
    ---
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
        [Adapted implementation] get text returns only returns description
        """
        __doc__ = MultilabelTrainingDataset._get_text.__doc__

        data_record = self.dataset[idx]

        short_title = data_record.get("short_title", "")
        description = data_record.get("motivation", "")

        return f"""\
        Een besluit over: {description}
        """

    _get_text.__doc__ += MultilabelTrainingDataset._get_text.__doc__
