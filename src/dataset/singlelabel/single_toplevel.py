from __future__ import annotations
from abc import ABC

from .base import TrainingDataset

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.data_models import Taxonomy
    from logging import Logger

# other imports
from transformers import AutoTokenizer


class SingleTopLevel(TrainingDataset):
    """
    [Adapated from baseclase] This implementation returns full text

    PARENT DOCS:
    ---
    """
    __doc__ += TrainingDataset.__doc__

    def __init__(
            self,
            config: Config,
            logger: Logger,
            taxonomy: Taxonomy,
            data: dict[str, str],
            tokenizer: AutoTokenizer = None,
            device: str = "cpu"
    ):

        super().__init__(
            config=config,
            logger=logger,
            taxonomy=taxonomy,
            tokenizer=tokenizer,
            data=data,
            device=device
        )

    def _get_label(self, idx: int) -> str:
        """
        This function implements the abstract logic for building the labels, it can be overwritten and adapted without
        problems (as long as the default input output signature is kept).

        :param idx: the integer value of the input index
        :return: the label
        """
        return self.dataset[idx].get("labels")

    def _get_text(self, idx: int) -> str:
        """
        This function implements the abstract logic in order to retrieve the text from the provided dataset.
        It is possible to overwrite this function and define custom behaviour to create the text input for the model.

        :param idx: the integer value of the input index
        :return: a list of integer values where the labels should be predicted
        """

        data_record = self.dataset[idx]

        short_title = data_record.get("short_title", "")
        motivation = data_record.get("motivation", "")
        description = data_record.get("motivation", "")
        articles = "\n".join(data_record.get("articles", []))

        return f"""\
        {short_title}: 
        
        {motivation} 
        Artikels:
        {articles}
        
        {description}
        """

    def __getitem__(self, idx) -> dict[str, str]:
        label = self._get_label(idx) if self.config.run.dataset.get_label else []
        text = self._get_text(idx)

        return dict(uri=self.dataset[idx].get("uri"), text=text, label=label)
