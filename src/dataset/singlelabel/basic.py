from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...data_models import Taxonomy
    from logging import Logger

from .base import TrainingDataset

import pandas as pd
from transformers import AutoTokenizer


class BasicDataset(TrainingDataset):
    """
    [unused]
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            taxonomy: Taxonomy,
            tokenizer: AutoTokenizer,
            data: pd.DataFrame,
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
