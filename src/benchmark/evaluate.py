from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..models import Model

    from logging import Logger
    from ..dataset import TrainDataset
    from typing import Any

from .metrics import Metrics
from tqdm import tqdm
import pandas as pd
import os

from uuid import uuid4


# TODO FIX DATASET NAMING LATER


class MultilabelEvaluation:
    """
    This class is the framework that executes a specific evaluation of a model.
    The evaluation loops over the dataset and generates predictions for each record, once completed it executes the metric
    calculation script.

    Example usage:
        >>> multilabel_eval = MultilabelEvaluation(
                config=Config(),
                logger=logging.logger,
                model = Model(...),
                dataset = TrainingDataset(...),
                multilabel = True
            )
        >>> multilabel_eval.evaluate()
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model: Model,
            dataset: TrainDataset,
            multilabel: bool = True
    ):
        # dataset
        self.config = config
        self.logger = logger
        self.dataset: TrainDataset = dataset
        self.model = model

        self.base_folder = os.path.join("/tmp/probe", uuid4().hex)

        # model_id
        self.candid_labels = self.dataset.candid_labels
        self.multilabel = multilabel
        self.metrics = Metrics(
            config=config.run.benchmark,
            logger=logger,
            base_folder=self.base_folder,
            model_id=self.model.model_id,
            classes=self.candid_labels
        )

    def evaluate(self) -> None:
        """
        This function starts the evaluation process for the given dataset
        :return:
        """

        # when the dataset is not large, the next line is an easy optimization
        data: list[dict[str, Any]] = [d for d in self.dataset]
        columns = self.dataset.candid_labels

        predictions = pd.DataFrame(
            [
                self.model.classify(
                    text=text.get("text"),
                    multi_label=self.multilabel
                )
                for text in tqdm(data, desc="generating prediction mlb array")
            ]
        )[columns]

        true_labels = pd.DataFrame([
            d.get("labels")
            for d in tqdm(data, desc="generating true labels mlb array")
        ],
            columns=columns
        )[columns]

        self.logger.debug(f"predictions {predictions}")
        self.logger.debug(f"ground truth {true_labels}")
        self.metrics.compute(true_labels.values, predictions.values)
