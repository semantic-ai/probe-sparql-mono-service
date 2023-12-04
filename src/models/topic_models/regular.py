from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...dataset import DatasetBuilder
    from logging import Logger

from .base import TopicModel

import pandas as pd
from bertopic import BERTopic
from abc import ABC
from uuid import uuid4
import mlflow, os


class RegularTopicModel(TopicModel):
    """
    Regular topic modeling implementation.

    This class creates the following artifacts:
        1. Topic report (keywords per topic and the found topics in excels sheet)
        2. Topic distribution plot (barchart that visualizes what amount of docs can be found under the topics)
        3. Topic heatmap (a plot that visually can represent how close certain topics are to each-other)
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            dataset_builder: DatasetBuilder
    ) -> None:

        super().__init__(
            config=config,
            logger=logger,
            dataset_builder=dataset_builder
        )

        self._create_dataset()
        self.base_path = os.path.join("/tmp", uuid4().hex)
        os.makedirs(self.base_path, exist_ok=True)

    def _create_dataset(self):
        self.df = pd.DataFrame(self.dataset_builder.train_dataset)

    def analyse(self):

        with mlflow.start_run():
            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(self.df.description.tolist())

            topic_model.visualize_topics().write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "topic_model_visual.html"
                    ),
                    "w"
                )
            )

