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
from ...dataset import create_dataset
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
            dataset_builder: DatasetBuilder,
            embedding_model: str = None
    ) -> None:
        super().__init__(
            config=config,
            logger=logger,
            dataset_builder=dataset_builder
        )

        self.embedding_model = embedding_model

        self._create_dataset()
        self.base_path = os.path.join("/tmp", uuid4().hex)
        os.makedirs(self.base_path, exist_ok=True)

    def _create_dataset(self):
        dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy
        )
        df = pd.DataFrame(list(dataset))
        self.docs = df.text.tolist()

    def analyse(self):
        with mlflow.start_run():

            min_topic_size = 5 if len(self.docs) < 500 else 10
            topic_model = BERTopic(
                min_topic_size=min_topic_size,
                top_n_words=10,
                embedding_model=self.embedding_model,
                n_gram_range=(1, 1),  # fun param to play with
                language="dutch"
            )
            topics, probs = topic_model.fit_transform(self.docs)

            topic_model.get_topic_info().to_csv(
                open(
                    os.path.join(
                        self.base_path,
                        "visualized_topics.csv"
                    ),
                    "w"
                )
            )

            topic_model.visualize_heatmap().write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "heatmap.html"
                    ),
                    "w"
                )
            )

            topic_model.visualize_documents(self.docs).write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "visualized_documents.html"
                    ),
                    "w"
                )
            )

            topic_model.visualize_barchart().write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "topic_barchart.html"
                    ),
                    "w"
                )
            )

            mlflow.log_artifacts(
                local_dir=self.base_path,
                artifact_path="artifacts"
            )
