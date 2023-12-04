from __future__ import annotations

from .regular import RegularTopicModel
from ...dataset import create_dataset

from bertopic import BERTopic
import mlflow, os
import pandas as pd


class DynamicTopicModel(RegularTopicModel):

    def _create_dataset(self):
        dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy
        )
        df = pd.DataFrame(list(dataset))
        self.docs = df.text.tolist()

        # TODO: verify if we can provide the timestamps based on the current sparql endpoint
        self.timestamps = df.timestamps.tolist()

    def classify(self):
        with mlflow.start_run():
            topic_model = BERTopic()

            topics_over_time = topic_model.topics_over_time(self.docs, self.timestamps)
            topic_model.visualize_topics_over_time(
                topics_over_time
            ).write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "topics_over_time.html"
                    ),
                    "w"
                )
            )
