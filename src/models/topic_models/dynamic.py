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

        # sub_selection to remove decisions where it was not able to extract year
        df = df[df.date != -1]

        self.docs = df.text.tolist()
        self.timestamps = df.date.tolist()

    def analyse(self):
        with mlflow.start_run():
            topic_model = BERTopic()

            topics, probs = topic_model.fit_transform(self.docs)

            topic_model.visualize_documents(self.docs).write_html(
                open(
                    os.path.join(
                        self.base_path,
                        "visualized_documents.html"
                    ),
                    "w"
                )
            )

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
