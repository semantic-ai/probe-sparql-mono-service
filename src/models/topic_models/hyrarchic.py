from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ...dataset import DatasetBuilder
    from logging import Logger

from .base import TopicModel
from .regular import RegularTopicModel

import pandas as pd
from bertopic import BERTopic
from abc import ABC
import mlflow


class HyrarchicTopicModel(RegularTopicModel):
    """
    Regular topic modeling implementation.

    This class creates the following artifacts:
        1. Topic report (keywords per topic and the found topics in excels sheet)
        2. Topic distribution plot (barchart that visualizes what amount of docs can be found under the topics)
        3. Topic heatmap (a plot that visually can represent how close certain topics are to each-other)
    """

    def analyse(self):
        with mlflow.start_run():
            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(self.df.description.tolist())

            topic_model.visualize_topics()
