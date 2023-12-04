from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import Taxonomy

from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .training import get_training_module
from .enums import DecisionQuery, DatasetType, ModelType
from .models import get_topic_model

import mlflow
import fire
import copy


def main(
        dataset_type: DatasetType,
        model_type: ModelType,
        taxonomy_uri: str = "http://stad.gent/id/concepts/gent_words",
        checkpoint_folder: str = None
):
    config = Config()
    config.run.dataset.type = dataset_type
    logger = LoggingBase(config=config.logging).logger
    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    if checkpoint_folder is None:
        dataset_builder = DatasetBuilder.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            taxonomy_uri=taxonomy_uri,
            query_type=DecisionQuery.ALL,
            do_train_test_split=False
        )
        dataset_builder.create_checkpoint("data")

    else:

        dataset_builder = DatasetBuilder.from_checkpoint(
            config=config,
            logger=logger,
            checkpoint_folder=checkpoint_folder
        )

    topic_model = get_topic_model(
        model_type=model_type,
        config=config,
        logger=logger,
        dataset_builder=dataset_builder
    )

    topic_model.analyse()


if __name__ == "__main__":
    fire.Fire(main)
