from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import Taxonomy

from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .enums import DecisionQuery, DatasetType, ModelType
from .models import get_topic_model

import fire


def main(
        dataset_type: DatasetType,
        model_type: ModelType,
        checkpoint_folder: str = None
):
    """
    This function is the entrypoint for the topic modeling functionality.

    It calls on the specified class to generate the topic modeling artifacts that can be user for further analysis

    :param dataset_type: the type of dataset to use as input formatting
    :param model_type:  the type of topic modeling to use
    :param checkpoint_folder: a checkpoint that can be used to restore the input data from
    :return:
    """
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
            query_type=DecisionQuery.ALL,
            do_train_test_split=False
        )
        dataset_builder.create_checkpoint("/tmp/data")

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
