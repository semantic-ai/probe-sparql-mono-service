from .config import Config
from .dataset import DatasetBuilder, create_dataset
from .enums import DecisionQuery
from .sparql import RequestHandler
from .utils import LoggingBase
from .enums import DatasetType

import logging
import mlflow
import json
import os

from fire import Fire
from uuid import uuid4


def main(
        dataset_type: DatasetType = DatasetType.UNPROCESSED,
        taxonomy_uri: str = "http://stad.gent/id/concepts/gent_words",
        checkpoint_location: str = None
):
    """
    This function provides access to the creation and retrieval of training datasets.
    Based on the provided configuration, it will create the respective dataset.

    :param checkpoint_location:  The location of the pre-downloaded dataset information (This must be of the same taxonomy type)
    :param dataset_type: The type of dataset you want to extract
    :param taxonomy_uri: The taxonomy you want to create the dataset for.
    :return: Nothin, all artifacts are logged to mlflow
    """

    # setting up global variables
    config = Config()
    config.run.dataset.type = dataset_type

    logger = LoggingBase(config.logging).logger
    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    with mlflow.start_run():

        # build dataset
        if checkpoint_location is None:
            dataset_builder = DatasetBuilder.from_sparql(
                config=config,
                logger=logger,
                request_handler=request_handler,
                taxonomy_uri=taxonomy_uri,
                query_type=DecisionQuery.ANNOTATED,
                limit=5000,
            )
        else:
            dataset_builder = DatasetBuilder.from_checkpoint(
                config=config,
                logger=logger,
                checkpoint_folder=checkpoint_location
            )

        # select specific dataset flavour
        dataset = create_dataset(
            config=config,
            logger=logger,
            dataset=dataset_builder.train_dataset,
            taxonomy=dataset_builder.taxonomy
        )

        # creating list of json records for the given dataset
        dataset_memory = []
        for i in range(len(dataset)):
            dataset_memory.append(dataset[i])

        artifact_folder = f"/tmp/mlflow_artifacts_{uuid4().hex}"
        os.makedirs(artifact_folder, exist_ok=True)
        with open(os.path.join(artifact_folder, "dataset"), "w+") as f:
            json.dump(dataset_memory, f)

        mlflow.log_artifacts(artifact_folder)


if __name__ == "__main__":
    Fire(main)
