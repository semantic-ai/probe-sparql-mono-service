from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .helpers import GenerateTaxonomyStatistics
from .data_models import TaxonomyTypes
from .enums import EndpointType

import os
import mlflow

from uuid import uuid4
from fire import Fire


def main(
        max_level: int = 4
):
    """
    This main function provides access to the dataset statistic generation class. This is usefull to extract
    information about the label/ data distribution in the provided taxonomy annotated decision pool.

    The goal here is to create an overview of what labels have what degree of class balance/ samples.

    :param max_level: The maximum depth to calculate these summary statistics for
    :return: nothing, everything is logged in mlflow artifacts.
    """

    # creation of globally used variables
    config = Config()
    logger = LoggingBase(config.logging).logger
    request_handler = RequestHandler(config=config, logger=logger)

    # retrieving all taxonomies
    taxonomy_types = TaxonomyTypes.from_sparql(
        config=config.data_models,
        logger=logger,
        request_handler=request_handler,
        endpoint=EndpointType.TAXONOMY
    )

    # configuring local caching storage
    local_storage_dir = os.path.join("/tmp/summary_stats", uuid4().hex)
    os.makedirs(local_storage_dir, exist_ok=True)

    with mlflow.start_run():

        # mlflow.log_dict({"taxonomies": taxonomy_types.taxonomies}, "taxonomies.json")
        logger.info(f"current taxonomies: {[t.uri for t in taxonomy_types.taxonomies]}")
        for taxonomy in taxonomy_types.taxonomies:

            logger.info(f"Current taxonomy: {taxonomy.uri}")
            statistics = GenerateTaxonomyStatistics.from_sparql(
                config=config,
                logger=logger,
                request_handler=request_handler,
                taxonomy_uri=taxonomy.uri,
                max_level=max_level,
                local_storage_dir=local_storage_dir,
            )

            statistics.calculate_stats()

        mlflow.log_artifacts(local_storage_dir)


if __name__ == "__main__":
    Fire(main)
