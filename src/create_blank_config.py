from .data_models import TaxonomyTypes
from .sparql import RequestHandler
from .config import Config
from .utils import LoggingBase
from .enums import EndpointType

import yaml
import fire
import uuid
import os
import json
import mlflow


def main():
    """
    This script can be used to re-generate a blank model config.
    This would be helpfull if the taxonomy is adapted, extended etc.

    :return: nothing, artifacts are pushed to a designated airflow run
    """
    config = Config()
    logger = LoggingBase(
        config=config.logging,
    ).logger

    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    taxonomies = TaxonomyTypes.from_sparql(
        config=config.data_models,
        logger=logger,
        request_handler=request_handler,
        endpoint=EndpointType.TAXONOMY
    )

    base_path = os.path.join("/tmp", uuid.uuid4().hex)

    for taxonomy in taxonomies.taxonomies:
        logger.info(f"creating blank configs for: {taxonomy.label}")

        sub_path = os.path.join(base_path, taxonomy.uri.split("/")[-1])
        os.makedirs(sub_path, exist_ok=True)

        with open(os.path.join(sub_path, "blank_model_config.json"), "w") as f:
            json.dump(taxonomy.create_blank_config(), f)

        with open(os.path.join(sub_path, "blank_model_config.yaml"), "w") as f:
            yaml.safe_dump(taxonomy.create_blank_config(), f)

    mlflow.log_artifacts(base_path)


if __name__ == "__main__":
    fire.Fire(main)
