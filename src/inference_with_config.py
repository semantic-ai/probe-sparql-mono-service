from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .helpers import Inference, InferenceModelTree
from .training import get_training_module
from .enums import DecisionQuery, DatasetType, TrainingFlavours

import mlflow
from fire import Fire
import json


def main(
        taxonomy_uri: str,
        model_config: str,
        dataset_type: DatasetType,
        checkpoint_folder: str = None
):
    # defining global vars
    config = Config()
    logger = LoggingBase(
        config=config.logging
    ).logger
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
            do_train_test_split=False,
        )
        dataset_builder.create_checkpoint("data")

    else:

        dataset_builder = DatasetBuilder.from_checkpoint(
            config=config,
            logger=logger,
            checkpoint_folder=checkpoint_folder
        )

    model_tree = InferenceModelTree.from_model_config(
        config=config,
        logger=logger,
        request_handler=request_handler,
        model_configuration=model_config
    )

    inference = Inference(
        config=config,
        logger=logger,
        request_handler=request_handler,
        dataset_builder=dataset_builder,
        inference_model_tree=model_tree,
        dataset_type=dataset_type,
    )

    inference.execute()


if __name__ == "__main__":
    d = {"uri":"http://stad.gent/id/concepts/business_capabilities","model_id": "mlflow:/bert__business_capabilities__parent_node","flavour":"huggingface_model","stage":"Production","sub_nodes":[{"uri": "http://stad.gent/id/concepts/business_capabilities/concept_90","model_id":"mlflow:/bert__business_capabilities__ondersteunende_capabilities","flavour":"huggingface_model","stage":"Production","sub_nodes":[]},{"uri":"http://stad.gent/id/concepts/business_capabilities/concept_1","model_id":"mlflow:/bert__business_capabilities__sturende_capabilities","flavour":"huggingface_model","stage":"Production","sub_nodes":[]},{"uri":"http://stad.gent/id/concepts/business_capabilities/concept_13","model_id":"mlflow:/bert__business_capabilities__uitvoerende_capabilities","flavour":"huggingface_model","stage":"Production","sub_nodes":[]}]}
    with open("test.txt", "w+") as f:
        json.dump(d, f)

    Fire(main)





