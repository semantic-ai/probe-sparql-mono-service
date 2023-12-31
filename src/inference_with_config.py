from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .data_models import Model
from .helpers import Inference, InferenceModelTree
from .enums import DecisionQuery, DatasetType

import mlflow
from fire import Fire
import ast
import json


def main(
        taxonomy_uri: str,
        model_config: dict,
        dataset_type: DatasetType,
        checkpoint_folder: str = None
):
    """
    This script implements the prediction based on config.

    The goal of the prediction based on config script is to rebuild a tree based on the provided models,
    these models take in a text and execute the predictions for the classes they are trained on.
    This process is done by computing the parent node, all sub nodes that are above a certain threshold are then also computed.
    This goes on recursively until it has reached the maximum node_depth for the provided config.

    :param taxonomy_uri: the taxonomy to use for the prediction
    :param model_config: the json config to use to create the prediction trees with
    :param dataset_type: the type of dataset to use for the model training
    :param checkpoint_folder: path to a local checkpoint, this can easily be used when debugging locally.
    """
    # defining global vars
    config = Config()
    logger = LoggingBase(
        config=config.logging
    ).logger
    request_handler = RequestHandler(
        config=config,
        logger=logger
    )
    logger.info(f"type: {type(model_config)} {model_config}")
    logger.info(f"register custom config")
    model = Model(
        config=config.data_models,
        logger=logger,
        name="config_based_inference_model",
        category="tree_config",
        mlflow_reference=model_config
    )
    query = model.write_to_sparql(request_handler)

    logger.info(f"start creating dataset")

    if checkpoint_folder is None:
        dataset_builder = DatasetBuilder.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            taxonomy_uri=taxonomy_uri,
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
    logger.info(f"Successfully created dataset")

    logger.info(f"Loading model config: {str(model_config)}")
    model_config = ast.literal_eval(str(model_config))

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
        model_reference=model
    )

    inference.execute()


if __name__ == "__main__":
    Fire(main)
