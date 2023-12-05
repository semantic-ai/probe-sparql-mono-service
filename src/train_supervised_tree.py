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
from .enums import DecisionQuery, DatasetType, TrainingFlavours

import mlflow
import fire
import copy


def main(
        train_flavour: TrainingFlavours = None,
        dataset_type: DatasetType = None,
        model_id: str = None,
        train_test_split: bool = True,
        checkpoint_folder: str = None,
        max_depth: int = 2,
        taxonomy_url: str = "http://stad.gent/id/concepts/gent_words"
):
    """
    This script provides the functionality to train all supervised models up to a predefined depth.
    It enables an easy way to bootstrap the models to be t

        1. You can train a classifier on an entire level of the taxonomy (Only relevant if there is enough information
        available per label in deeper taxonomies nodes.)

        2. You can train a classifier on a specific node of the taxonomy (i.e. level 1 taxonomy is trained on the specific start
        node of the taxonomy, while a node from the level 1 can be used for training on a specific sub node in the taxonomy.)

    Example usage:
        >>>

    :param max_depth:
        The maximum depth the taxonomy tree training should be used on
    :param train_flavour:
        The specific type of model training you would want, this is a one on one mapping with the training flavour enum
        It allows you to select a specific type of model you would like to train (BERT, DISTIL_BERT, SETFIT)
    :param dataset_type:
        The specific type of dataset you want to use for model training, this is another one on one mapping with the
        dataset type enum. for more information about what dataset types are available, check out enums.
    :param model_id:
        What base model to use when training one of the selected model flavours, this also can be left empty. The defaults
        are provided in the config and will be pulled from there.
    :param train_test_split:
        Flag that allows you to do train_test split, this will force your code to create a split during the creation of
        your dataset. The behaviour is specified with the config (predefined split yes/no , split size ...)
    :param checkpoint_folder:
        Mainly used for debugging, when you have a pre-downloaded dataset checkpoints, you can provide this here.
    :param taxonomy_url:
        The taxonomy to use for the model training.:return:
    """
    config = Config()
    logger = LoggingBase(config=config.logging).logger
    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    # setting default train flavour
    if train_flavour is None:
        train_flavour = TrainingFlavours.BERT
        model_id = None  # overwriting provided model id -> makes no sense if it is filled in without a flavour
        config.run.model.flavour = train_flavour
    else:
        config.run.model.flavour = train_flavour

    # loading default if no model_id is provided
    if model_id is None:
        model_id = TrainingFlavours.get_default_model_for_type(
            config=config,
            model_flavour=train_flavour
        )

    # adding values to config for further use
    if dataset_type is None:
        config.run.dataset.type = DatasetType.MULTI_TOP_LEVEL_ALL_BASED
    else:
        config.run.dataset.type = dataset_type

    if checkpoint_folder is None:
        dataset_builder = DatasetBuilder.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            taxonomy_uri=taxonomy_url,
            query_type=DecisionQuery.ANNOTATED,
            do_train_test_split=train_test_split,
        )
        dataset_builder.create_checkpoint("/tmp/data")

    else:

        dataset_builder = DatasetBuilder.from_checkpoint(
            config=config,
            logger=logger,
            checkpoint_folder=checkpoint_folder
        )

    def train(taxonomy_node: Taxonomy, depth: int = 1) -> None:
        """
        Function to execute recursive training

        :param depth: the current depth the training is executed on
        :param taxonomy_node: the taxonomy subnode to train on
        :return: Nothing at all
        """

        with mlflow.start_run(run_name=f"{train_flavour}_{taxonomy_node.uri}", nested=True):
            training_module = get_training_module(
                config=config,
                logger=logger,
                base_model_id=model_id,
                dataset_builder=copy.deepcopy(dataset_builder),
                sub_node=None if depth == 1 else taxonomy_node.uri,
                nested_mlflow_run=True
            )
            training_module.train()

            try:
                new_depth = depth + 1
                for child_taxonomy in taxonomy_node.children:
                    if new_depth > max_depth:
                        logger.info(f"Did not train on {child_taxonomy.label} since its beyond the max defined depth")
                    else:
                        train(taxonomy_node=child_taxonomy, depth=new_depth)
            except Exception as ex:
                logger.warning(f"Failed on {child_taxonomy.uri}, exception: {ex}")

    train(dataset_builder.taxonomy)


if __name__ == "__main__":
    fire.Fire(main)
