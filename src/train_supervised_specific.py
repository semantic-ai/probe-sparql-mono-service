import fire

from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .training import get_training_module
from .enums import DecisionQuery, DatasetType, TrainingFlavours, TrainerTypes

import mlflow


def main(
        train_flavour: TrainingFlavours = None,
        dataset_type: DatasetType = None,
        model_id: str = None,
        train_test_split: bool = True,
        checkpoint_folder: str = None,
        taxonomy_url: str = "http://stad.gent/id/concepts/gent_words",
        taxonomy_sub_node: str = None,
        trainer_type: TrainerTypes = TrainerTypes.CUSTOM
):
    """
    This script provides the functionality to train the supervised models, the training can be configured in multiple
    ways:

        1. You can train a classifier on an entire level of the taxonomy (Only relevant if there is enough information
        available per label in deeper taxonomies nodes.)

        2. You can train a classifier on a specific node of the taxonomy (i.e. level 1 taxonomy is trained on the specific start
        node of the taxonomy, while a node from the level 1 can be used for training on a specific sub node in the taxonomy.)

    Example usage:
        >>> python -m src.train_supervised_tree --train_flavour="bert" --dataset_type="dynamic_general" --taxonomy_url='http://stad.gent/id/concepts/business_capabilities' --checkpoint_folder="data/business_capabilities"


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
        The taxonomy to use for the model training.
    :param taxonomy_sub_node:
        If provided, it will train the model only on the specific sub-level node that has been selected.
        This will be used to train specialized models that represent a node in the taxonomy.
    :return:
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

    with mlflow.start_run():

        training_module = get_training_module(
            config=config,
            logger=logger,
            base_model_id=model_id,
            dataset_builder=dataset_builder,
            sub_node=taxonomy_sub_node,
            nested_mlflow_run=True,
            trainer_type=trainer_type
        )
        training_module.train()


if __name__ == "__main__":
    fire.Fire(main)
