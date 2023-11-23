from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..dataset import DatasetBuilder

from .base import Training
from .distilbert import DistilBertTraining
from .setfit import SetfitTraining
from .bert_multilabel import BertTraining

from ..enums import TrainingFlavours


def get_training_module(
        config: Config,
        logger: Logger,
        base_model_id: str,
        dataset_builder: DatasetBuilder,
        setfit_head=None,
        sub_node=None,
        nested_mlflow_run: bool = False
) -> Training:
    """
    This function returns the model that is selected using the config file.


    :param config: the global config to use
    :param logger: the global logger to use
    :param base_model_id: model id that is used as base model
    :param dataset_builder: dataset builder object
    :param setfit_head: setfit head to use (only relevant when pulling setfit models)
    :param sub_node: specific node reference to train on
    :return:
    """
    match config.run.model.flavour:

        case TrainingFlavours.SETFIT | TrainingFlavours.SETFIT.value:
            logger.debug("Selected Setfit")
            return SetfitTraining(
                config=config,
                logger=logger,
                base_model_id=base_model_id,
                dataset_builder=dataset_builder,
                setfit_head=setfit_head,
                sub_node=sub_node,
                nested_mlflow_run=nested_mlflow_run
            )

        case TrainingFlavours.DISTIL_BERT | TrainingFlavours.DISTIL_BERT.value:
            logger.debug("Selected Distilbert")
            return DistilBertTraining(
                config=config,
                logger=logger,
                base_model_id=base_model_id,
                dataset_builder=dataset_builder,
                sub_node=sub_node,
                nested_mlflow_run=nested_mlflow_run
            )

        case TrainingFlavours.BERT | TrainingFlavours.BERT.value:
            logger.debug("Selected Bert")
            return BertTraining(
                config=config,
                logger=logger,
                base_model_id=base_model_id,
                dataset_builder=dataset_builder,
                sub_node=sub_node,
                nested_mlflow_run=nested_mlflow_run
            )

        case _:
            raise ValueError("Provided training module does not exists!")
