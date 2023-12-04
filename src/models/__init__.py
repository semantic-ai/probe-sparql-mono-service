from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..data_models import Taxonomy
    from ..dataset import DatasetBuilder

from .base import Model
from .embedding import EmbeddingModel, SentenceEmbeddingModel, ChunkedEmbeddingModel, GroundUpRegularEmbeddingModel, \
    GroundUpGreedyEmbeddingModel, ChildLabelsEmbeddingModel
from .zeroshot import ZeroshotModel, SentenceZeroshotModel, ChunkedZeroshotModel, ChildLabelsZeroshotModel
from .classifier import ClassifierModel, HuggingfaceModel
from .hybrid import HybridModel, SelectiveHybridModel
from .topic_models import RegularTopicModel, HierarchicTopicModel, DynamicTopicModel

from ..enums import ModelType


def get_model(
        config: Config,
        logger: Logger,
        model_id: str,
        taxonomy: Taxonomy,
        specific_model_type: ModelType = None,
        **kwargs
) -> Model:
    """
    This function provides you with an instantiated model based on the provided arguments and config.

    :param config: the global config object
    :param logger: the global logger object
    :param model_id: model it to use as base model weights
    :param taxonomy: the taxonomy to use for label predicitons
    :param specific_model_type: model flavour explicitly defined (overrule config)
    :param kwargs: extra kwargs used to initialize models
    :return: An instance of Model class
    """

    logger.debug(f"received model config type {config.run.model.type}")
    logger.debug(f"received model id {config.run.model.type}")

    match specific_model_type or config.run.model.type:

        ##
        # Zeroshot models
        ##

        case ModelType.ZEROSHOT_REGULAR.value | ModelType.ZEROSHOT_REGULAR:
            logger.info("Selected ZeroshotModel")
            return ZeroshotModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.ZEROSHOT_SENTENCE.value | ModelType.ZEROSHOT_SENTENCE:
            logger.info("Selected SentenceZeroshotModel")
            return SentenceZeroshotModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.ZEROSHOT_CHUNKED.value | ModelType.ZEROSHOT_CHUNKED:
            logger.info("Selected ChunkedZeroshotModel")
            return ChunkedZeroshotModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.ZEROSHOT_CHILD_LABELS.value | ModelType.ZEROSHOT_CHILD_LABELS:
            logger.info("Selected ChildLabelsZeroshotModel")
            return ChildLabelsZeroshotModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        ##
        # Embedding models
        ##

        case ModelType.EMBEDDING_REGULAR.value | ModelType.EMBEDDING_REGULAR:
            logger.info("Selected EmbeddingModel")
            return EmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.EMBEDDING_SENTENCE.value | ModelType.EMBEDDING_SENTENCE:
            logger.info("Selected SentenceEmbeddingModel")
            return SentenceEmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.EMBEDDING_GROUND_UP | ModelType.EMBEDDING_GROUND_UP.value:
            logger.info("Selected Embedding GroundUp")
            return GroundUpRegularEmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.EMBEDDING_GROUND_UP_GREEDY | ModelType.EMBEDDING_GROUND_UP_GREEDY.value:
            logger.info("Selected Embedding GrounUpGreedy")
            return GroundUpGreedyEmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.EMBEDDING_CHUNKED.value | ModelType.EMBEDDING_CHUNKED:
            logger.info("Selected ChunkedEmbeddingModel")
            return ChunkedEmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        case ModelType.EMBEDDING_CHILD_LABELS.value | ModelType.EMBEDDING_CHILD_LABELS:
            logger.info("Selected ChildLabelsEmbeddingModel")
            return ChildLabelsEmbeddingModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy
            )

        ###
        # Classifier model
        ###

        case ModelType.HUGGINGFACE_MODEL | ModelType.HUGGINGFACE_MODEL.value:
            logger.info("Selected HuggingfaceModel")
            logger.info(f"huggingface model_id {model_id}")
            return HuggingfaceModel(
                config=config,
                logger=logger,
                model_id=model_id,
                taxonomy=taxonomy,
                stage=kwargs.get("model_stage", "Production")
            )

        ###
        # Other models
        ###

        case ModelType.HYBRID_BASE_MODEL | ModelType.HYBRID_BASE_MODEL.value:
            logger.info("Selected HybridModel")
            return HybridModel(
                config=config,
                logger=logger,
                taxonomy=taxonomy,
                supervised_model=kwargs.get("supervised_model"),
                unsupervised_model=kwargs.get("unsupervised_model"),
            )

        case ModelType.HYBRID_SELECTIVE_MODEL | ModelType.HYBRID_SELECTIVE_MODEL.value:
            logger.info("Selected HybridModel")
            return SelectiveHybridModel(
                config=config,
                logger=logger,
                taxonomy=taxonomy,
                supervised_model=kwargs.get("supervised_model"),
                unsupervised_model=kwargs.get("unsupervised_model"),
            )

        case _:
            raise NotImplementedError("No such model available")


def get_topic_model(
        model_type: ModelType,
        config: Config,
        logger: Logger,
        dataset_builder: DatasetBuilder
):
    """
    Model provided specifically for the topic models.


    :param model_type: the specific model type requested
    :param config: the global config object
    :param logger: the global logger object
    :param dataset_builder: the dataset builder object containing all the relevant information that could be used for
                            the topic modeling
    :return: An instance of the requested topic modeling
    """

    match model_type:

        case ModelType.REGULAR_TOPIC_MODEL | ModelType.REGULAR_TOPIC_MODEL.value:
            logger.info("Selected RegularTopicModel")
            return RegularTopicModel(
                config=config,
                logger=logger,
                dataset_builder=dataset_builder
            )

        case ModelType.DYNAMIC_TOPIC_MODEL | ModelType.DYNAMIC_TOPIC_MODEL.value:
            logger.info("Selected DynamicTopicModel")
            return DynamicTopicModel(
                config=config,
                logger=logger,
                dataset_builder=dataset_builder
            )

        case ModelType.HIERARCHIC_TOPIC_MODEL | ModelType.HIERARCHIC_TOPIC_MODEL.value:
            logger.info("Selected HierarchicTopicModel")
            return HierarchicTopicModel(
                config=config,
                logger=logger,
                dataset_builder=dataset_builder
            )


        case _:
            raise NotImplementedError("No such topic-model available")

