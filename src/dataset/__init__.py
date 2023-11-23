from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from typing import Any
    from ..data_models import Taxonomy

# single label datasets
from .singlelabel import TrainingDataset, SingleTopLevel, \
    BasicDataset

# multilabel datasets
from .multilabel import MultiLabelTopLevelFullText, MultiLabelSecondLevelFullText, \
    MultiLabelTopLevelArticleBased, MultiLabelTopLevelDescriptionBased, MultiLabelTopLevelMotivationBased, \
    MultiLabelTopLevelShortTitleBased, MultiLabelTopLevelArticleSplit, SummaryStatisticDataset,\
    DynamicMultilabelTrainingDataset

from .builder import DatasetBuilder
from .base import TrainDataset
from ..enums import DatasetType


def create_dataset(
        config: Config,
        logger: Logger,
        dataset: list[dict[str, Any]],
        taxonomy: Taxonomy,
        tokenizer: object = None,
        sub_node: str = None,
) -> TrainDataset | list[dict]:
    """
    Function that creates the dataset based on the configuration that is provided

    :param sub_node: sub_node to reselect input data for
    :param tokenizer: tokenizer to use in the dataset if provided
    :param config: configuration object
    :param logger: logger object
    :param dataset: the created dataset (list of dict)
    :param taxonomy: the taxonomy that is used for
    :return:
    """

    match config.run.dataset.type:
        case DatasetType.MULTI_TOP_LEVEL_ALL_BASED.value | DatasetType.MULTI_TOP_LEVEL_ALL_BASED:
            logger.info("Selected MultiLabelTopLevelFullText")
            return MultiLabelTopLevelFullText(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node = sub_node
            )

        case DatasetType.MULTI_SECOND_LEVEL_ALL_BASED.value | DatasetType.MULTI_SECOND_LEVEL_ALL_BASED:
            logger.info("Selected MultiLabelSecondLevelFullText")
            return MultiLabelSecondLevelFullText(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.MULTI_TOP_LEVEL_ARTICLE_BASED.value | DatasetType.MULTI_TOP_LEVEL_ARTICLE_BASED:
            logger.info("Selected MultiLabelTopLevelArticleBased")
            return MultiLabelTopLevelArticleBased(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.MULTI_TOP_LEVEL_DESCRIPTION_BASED.value | DatasetType.MULTI_TOP_LEVEL_DESCRIPTION_BASED:
            logger.info("Selected MultiLabelTopLevelDescriptionBased")
            return MultiLabelTopLevelDescriptionBased(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.MULTI_TOP_LEVEL_MOTIVATION_BASED.value | DatasetType.MULTI_TOP_LEVEL_MOTIVATION_BASED:
            logger.info("Selected MultiLabelTopLevelMotivationBased")
            return MultiLabelTopLevelMotivationBased(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.MULTI_TOP_LEVEL_SHORT_TITLE_BASED.value | DatasetType.MULTI_TOP_LEVEL_SHORT_TITLE_BASED:
            logger.info("Selected MultiLabelTopLevelShortTitleBased")
            return MultiLabelTopLevelShortTitleBased(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.MULTI_TOP_LEVEL_ARTICLE_SPLIT.value | DatasetType.MULTI_TOP_LEVEL_ARTICLE_SPLIT:
            logger.info("Selected MultiLabelTopLevelArticleSplit")
            return MultiLabelTopLevelArticleSplit(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.SUMMARY_STATISTIC_DATASET | DatasetType.SUMMARY_STATISTIC_DATASET.value:
            logger.info("Selected SummaryStatisticDataset")
            return SummaryStatisticDataset(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.DYNAMIC | DatasetType.DYNAMIC.value:
            logger.info(f"DynamicMultilabelTrainingDataset")
            return DynamicMultilabelTrainingDataset(
                config=config,
                dataset=dataset,
                taxonomy=taxonomy,
                logger=logger,
                tokenizer=tokenizer,
                sub_node=sub_node
            )

        case DatasetType.UNPROCESSED | DatasetType.UNPROCESSED.value:
            return dataset
        case _:
            raise NotImplementedError("No such dataset available")