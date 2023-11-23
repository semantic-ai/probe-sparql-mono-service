from __future__ import annotations
import enum


class DatasetType(str, enum.Enum):
    """
    This enum is used to specify what type of dataset to use,
    """
    # default
    UNPROCESSED: str = "mirror"

    # single-label dataset
    SINGLE_BASIC: str = ""
    SINGLE_TOP_LEVEL_ALL_BASED: str = "s1_general"

    # multilabel dataset
    MULTI_SECOND_LEVEL_ALL_BASED: str = "m2_general"
    MULTI_TOP_LEVEL_ALL_BASED: str = "m1_general"
    MULTI_TOP_LEVEL_ARTICLE_BASED: str = "m1_article"
    MULTI_TOP_LEVEL_ARTICLE_SPLIT: str = "m1_article_split"
    MULTI_TOP_LEVEL_DESCRIPTION_BASED: str = "m1_description"
    MULTI_TOP_LEVEL_MOTIVATION_BASED: str = "m1_motivation"
    MULTI_TOP_LEVEL_SHORT_TITLE_BASED: str = "m1_shorttitle"

    DYNAMIC: str = "dynamic_general"
    # summary statistic dataset
    SUMMARY_STATISTIC_DATASET: str = "summary_stat_dataset"

    # other?

    @classmethod
    def _list(cls):
        """
        internal classmethod that allows us to retrieve all possible datasets
        :return:
        """
        return list(map(lambda c: c.value, cls))

    @staticmethod
    def get_multilevel_datasets(level: int = 1):
        """
        this function allows us to retrieve only the multilabel datasets of a specific level

        :param level: the label level you want to retrieve datasets for
        :return: a list with dataset that comply with the filter
        """
        lvl = None
        match level:
            case 1:
                lvl = "m1"

        return [v for v in DatasetType._list() if v.split("_")[0] == lvl]

    @staticmethod
    def get_from_prefix(model_type: str):
        """
        this function allows us to retrieve only the models compliant with the prefix filter

        :param model_type: the string prefix to filter the models with
        :return: a list with models that comply with the filter
        """
        return [v for v in DatasetType._list() if v.startswith(model_type)]

    def get_single_level_datasets(self):
        return NotImplementedError
