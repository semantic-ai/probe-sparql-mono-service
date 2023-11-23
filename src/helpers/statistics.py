from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.data_models import Taxonomy
    from src.sparql import RequestHandler
    from logging import Logger

from src.dataset import DatasetBuilder, create_dataset
from src.enums import DecisionQuery, DatasetType

import os
import mlflow

from uuid import uuid4
import matplotlib.pyplot as plt
from textwrap import wrap


class GenerateTaxonomyStatistics:
    """
    This class calculates the distribution of labels for a given taxonomy, dataset and label depth combination.

    Typical usage example:
        >>> dataset_builder = DatasetBuilder(...)
        >>> stats = GenerateTaxonomyStatistics(
                config=Config(),
                logger=logging.logger,
                dataset=dataset_builder.train_dataset
                taxonomy=Taxonomy(...),
                max_level=4,
                local_storage_dir="..."
            )
        >>> stats.calculate_stats()
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            dataset: list[dict[str, str]],
            taxonomy: Taxonomy,
            max_level: int,
            local_storage_dir: str
    ) -> None:

        self.config = config
        self.config.run.dataset.type = DatasetType.SUMMARY_STATISTIC_DATASET
        self.logger = logger

        self.taxonomy = taxonomy
        self.taxonomy_name = taxonomy.uri
        self.dataset = dataset
        self.max_level = max_level
        self.local_storage_dir = os.path.join(
            local_storage_dir,
            self.taxonomy_name.split("/")[-1]
        )

        os.makedirs(self.local_storage_dir, exist_ok=True)

        self._prep_dataset()

    def calculate_stats(self):
        """
        This function calculates the stats for each level up until the max level.

        :return:
        """
        for i in range(1, self.max_level + 1):
            self._generate_level_stats(i)

    def _prep_dataset(self) -> None:
        """
        This function creates a dataset object from the provided dataset.
        This dataset object does most of the remapping in order to easily calculate statistics.

        :return:
        """

        self.ds = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset,
            taxonomy=self.taxonomy
        )

    def _get_level_labels(self, level: int) -> list[str]:
        """
        This function is a wrapper around the taxonomy get_level_specific_labels function.

        :param level: level to retrieve labels from
        :return: the list of labels that occur on the provided level
        """
        return self.taxonomy.get_level_specific_labels(level=level)

    def _generate_level_stats(self, level: int) -> None:
        """
        internal function that does the actual calculations of the statistics about the label distribution

        :param level: the level to generate the statistics for
        :return: Nothing at al
        """

        def get_record_stats(idx: int, level: int):
            cache = []
            for label in self.ds.get_specific_record(
                    idx=idx,
                    label_level=level
            ).get("labels", []):
                cache.append(label)

            return cache

        label_log = {label: 0 for label in self._get_level_labels(level=level)}

        for i in range(0, len(self.ds)):
            for label in get_record_stats(
                    idx=i,
                    level=level
            ):
                label_log[label] += 1

        plt.figure(figsize=(24, 6))
        plt.subplots_adjust(bottom=0.6)
        plt.bar(
            ['\n'.join(wrap(label, 50)) for label in list(label_log.keys())],
            list(label_log.values()),
        )
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.local_storage_dir, f"distribution_label_level_{level:02}.png"))
        plt.clf()

    def __call__(self, *args, **kwargs) -> None:
        """
        This call-function wraps the calculate_stats functionality

        :param args:
        :param kwargs:
        :return: Nothing at all
        """
        self.calculate_stats()

    @classmethod
    def from_sparql(
            cls,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            taxonomy_uri: str,
            max_level: int,
            local_storage_dir: str,
            **kwargs
    ) -> GenerateTaxonomyStatistics:
        """
        Classmethod that creates the dataset from sparql, this is helpfull when simply calculating intermediate statistics
        on datasets to track progression of the labeling process.

        :param config: the general config that is used throughout the project
        :param logger: logger object for logging
        :param request_handler: the instantiated request handler to use
        :param taxonomy_uri: the taxonomy uri
        :param max_level: max depth specified as int
        :param local_storage_dir: local caching/ artifact trakcing dir
        :param kwargs:
        :return:
        """
        dataset_builder = DatasetBuilder.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            taxonomy_uri=taxonomy_uri,
            query_type=DecisionQuery.ANNOTATED,
            do_train_test_split=False,
            **kwargs
        )

        return cls(
            config=config,
            logger=logger,
            dataset=dataset_builder.train_dataset,
            taxonomy=dataset_builder.taxonomy,
            max_level=int(max_level),
            local_storage_dir=local_storage_dir
        )

    @classmethod
    def from_checkpoint(
            cls,
            config: Config,
            logger: Logger,
            checkpoint_folder: str,
            max_level: int,
            local_storage_dir: str

    ) -> GenerateTaxonomyStatistics:
        """
        Classmethod to instantiate taxonomy statistics class from a dataset checkpoint.

        :param config: the general config used throughout the project
        :param logger: the logger object
        :param checkpoint_folder: checkpoint location where we can load the dataset from
        :param max_level: maximum depth defined as integer
        :param local_storage_dir: local storage dir for caching/ mlflow artifacts
        :return:
        """

        dataset_builder = DatasetBuilder.from_checkpoint(
            config=config,
            logger=logger,
            checkpoint_folder=checkpoint_folder,
        )

        return cls(
            config=config,
            logger=logger,
            dataset=dataset_builder.train_dataset,
            taxonomy=dataset_builder.taxonomy,
            max_level=int(max_level),
            local_storage_dir=local_storage_dir
        )
