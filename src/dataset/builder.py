from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

    from logging import Logger
    from typing import Any

from ..sparql import RequestHandler
from ..enums import EndpointType, DecisionQuery
from ..data_models import Taxonomy

from tqdm import tqdm
from uuid import uuid4
import json, os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def binarize(taxonomy, labels) -> list[int]:
    """
    This function maps labels to a multilabel binazired input.

    :return: list of 0 or 1 values based on provided labels
    """
    _tmp = {label: 0 for label in taxonomy.get_labels(max_depth=1)}
    for k, v in _tmp.items():
        if k in labels:
            _tmp[k] = 1

    return list(_tmp.values())


class DatasetBuilder:
    """
    The builder class is mainly used to control the creation/loading of datasets.
    During the creation of a new dataset, it is possible to tweak behaviour by setting specific values in the config.
    You have control over the usage of the predefined train-test split, split size, ... more info can be found in the config module.

    In general there are two main approaches to interface/load datasets:

        1. Loading dataset from sparql
            >>> dataset_builder = DatasetBuilder.from_sparql(...)
            >>> train_dataset = dataset_builder.train_dataset

            more related information can be found at the classmethod from_sparql


        2. Loading dataset from local checkpoint
            >>> dataset_builder = DatasetBuilder.from_checkpoint(...)
            >>> train_dataset = dataset_builder.train_dataset

            more related information can be found at the classmethod from_checkpoint
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            train_dataset: list[dict[str, str]],
            test_dataset: list[dict[str, str]],
            taxonomy: Taxonomy
    ):

        self.config = config
        self.logger = logger

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.taxonomy = taxonomy

    def _dump_json(self, file_path: str, dictionary: dict | list[dict[Any, Any]]) -> None:
        """
        This function dumps the content from the provided dictionary to the provided filepath.

        :param file_path: path where to dump the json
        :param dictionary: dictionary that will be saved to json file
        :return: Nothing at al
        """
        self.logger.debug(f"Dumping content to file: {file_path}")
        with open(file_path, "w+") as f:
            json.dump(dictionary, f)

    def create_checkpoint(self, checkpoint_folder: str) -> str:
        """
        This function provides functionality to save all relevant information (train- and test dataset + taxonomy).
        These checkpoints are a 1-on-1 match for loading when using the from_checkpoint classmethod

        Example usage:
            >>> dataset_builder = DatasetBuilder(...)
            >>> dataset_builder.create_checkpoint(checkpoint_folder="...")

        :param checkpoint_folder: folder to save checkpoint to
        :return: returns the unique checkpoint subfolder where the artifacts were saved to
        """

        checkpoint_sub_folder = os.path.join(checkpoint_folder, uuid4().hex)
        os.makedirs(checkpoint_sub_folder)

        train_dataset_path = os.path.join(checkpoint_sub_folder, "train_dataset.json")
        test_dataset_path = os.path.join(checkpoint_sub_folder, "test_dataset.json")
        taxonomy_path = os.path.join(checkpoint_sub_folder, "taxonomy.json")

        self._dump_json(train_dataset_path, self.train_dataset)
        self._dump_json(test_dataset_path, self.test_dataset)
        self._dump_json(taxonomy_path, self.taxonomy.todict(with_children=True))

        return checkpoint_sub_folder

    @classmethod
    def from_sparql(
            cls,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            taxonomy_uri: str,
            query_type: DecisionQuery,
            do_train_test_split: bool = True,
            **kwargs
    ):
        """
        Class method for class initialization from sparql.
        When provided with a taxonomy uri, it will create a new dataset based on annotated decisions that can be
        found in the sparql database.

        Example usage:
            >>> annotation = DatasetBuilder.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = RequestHandler(...),
                    taxonomy_uri = "...",
                    query_type = DecisionQuery.ANNOTATED
                )

        :param config: the general DataModelConfig
        :param logger: logger object that can be used for logs
        :param request_handler: the request wrapper used for sparql requests
        :param query_type: What type of query will be executed
        :param taxonomy_uri: what taxonomy to pull when using annotated dataset query_type
        :param do_train_test_split: wheter or not to execute the train test split (not via config, check code for clarity)
        :return: an instance of the DatasetBuilder Class
        """

        # import this way because of circular imports...
        from ..data_models import Decision, TaxonomyTypes

        # Loading dataset
        dataset = []

        decision_query = DecisionQuery.match(config.data_models, query_type).format(
            taxonomy_uri=taxonomy_uri
        )
        logger.debug(f"Decision query: {decision_query}")

        # filter only for most recent annotations (done by date)
        _memory = dict()

        decision_response = request_handler.post2json(decision_query)

        if limit := kwargs.get("limit", False):
            decision_response = decision_response[:limit]

        for response in decision_response:
            decision_uri = response.get("_besluit")

            # adding annotation uri to use while pulling the data.
            # overwrite with most recent annotation (if 2 of same data take latest)
            value = int(response.get("date", 0))
            if _memory.get(decision_uri, [0])[0] <= value:
                _memory[decision_uri] = [value, response.get("anno", None)]

        # create cleaned list of decisions and pull all relevant information
        annotated_decisions = [
            dict(
                decision=k,
                date=v[0],
                annotation=v[1]
            ) for k, v in _memory.items()
        ]

        logger.info(f"Creating dataset from sparql")
        for response in tqdm(annotated_decisions, desc="pulling data from endpoint"):
            decision = Decision.from_sparql(
                config=config.data_models,
                logger=logger,
                decision_uri=response.get("decision"),
                request_handler=request_handler,
                annotation_uri=response.get("annotation", None)
            )

            dataset.append(decision.train_record)

        taxonomy = TaxonomyTypes.from_sparql(
            config=config.data_models,
            logger=logger,
            request_handler=request_handler,
            endpoint=EndpointType.TAXONOMY
        )

        selected_taxonomy = taxonomy.get(taxonomy_uri)
        df = pd.DataFrame(dataset)
        assert len(df) != 0, "Length of dataset is 0"
        logger.debug(f"dataframe: {df}")

        if do_train_test_split:

            # create top labels
            df["top_labels"] = df.labels.apply(
                lambda x: list(set([selected_taxonomy.find(label).get(1, {}).get("label") for label in x]))
            )

            train_dataset = []
            test_dataset = []

            if config.run.dataset.use_predefined_split:

                for record in dataset:
                    if record.get("uri") in config.run.dataset.predefined_uris:
                        test_dataset.append(record)
                    else:
                        train_dataset.append(record)

                logger.debug(f"Train examples: {len(train_dataset)}, Test examples: {len(test_dataset)}")

                y_test = [
                    binarize(
                        taxonomy=selected_taxonomy,
                        labels=label
                    ) for label in df.top_labels.tolist()
                ]

                logger.info(f"Current class split count: {np.asarray(y_test).sum(axis=0)}")

            else:
                X = np.asarray(df.uri.tolist())
                y = np.asarray([
                    binarize(
                        taxonomy=selected_taxonomy,
                        labels=label
                    ) for label in df.top_labels.tolist()
                ])

                # specifically splitting on uri -> this way we can select original dataset results later
                _, test_uris, _, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.run.dataset.train_test_split,
                    random_state=42
                )

                for record in dataset:
                    if record.get("uri") in test_uris.tolist():
                        test_dataset.append(record)
                    else:
                        train_dataset.append(record)

                logger.info(f"Current class split count: {y_test.sum(axis=0)}")
        else:
            train_dataset, test_dataset = dataset, None

        logger.info(f"Training records: {len(train_dataset)} test records: {len(test_dataset) if test_dataset else 0}")

        selected_taxonomy.label = taxonomy_uri.split("/")[-1]
        selected_taxonomy.uri = taxonomy_uri

        return cls(
            config=config,
            logger=logger,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            taxonomy=selected_taxonomy
        )

    @classmethod
    def from_checkpoint(
            cls,
            config: Config,
            logger: Logger,
            checkpoint_folder: str
    ):
        """
        Classmethod to create an instance of the databuilder class from a checkpoint.
        This checkpoint is based on the checkpoints that are created using the 'create_checkpoint' method.



        :param config: general config provided
        :param logger: logging object that is used throughout the project
        :param checkpoint_folder:  folder where to save everything to
        :return: an instance of the DatasetBuilder object
        """

        # Loading dataset
        with open(os.path.join(checkpoint_folder, "train_dataset.json"), "r") as f:
            train_dataset = json.load(f)

        with open(os.path.join(checkpoint_folder, "test_dataset.json"), "r") as f:
            test_dataset = json.load(f)

        # loading taxonomy
        taxonomy = Taxonomy.from_checkpoint(
            config=config.data_models,
            logger=logger,
            checkpoint_folder=
            os.path.join(checkpoint_folder)
        )

        return cls(
            config=config,
            logger=logger,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            taxonomy=taxonomy
        )
