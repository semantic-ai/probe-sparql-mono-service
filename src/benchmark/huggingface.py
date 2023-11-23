from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..sparql import RequestHandler

from .regular import BenchmarkWrapper


from ..dataset import DatasetBuilder, create_dataset
from ..enums import DecisionQuery


class HuggingfaceBenchmark(BenchmarkWrapper):

    def __init__(
            self,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            model_ids: list[str] | str,
            taxonomy_reference: str = "http://stad.gent/id/concepts/gent_words",
            nested_mlflow_run: bool = False,
            checkpoint_dir: str = "data"
    ):

        super().__init__(
            config=config,
            logger=logger,
            request_handler=request_handler,
            model_ids=model_ids,
            taxonomy_reference=taxonomy_reference,
            nested_mlflow_run=nested_mlflow_run,
            checkpoint_dir=checkpoint_dir
        )

        self._default_mlflow_tags = {"model_type": "Huggingface"}
        self._default_description = "Running for huggingface pre-trained models"

    def _create_dataset(self, checkpoint: str | None) -> None:

        if checkpoint is None:
            dataset_builder = DatasetBuilder.from_sparql(
                config=self.config,
                logger=self.logger,
                request_handler=self.request_handler,
                taxonomy_uri=self.taxonomy_reference,
                query_type=DecisionQuery.ANNOTATED
            )

        else:
            dataset_builder = DatasetBuilder.from_checkpoint(
                config=self.config,
                logger=self.logger,
                checkpoint_folder=checkpoint
            )

        taxonomy = dataset_builder.taxonomy
        self.train_ds = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=dataset_builder.test_dataset,
            taxonomy=taxonomy
        )


    @property
    def default_mlflow_tags(self):
        return self._default_mlflow_tags

    @property
    def default_description(self):
        return self._default_description
