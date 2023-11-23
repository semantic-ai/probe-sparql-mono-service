from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..sparql import RequestHandler

from .regular import BenchmarkWrapper
from ..models import Model, get_model
from ..enums import ModelType


class HybridBenchmark(BenchmarkWrapper):
    """
    This is the wrapper class for hybrid model benchmarking, for more information check out the baseclass

        >>> benchmark = HybridBenchmark(
                config=Config(),
                logger=logging.logger,
                request_handler=RequestHandler(),
                model_ids=["...", ...],
                taxonomoy_reference="..."
            )
        >>> benchmark()
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            unsupervised_model_ids: list[str] | str,
            supervised_model_id: str,
            unsupervised_model_type: ModelType,
            taxonomy_reference: str = "http://stad.gent/id/concepts/gent_words",
            checkpoint_dir: str = "data",
            nested_mlflow_run: bool = False
    ) -> None:

        super().__init__(
            config=config,
            logger=logger,
            request_handler=request_handler,
            model_ids=unsupervised_model_ids,
            taxonomy_reference=taxonomy_reference,
            nested_mlflow_run=nested_mlflow_run,
            checkpoint_dir=checkpoint_dir
        )

        self._default_mlflow_tags = {"model_type": self.config.run.model.type}
        self._default_description = "Running evaluation over all specified zeroshot models"

        self.supervised_model_id = supervised_model_id
        self.unsupervised_model_type = unsupervised_model_type

    def _create_model(self, model_id: str) -> Model:

        supervised_model = get_model(
            config=self.config,
            logger=self.logger,
            model_id=self.supervised_model_id,
            taxonomy=self.train_ds.taxonomy,
            specific_model_type=ModelType.HUGGINGFACE_MODEL

        )

        unsupervised_model = get_model(
            config=self.config,
            logger=self.logger,
            model_id=model_id,
            taxonomy=self.train_ds.taxonomy,
            specific_model_type=self.unsupervised_model_type,
        )

        return get_model(
            config=self.config,
            logger=self.logger,
            taxonomy=self.train_ds.taxonomy,
            supervised_model=supervised_model,
            unsupervised_model=unsupervised_model,
            model_id=""
        )
    @property
    def default_mlflow_tags(self):
        """
        This property provides a getter for the default mlflow tags that are provided by the selection of the class

        Example usage:
            >>> benchmark = HybridBenchmark(...)
            >>> mlflow_tags = benchmark.default_mlflow_tags

        :return: tags for mlflow
        """
        return self._default_mlflow_tags

    @property
    def default_description(self):
        """
        This property provides a getter for the default description that should be provided for mlflow logging

        Example usage:
            >>> benchmark = HybridBenchmark(...)
            >>> description = benchmark.default_description

        :return: string description for mlflow run
        """
        return self._default_description

