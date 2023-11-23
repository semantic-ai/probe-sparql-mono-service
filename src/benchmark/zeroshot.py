from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..sparql import RequestHandler

from .regular import BenchmarkWrapper


class ZeroshotBenchmark(BenchmarkWrapper):
    """
    This is the wrapper class for zeroshot models benchmarking, for more information check out the baseclass

        >>> benchmark = ZeroshotBenchmark(
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
            model_ids: list[str] | str,
            taxonomy_reference: str = "http://stad.gent/id/concepts/gent_words",
            checkpoint_dir: str = "data",
            nested_mlflow_run: bool = False
    ) -> None:

        super().__init__(
            config=config,
            logger=logger,
            request_handler=request_handler,
            model_ids=model_ids,
            taxonomy_reference=taxonomy_reference,
            nested_mlflow_run=nested_mlflow_run,
            checkpoint_dir=checkpoint_dir
        )

        self._default_mlflow_tags = {
            "model_type": self.config.run.model.type,
            "supervised_model_id": model_ids
        }
        self._default_description = "Running evaluation over all specified zeroshot models"

    @property
    def default_mlflow_tags(self):
        """
        This property provides a getter for the default mlflow tags that are provided by the selection of the class

        Example usage:
            >>> benchmark = ZeroshotBenchmark(...)
            >>> mlflow_tags = benchmark.default_mlflow_tags

        :return: tags for mlflow
        """
        return self._default_mlflow_tags

    @property
    def default_description(self):
        """
        This property provides a getter for the default description that should be provided for mlflow logging

        Example usage:
            >>> benchmark = ZeroshotBenchmark(...)
            >>> description = benchmark.default_description

        :return: string description for mlflow run
        """
        return self._default_description

