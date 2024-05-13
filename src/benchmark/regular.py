from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger
    from ..sparql import RequestHandler

from ..dataset import DatasetBuilder, create_dataset
from ..enums import DecisionQuery
from ..models import Model, get_model
from .evaluate import MultilabelEvaluation
from .base import BenchmarkBase


class BenchmarkWrapper(BenchmarkBase):
    """
    This is the baseclass for benchmarking models.

    The main objective of this class is brining multiple components togheter, these components all contain their own
    custom behaviour for a certain part of the code.

    These sub-components are:
        *   Dataset: How do we transform the data from sparql data into training/inference data?
        *   Model architecture (or type): A wrapper object around a model to allow abstract usage of methods that are universally implmented.
        *   Model base: Model weights to load in the predefined model Architecture
        *   Taxonomy: A specific taxonomy to use for predictions

    The previously mentioned components are mostly declared in the config, only the model base is provided under the
    model_ids parameter.

    Typical usage:
        >>> benchmark = BenchmarkWrapper(
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
            taxonomy_reference: str,
            checkpoint_dir: str = "data",
            nested_mlflow_run: bool = False
    ) -> None:

        self.config = config
        self.logger = logger
        self.request_handler = request_handler

        self.model_ids = [model_ids] if isinstance(model_ids, str) else model_ids
        self.taxonomy_reference = taxonomy_reference

        self._create_dataset(checkpoint=checkpoint_dir)

        self.nested_mlflow_run = nested_mlflow_run
        self._default_mlflow_tags = dict()
        self._default_description = str()

    def _create_dataset(self, checkpoint: str | None) -> None:
        """
        Internal function that is responsible for the creation of the benchmarking dataset.
        When it is not provided with a checkpoint, it will automatically start building the dataset by pulling all
        annotated information for the provided taxonomy.


        :param checkpoint: folder or path where the benchmark data can be found
        :return: Nothing
        """

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
            dataset=dataset_builder.train_dataset,
            taxonomy=taxonomy
        )

    def _create_model(self, model_id: str) -> Model:
        return get_model(
            config=self.config,
            logger=self.logger,
            model_id=model_id,
            taxonomy=self.train_ds.taxonomy
        )

    def _create_run_name(self, model_id: str | None = None) -> str:
        """
        Internal function that generates custom run names, these are used for verbose naming in the mlflow
        tracking.

        :param model_id: the model id of the currently selected model
        :return: a custom string that represents the unique combination of components
        """
        if model_id is None:
            return f"{self.config.run.model.type}_{self.config.run.dataset.type}"
        else:
            return f"{self.config.run.model.type}_{self.config.run.dataset.type}_{model_id.split('/')[-1]}"

    def exec(self):
        """
        This function is responsible for the execution of the benchmark.
        It creates (nested-)mlflow runs, these runs are based on the pre-defined config and selected base model.

        For each combination, a (nested) run will appear in the mlflow interface containing all the artifacts created by
        a benchmark run. (more info about the artifacts can be found in the evaluate class)

        Example usage:
            >>> benchmark = BenchmarkWrapper(...)
            >>> benchmark.exec()

        :return: Nothing at all
        """
        with mlflow.start_run(
                tags=self.default_mlflow_tags,
                description=self.default_description,
                nested=self.nested_mlflow_run,
                run_name=self._create_run_name()
        ):

            mlflow.set_tag(
                key="dataset_type",
                value=self.config.run.dataset.type
            )

            for model_id in self.model_ids:
                self.logger.debug(f"model_id in list of models {model_id}")
                with mlflow.start_run(
                        nested=True,
                        tags={"model_id": model_id},
                        description=f"Evaluate performance for: {model_id}",
                        run_name=self._create_run_name(model_id=model_id)
                ):
                    try:
                        _model = self._create_model(model_id=model_id)
                    except Exception as ex:
                        traceback.print_exception(ex)
                        self.logger.error(f"The following error occured during initalization of the model {ex}")


                    try:
                        MultilabelEvaluation(
                            config=self.config,
                            logger=self.logger,
                            dataset=self.train_ds,
                            model=_model
                        ).evaluate()
                    except Exception as ex:
                        self.logger.warning(f"Benchmark failed for {model_id} with error: \n {traceback.format_exc()}")

            mlflow.log_dict(
                dictionary=self.config.to_dict(),
                artifact_file="config.json"
            )

    def __call__(self):
        """
        The call function references the exec funciton, for more information check those docs

        Example usage:
            >>> benchmark = BenchmarkWrapper(...)
            >>> benchmark()

        :return: Nothing at all
        """
        self.exec()

    @property
    def default_mlflow_tags(self) -> dict[str, str]:
        """
        This property provides a getter for the default mlflow tags that are provided by the selection of the class

        Example usage:
            >>> benchmark = BenchmarkWrapper(...)
            >>> mlflow_tags = benchmark.default_mlflow_tags

        :return: tags for mlflow
        """
        return self._default_mlflow_tags

    @property
    def default_description(self) -> str:
        """
        This property provides a getter for the default description that should be provided for mlflow logging

        Example usage:
            >>> benchmark = BenchmarkWrapper(...)
            >>> description = benchmark.default_description

        :return: string description for mlflow run
        """
        return self._default_description
