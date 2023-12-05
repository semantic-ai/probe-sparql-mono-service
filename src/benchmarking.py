from .benchmark import *
from .dataset import DatasetBuilder
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .enums import ModelType, DatasetType, DecisionQuery

from fire import Fire
from uuid import uuid4
import mlflow


def main(
        model_types: list[ModelType] | str,
        dataset_types: list[DatasetType] | str,
        model_ids: list[str] | str,
        taxonomy_uri: str = "http://stad.gent/id/concepts/gent_words",
        checkpoint_location: str = None
):
    """
    This main function contains the script for the execution of benchmarking with customizable inputs.
    During benchmarking all the permutations for the variables below will be executed, results will be accumulated in
    the mlflow interface under the pre-specified mlflow experiment id.

    [sidenotes]
        When executing the benchmarking on supervised models, keep in mind that the models are not nessecearly trained
        on the selected taxonomy. They will output the taxonomy they have been trained on (this however is specified in
        the naming convention of the model.)

    Example usage:

        When you specify the model/datasets instead of using the prefix it should look like this:
            >>> main(
                    model_types=[
                        "embedding_child_labels",
                         "embedding_chunked",
                         "embedding_sentence"
                    ],
                    dataset_types=[
                        "m1_article_split",
                         "m1_shorttitle",
                         "m1_general"
                     ],
                    model_ids=[
                        "paraphrase-multilingual-mpnet-base-v2",
                        "intfloat/multilingual-e5-small",
                        "thenlper/gte-large",
                        "multi-qa-mpnet-base-dot-v1"
                    ]
                    taxonomy_uri="http://stad.gent/id/concepts/gent_words"
                )

        When you use the prefix method, it should look like this:
             >>> main(
                    model_types="embedding",
                    dataset_types="m1",
                    model_ids=[
                        "paraphrase-multilingual-mpnet-base-v2",
                        "intfloat/multilingual-e5-small",
                        "thenlper/gte-large",
                        "multi-qa-mpnet-base-dot-v1"
                    ]
                    taxonomy_uri="http://stad.gent/id/concepts/gent_words"
                )


    Run command:
        When running specific model types and dataset types.
            >>> python -m src.benchmarking --model_types="embedding_chunked, embedding_sentence" --dataset_types="m1_article_split,m1_general" --model_ids="paraphrase-multilingual-mpnet-base-v2,intfloat/multilingual-e5-small,multi-qa-mpnet-base-dot-v1,thenlper/gte-large" --taxonomy_uri="http://stad.gent/id/concepts/gent_words"

        When running one prefix dataset and one prefix model type
            >>> python -m src.benchmarking --model_types="embedding" --dataset_types="m1" --model_ids="paraphrase-multilingual-mpnet-base-v2,intfloat/multilingual-e5-small,multi-qa-mpnet-base-dot-v1,thenlper/gte-large" --taxonomy_uri="http://stad.gent/id/concepts/gent_words"



    :param checkpoint_location:
        The location where a dataset checkpoint can be found, this location should contain content from the create_checkpoint
        function as implemented in the dataset builder class. If it does not provided the same input, it wont load propperly
        or result in an error.
    :param model_types:
        The model type parameter is used to specify what model type(s) you are using for your benchmarking.
        This can be provided in two separate ways:

            1.  An actual list with values that can be found in the ModelType enum (most likely passed as string values).
                Keep in mind that the values in this list need te be matching with the enum values exactly!
                This list should be provided with comma's to separate the values (see run command example).

            2.  You can provide a string prefix, this prefix will be used to identify all relevant models.
                These models will automaticly be added to the list of model types to experiment with.

    :param dataset_types:
        The dataset type parameter is used to specify what type of dataset(s) you want to use for the benchmarking run.
        The dataset type can be provided in two separate ways:

            1.  An actual list with values that can be found in the DatasetType enum (will be passed as string values),
                this list should be provided with comma's for separation (see run command example).

            2.  String prefix value that will be matched to the current enum values. all matching enum values will be used
                for the benchmarking run.

    :param model_ids:
        This parameter requires a list of models to run the benchmarking with, this will be huggingface model references.
        Keep in mind that models from zeroshot might work for embeddings. however the other way around they won't.
        It is advised not to mix zeroshot, benchmarking and classifier models when running experiments, it would be better
        (and more funtional) to split these up in multiple benchmarking runs.

    :param taxonomy_uri:
        The string reference to what taxonomy you want to use for the benchmarking process.


    :return: Nothing
    """

    # defining global vars
    config = Config()
    logger = LoggingBase(
        config=config.logging
    ).logger
    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    # process input variables
    new_model_types = []
    for model_type in model_types:
        new_model_types += ModelType.get_from_prefix(model_type)
    model_types = list(set(new_model_types))

    new_dataset_types = []
    for dataset_type in dataset_types:
        new_dataset_types += DatasetType.get_from_prefix(dataset_type)
    dataset_types = list(set(new_dataset_types))

    if checkpoint_location is None:
        # creating dataset checkpoint
        dataset_builder = DatasetBuilder.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            taxonomy_uri=taxonomy_uri,
            query_type=DecisionQuery.ANNOTATED
        )

        # creating checkpoint from pulled dataset
        checkpoint_location = dataset_builder.create_checkpoint("/tmp/data_checkpoint")

    with mlflow.start_run(run_name=f"benchmarking_{uuid4().hex}"):
        # creating all permutations
        for dataset_type in dataset_types:
            config.run.dataset.type = dataset_type

            for model_type in model_types:
                config.run.model.type = model_type

                try:
                    match model_type.split("_")[0]:

                        case "embedding":
                            benchmark = EmbeddingSimilarityBenchmark(
                                config=config,
                                logger=logger,
                                request_handler=request_handler,
                                model_ids=model_ids,
                                taxonomy_reference=taxonomy_uri,
                                nested_mlflow_run=True,
                                checkpoint_dir=checkpoint_location
                            )
                        case "huggingface":
                            benchmark = HuggingfaceBenchmark(
                                config=config,
                                logger=logger,
                                request_handler=request_handler,
                                model_ids=model_ids,
                                taxonomy_reference=taxonomy_uri,
                                nested_mlflow_run=True,
                                checkpoint_dir=checkpoint_location
                            )
                        case "zeroshot":
                            benchmark = ZeroshotBenchmark(
                                config=config,
                                logger=logger,
                                request_handler=request_handler,
                                model_ids=model_ids,
                                taxonomy_reference=taxonomy_uri,
                                nested_mlflow_run=True,
                                checkpoint_dir=checkpoint_location
                            )
                        case "hybrid":
                            raise NotImplementedError(
                                "Hybrid models currently is not implemented (does not make sense)")
                            # HybridBenchmark(
                            #     config=config,
                            #     logger=logger,
                            #     request_handler=request_handler,
                            #     #
                            #     # fix the rest here
                            #     #
                            #     taxonomy_reference=taxonomy_uri,
                            #     nested_mlflow_run=True,
                            #     checkpoint_dir=checkpoint_location
                            # )
                        case _:
                            raise Exception("Model not in model type list")

                    benchmark()

                except Exception as ex:
                    logger.error(
                        f"Benchmarking failed with current configuration model_type:{model_type}, dataset_type: {dataset_type}")
                    logger.error(f"Exception that was caught: {ex}")
                    raise ex


if __name__ == "__main__":
    Fire(main)
