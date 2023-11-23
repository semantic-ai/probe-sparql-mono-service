from .benchmark import HybridBenchmark
from .config import Config
from .sparql import RequestHandler
from .utils import LoggingBase
from .enums import ModelType, DatasetType

from fire import Fire


def main(
        selection_score: float = None
):
    """
    NOT USED?

    :param selection_score:
    :return:
    """
    config = Config()
    logger = LoggingBase(
        config=config.logging
    ).logger

    request_handler = RequestHandler(
        config=config,
        logger=logger
    )

    config.run.model.type = ModelType.HYBRID_SELECTIVE_MODEL
    config.run.dataset.type = DatasetType.MULTI_SECOND_LEVEL_ALL_BASED

    if selection_score is None:
        selection_score = 0
    config.run.benchmark.hybrid.minimum_threshold = selection_score

    model_ids = [
        "sentence-transformers/paraphrase-mpnet-base-v2"
    ]

    benchmarking = HybridBenchmark(
        config=config,
        logger=logger,
        request_handler=request_handler,
        unsupervised_model_ids=model_ids,
        supervised_model_id="mlflow:/ghent_words_bert_level_1",
        unsupervised_model_type=ModelType.EMBEDDING_REGULAR,
        checkpoint_dir="data/4a939084f1c14b45a4f0a0e45a5c1864"
    )
    benchmarking()


if __name__ == "__main__":
    Fire(main)
