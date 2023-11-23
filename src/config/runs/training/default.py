from .constant import CONFIG_PREFIX
from ...base import Settings


class DefaultConfig(Settings):
    bert_model_id: str = "GroNLP/bert-base-dutch-cased"
    distil_bert_model_id: str = "Geotrend/distilbert-base-nl-cased"
    setfit_model_id: str = "sentence-transformers/paraphrase-mpnet-base-v2"

    keep_negative_examples: bool = False

    class Config():
        env_prefix = f"{CONFIG_PREFIX}default_"
