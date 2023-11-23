from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_models import Taxonomy

from transformers import pipeline


class ZeroshotMatching:
    """
    Zeroshot wrapper model
    """

    def __init__(self, model_id: str, taxonomy: Taxonomy, verbose: bool = False):
        self.verbose = verbose
        self.taxonomy = taxonomy
        self.model_id = model_id

        self._init_model()
        self._prep_taxonomy()

    def _prep_taxonomy(self) -> None:
        """
        Formatting input taxonomy

        :return:nothing
        """
        self.candid_labels = [l.prefLabel for l in self.taxonomy.get_labels()]

    def _init_model(self):
        """
        Model initialization
        :return:
        """
        self.model = pipeline("zero-shot-classification", model=self.model_id)

    def match(self, text: str):
        """
        predicting with text input

        :param text: input text to classify
        :return: the classification response
        """
        return self.model(text, self.candid_labels)
