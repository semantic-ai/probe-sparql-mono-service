from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from logging import Logger

from ..data_models import Taxonomy
from transformers import AutoModel, AutoTokenizer
import torch


class Model:
    """
    Base model for all classes
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model_id: str
    ) -> None:
        self.config = config
        self.logger = logger
        self.model_id = model_id

    def _load_model(self, model_id: str) -> None:
        """
        This function enables custom model preperations before executing the classification

        :param model_id: model_id to pull
        :return:
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def _prep_labels(self, taxonomy: Taxonomy | list[str]) -> None:
        """
        The function that prepares the labels, this converts them to the required format for further processing with a model.
        :param taxonomy: Taxonomy object where we will use the labels from
        :return:
        """
        if not isinstance(taxonomy, list):
            self.labels = taxonomy.get_labels(max_depth=1)
        else:
            print("labels", self.labels)
            self.labels = taxonomy

    def add_labels(self, labels: list[str]) -> None:
        """
        This function enables the adding of extra labels to the models setup

        :param labels: list of new labels to add/ set in place
        :return: nothing
        """
        self._prep_labels(labels)

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:
        """
        Abstract function that executes the text classificatoin

        :param text: the text to classify
        :param multi_label: boolean to identify if it is a multilabel problem
        :param kwargs: potential extra vars
        :return: the results
        """
        raise NotImplementedError()

    @property
    def device(self):
        """
        This property returns the device that the model is running on.

        :return: torch device in use
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
