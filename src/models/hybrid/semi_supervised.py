from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from logging import Logger
    from ...data_models import Taxonomy

from ...enums import TaxonomyFindTypes
import torch

from ..base import Model


class HybridModel(Model):
    """
    The HybridModel class implements a semi-supervised approach.

    It starts with a base model and uses one of the unsupervised flavours for the rest of the predictions in the tree.
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            taxonomy: Taxonomy,
            supervised_model: Model,
            unsupervised_model: Model,

    ) -> None:
        super().__init__(
            config=config,
            logger=logger,
            model_id=""
        )

        self.taxonomy = taxonomy
        self.supervised_model = supervised_model
        self.unsupervised_model = unsupervised_model

        # print(self.supervised_model, self.unsupervised_model)

    def _prep_labels(self, taxonomy: Taxonomy) -> None:
        self.labels = self.taxonomy.get_level_specific_labels(level=2)

    _prep_labels.__doc__ = Model._prep_labels.__doc__

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:

        prediction = self.supervised_model.classify(
            text=text,
            multi_label=multi_label
        )

        children = self.taxonomy.children

        for label, score in prediction.items():
            if score >= self.config.run.benchmark.hybrid.minimum_threshold:

                child_taxo: Taxonomy = [taxo for taxo in children if taxo.label == label][0]
                child_labels: list[str] = [taxo.label for taxo in child_taxo.children]
                self.unsupervised_model.add_labels(child_labels)

                result = {
                    k: round(((v + score) / 2.0), 2)  # weighting the output by the previous level?
                    for k, v in self.unsupervised_model.classify(
                        text=text,
                        multi_label=multi_label,
                        labels=child_labels
                    ).items()
                }

                prediction = {
                    **prediction,
                    **result
                }

            # print(len(prediction.items()))

        return prediction

    classify.__doc__ = Model.classify.__doc__
