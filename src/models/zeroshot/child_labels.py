from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...data_models import Taxonomy


from ...data_models import Taxonomy
from .base import ZeroshotModel
import torch


class ChildLabelsZeroshotModel(ZeroshotModel):
    """
    Zeroshot approach that implements the label merging of all child labels
    """

    def _text_formatting(self, taxonomy_node: Taxonomy) -> str:
        """
        Custom text formatting logic

        :param taxonomy_node: taxonomy node to format text for
        :return:
        """
        parent_label = taxonomy_node.label
        sub_labels = [label for label in taxonomy_node.all_linked_labels if label != parent_label]

        custom_text = f"Tekst over {parent_label} of meer specifiek {' of'.join(sub_labels)}"
        self.logger.debug(f"Created custom text: {custom_text}")

        return custom_text

    def _prep_labels(self, taxonomy: Taxonomy) -> None:

        if not isinstance(taxonomy, list):
            self.labels = taxonomy.get_labels(max_depth=1)

            self.build_labels = [
                self._text_formatting(taxonomy_node=parent_node)
                for parent_node in taxonomy.children
            ]
        else:
            self.labels = taxonomy

    _prep_labels.__doc__ = ZeroshotModel._prep_labels.__doc__

    @torch.inference_mode()
    def classify(self, text: str, multi_label: bool, **kwargs) -> dict[str, float]:
        # custom adaptation required here, longer label input returns max length errors in
        # huggingface pipeline implementation...
        scores = []

        labels = kwargs.get("labels", self.build_labels)

        for label in labels:
            inputs = self.tokenizer.encode(
                text,
                label,
                return_tensors='pt',
                truncation=True,
                truncation_strategy='only_first'
            )

            logits = self.model(inputs.to(self.device))[0]
            contradiction_logits = logits[:, [0, 2]]
            probs = contradiction_logits.softmax(dim=1)
            scores.append(probs[:, 1])
        #
        # result = self.pipe(
        #     text,
        #     self.labels,
        #     multi_label=multi_label
        # )

        return {k: v for k, v in zip(kwargs.get("labels", self.labels), scores)}

    classify.__doc__ = ZeroshotModel._prep_labels.__doc__
