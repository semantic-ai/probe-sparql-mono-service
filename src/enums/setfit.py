from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

import enum


class SetfitClassifierHeads(str, enum.Enum):
    """
    This enum is used to identify what type of setfit head you want to use
    """
    # zeroshot
    SKLEARN_ONE_VS_REST: str = "sklearn_one-vs-rest"  # OneVsRestClassifier
    SKLEARN_MULTI_OUTPUT: str = "sklearn_multi-output"  # MultiOutputClassifier
    SKLEARN_CLASSIFIER_CHAIN: str = "sklearn_classifier-chain"  # ClassifierChain

    DIFFERENTIABLE_ONE_VS_REST: str = "differentiable_one-vs-rest"  # SetFitHead
    DIFFERENTIABLE_MULTI_OUTPUT: str = "differentiable_multi-output"  # SetFitHead


    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @staticmethod
    def get_prefixed_heads(prefix: str) -> list[str]:
        """
        This function filters the setfit heads on values and returns the resulting heads

        :param prefix: string to check if in head value
        :return: filtered output as list
        """
        return [v for v in SetfitClassifierHeads.list() if prefix in v]
    @staticmethod
    def match(config: Config, value: SetfitClassifierHeads):
        """
        this function allows us to verify the provided input and return the correct query

        :param config: the global configuration object
        :param value: the enum value that is used.
        :return: format-able query in string format
        """
        match value:

            case SetfitClassifierHeads.SKLEARN_ONE_VS_REST.value | SetfitClassifierHeads.SKLEARN_ONE_VS_REST:
                return dict(
                    use_differentiable_head=False,
                    multi_target_strategy="one-vs-rest"
                )

            case SetfitClassifierHeads.SKLEARN_MULTI_OUTPUT.value | SetfitClassifierHeads.SKLEARN_MULTI_OUTPUT:
                return dict(
                    use_differentiable_head=False,
                    multi_target_strategy="multi-output"
                )

            case SetfitClassifierHeads.SKLEARN_CLASSIFIER_CHAIN.value | SetfitClassifierHeads.SKLEARN_CLASSIFIER_CHAIN:
                return dict(
                    use_differentiable_head=False,
                    multi_target_strategy="classifier-chain"
                )

            case SetfitClassifierHeads.DIFFERENTIABLE_ONE_VS_REST.value | SetfitClassifierHeads.DIFFERENTIABLE_ONE_VS_REST:
                return dict(
                    use_differentiable_head=True,
                    multi_target_strategy="one-vs-rest"
                )

            case SetfitClassifierHeads.DIFFERENTIABLE_MULTI_OUTPUT.value | SetfitClassifierHeads.DIFFERENTIABLE_MULTI_OUTPUT:
                return dict(
                    use_differentiable_head=True,
                    multi_target_strategy="multi-output"
                )
