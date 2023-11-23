from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Model


from abc import ABC, abstractmethod


class BenchmarkBase(ABC):

    @abstractmethod
    def _create_dataset(self, checkpoint: str | None) -> None:
        pass

    @abstractmethod
    def _create_model(self, model_id: str) -> Model:
        pass

    @abstractmethod
    def _create_run_name(self, model_id: str | None = None) -> str:
        pass

    @abstractmethod
    def exec(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


    @property
    @abstractmethod
    def default_mlflow_tags(self) -> dict[str, str]:
        pass


    @property
    @abstractmethod
    def default_description(self) -> str:
        pass
