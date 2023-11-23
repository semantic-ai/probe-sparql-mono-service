from __future__ import annotations
from typing import NamedTuple

from .annotation import AnnotationConfig
from .decision import DecisionConfig
from .label import LabelConfig
from .model import ModelConfig
from .sparql import SparqlConfig
from .taxonomy import TaxonomyConfig
from .user import UserConfig


class DataModelConfig(NamedTuple):
    annotation = AnnotationConfig()
    decision = DecisionConfig()
    label = LabelConfig()
    model = ModelConfig()
    sparql = SparqlConfig()
    taxonomy = TaxonomyConfig()
    user = UserConfig()

    def to_dict(self):
        return dict(
            annotation=self.annotation.model_dump(),
            decision=self.decision.model_dump(),
            label=self.label.model_dump(),
            model=self.model.model_dump(),
            sparql=self.sparql.model_dump(),
            taxonomy=self.taxonomy.model_dump(),
            user=self.user.model_dump()
        )
