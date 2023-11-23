from __future__ import annotations
from typing import NamedTuple

from .base import Settings
from .benchmark import BenchmarkConfig
from .logging import LoggingConfig
from .request import RequestConfig
from .runs import RunConfig

from .data_models import DataModelConfig
from .data_models.sparql import SparqlConfig


class Config(NamedTuple):
    run = RunConfig()
    data_models = DataModelConfig()
    logging = LoggingConfig()
    request = RequestConfig()
    sparql = SparqlConfig()

    def to_dict(self):
        return dict(
            run=self.run.to_dict(),
            data_models=self.data_models.to_dict(),
            logging=self.logging.model_dump(),
            requests=self.request.model_dump(),
            sparql=self.sparql.model_dump()
        )
