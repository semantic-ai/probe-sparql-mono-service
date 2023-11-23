import unittest

from ..config import Config
from ..data_models import Model
from ..sparql import RequestHandler
from ..utils import LoggingBase


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        self.model = Model(
            self.config.data_models,
            logger=self.logger,
            name="test_model_name",
            mlflow_reference="http://mlflow.probe.ml2grow.cloud/models/test_model_name",
            category="zero-shot-model",
            registered_model="test_model_name",
            date=1
        )

        self.dateless_model = Model(
            self.config.data_models,
            logger=self.logger,
            name="test_model_name",
            mlflow_reference="http://mlflow.probe.ml2grow.cloud/models/test_model_name",
            category="zero-shot-model",
            registered_model="test_model_name"
        )

    def test_subquery(self):
        expected_query = "<https://lblod.data.gift/concepts/ml2grow/model/test> ext:creationDate 1 . " \
                         "<https://lblod.data.gift/concepts/ml2grow/model/test> ext:modelName \"test_model_name\". " \
                         "<https://lblod.data.gift/concepts/ml2grow/model/test> ext:modelCategory \"zero-shot-model\" " \
                         ". <https://lblod.data.gift/concepts/ml2grow/model/test> ext:mlflowLink " \
                         "\"http://mlflow.probe.ml2grow.cloud/models/test_model_name\" . " \
                         "<https://lblod.data.gift/concepts/ml2grow/model/test> ext:registeredMlflowModel " \
                         "\"test_model_name\" ."
        formatted_query = " ".join(
            [q.lstrip().rstrip() for q in self.model.subquery.split("\n") if not q.lstrip().rstrip() == ""])

        self.assertEqual(formatted_query, expected_query)

    def test_uri(self):
        self.assertEqual(self.model.uri, "<https://lblod.data.gift/concepts/ml2grow/model/test>")

    def test_date(self):
        self.assertTrue(isinstance(self.model.date, int))
        self.assertEqual(self.model.date, 1)
