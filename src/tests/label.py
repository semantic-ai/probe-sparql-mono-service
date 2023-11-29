import unittest

from ..data_models import Label
from ..sparql import RequestHandler
from ..config import Config
from ..utils import LoggingBase


class LabelTests(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        self.model_label = Label(
            config=self.config.data_models,
            logger=self.logger,
            taxonomy_node_uri="id1",
            score=0.8
        )

        self.user_label = Label(
            config=self.config.data_models,
            logger=self.logger,
            taxonomy_node_uri="id1"
        )

    def test_label_score_init(self):
        self.assertTrue(self.model_label.score == 0.8)
        self.assertTrue(self.user_label.score == 1)

    def test_label_subquery(self):
        expected_query = "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy <id1> . " \
                         "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 1.0."

        self.logger.info(f"expected: {expected_query}")

        formatted_query = " ".join(
            [q.lstrip().rstrip() for q in self.user_label.subquery.split("\n") if not q.lstrip().rstrip() == ""])
        self.logger.info(f"formatted: {formatted_query}")

        self.assertEqual(formatted_query, expected_query)

    # def test_from_sparql(self):
    #     raise NotImplementedError()
