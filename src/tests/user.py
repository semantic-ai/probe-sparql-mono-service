import unittest

from ..config import Config
from ..data_models import User
from ..sparql import RequestHandler
from ..utils import LoggingBase


class UserTests(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        # Existing decision to verify loading
        self.user = User(
            config=self.config.data_models,
            logger=self.logger,
            username="ruben",
            email="ruben@ml2grow.cloud"
        )

    def test_user_uri(self):
        self.assertEqual(self.user.uri, "<https://classifications.ghent.com/ml2grow/user/test>")

    def test_repr(self):
        self.assertEqual(self.user.__repr__(), "<User: ('<https://classifications.ghent.com/ml2grow/user/test>', "
                                               "'ruben')>")
