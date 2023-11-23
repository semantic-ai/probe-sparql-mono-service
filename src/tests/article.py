import unittest

from ..data_models import Article


class ArticleTests(unittest.TestCase):

    def setUp(self) -> None:
        from ..sparql import RequestHandler
        from ..config import Config
        from ..utils import LoggingBase

        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        self.article = Article(
            config=self.config.data_models,
            logger=self.logger,
            uri="article_uri_here",
            number=1,
            content="Article1 content"
        )

    def test_formatted_article(self):
        self.assertEqual(self.article.formatted_article, f"1: Article1 content")

    def test_uri(self):
        self.assertEqual(self.article.uri, "article_uri_here")