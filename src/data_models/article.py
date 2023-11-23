from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from logging import Logger

from ..utils import wrap, entering, exiting
from ..errors import CustomValueError

from .base import Base


class Article(Base):
    """
    This class is mainly used to parse articles from the structured input formate that is received from
    sparql, into a unstructured text representation.

    The main goal here is to simply create an instance of an 'Article' and enable abstract logic for data processing.
    It would be possible to extend this class in order to load from sparql_uri or write to sparql. This however, is out of scope for the current project.

    Typical usage example:
        >>> article = Article(
            config=DataModelConfig(),
            logger=logging.logger.
            uri="https://data_souce/content/some_article_id/",
            number=1,
            content="Confirmation about the deduction of the ..."
        )
    """

    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            uri: str,
            number: str,
            content: str
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )

        self._uri = None
        self._number = None
        self._content = None

        self.uri = uri
        self.number = number
        self.content = content

    @property
    @wrap(entering, exiting)
    def formatted_article(self) -> str | None:
        """
        This property returns a formatted string for the Article class
        This string is formatted like follows: -> {article_number}: {article_text}

        Example usage:
            >>> article = Article(...)
            >>> formatted_article = article.formatted_article

        :return: The formatted article as string
        """

        if (self.number is None) and (self.content is None):
            raise Exception("Empty article detected")

        return f"{self.number}: {self.content}"

    @property
    def uri(self) -> str:
        """
        This property is used for setting and getting the uri value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> article = Article(...)
            >>> article.uri = "..."
            >>> uri = article.uri

        :return: The string uri as string
        """
        return self._uri

    @uri.setter
    def uri(self, value: str) -> None:
        if isinstance(value, str):
            self._uri = value
        else:
            try:
                self._uri = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Article.uri",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def number(self) -> int:
        """
        This property is used for setting and getting the number value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> article = Article(...)
            >>> article.number = 10
            >>> article_number = article.number

        :return: The article number as int
        """
        return self._number

    @number.setter
    def number(self, value: str) -> None:
        if isinstance(value, str):
            self._number = value
        else:
            try:
                self._number = str(value)
            except Exception as ex:
                error = CustomValueError(
                    property="Article.number",
                    expected_type=str,
                    received_type=type(value)
                )
                self.logger.critical(error.message)
                raise error

    @property
    def content(self) -> str:
        """
        This property is used for setting and getting the content value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> article = Article(...)
            >>> article.content = "..."
            >>> article_content = article.content

        :return: The content as string
        """
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        if isinstance(value, str):
            self._content = value
        else:
            try:
                self._content = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Article.content",
                    expected_type=str,
                    received_type=type(value)
                )
