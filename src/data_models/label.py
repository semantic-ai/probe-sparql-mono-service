from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger

from ..utils import wrap, entering, exiting
from ..errors import CustomValueError
from .base import Base


class Label(Base):
    """
    This class is used for parsing a Label from the structured input that is received from the sparql endpoint, it is parsed
    into this python object, that enables extended functionality.

    Using the config, it is possible to extend this class with custom loading/saving behaviour.

    Typical usage example:
        >>> label = Label(
                config=DataModelConfig(),
                logger=logging.logger,
                taxonomy_node_uri="...",
                score=1.0,
                uri="..."
            )
    """

    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            taxonomy_node_uri: str,
            score: float = 1.0,
            uri: str = None
    ) -> None:
        super().__init__(
            config=config,
            logger=logger
        )

        self._taxonomy_node_uri = None
        self._score = None
        self._uri = None

        self.uri = uri or self.generate_uri(self.config.label.uri_base)
        self.taxonomy_node_uri = taxonomy_node_uri
        self.score = score


    @property
    def taxonomy_node_uri(self) -> str:
        """
        This property is used for setting and getting the taxonomy_node_uri value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> article = Label(...)
            >>> article.taxonomy_node_uri = "..."
            >>> uri = article.taxonomy_node_uri

        :return: The string taxonomy_node_uri
        """
        return self._taxonomy_node_uri

    @taxonomy_node_uri.setter
    def taxonomy_node_uri(self, value):
        if isinstance(value, str):
            self._taxonomy_node_uri = value
        elif value is None:
            self._taxonomy_node_uri = None
        else:
            try:
                self._taxonomy_node_uri = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Label.taxonomy_node_uri",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def score(self) -> float:
        """
        This property is used for setting and getting the score value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> article = Label(...)
            >>> article.taxonomy_node_uri = "..."
            >>> uri = article.taxonomy_node_uri

        :return: The string taxonomy_node_uri
        """
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        if isinstance(value, float):
            self._score = value
        elif value is None or value == "":
            self._score = None
        else:
            try:
                self._score = float(value)
            except Exception as ex:
                raise ex

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            uri: str
    ) -> Label:
        """
        Class method for class initialization from sparql.
        When provided with an uri from a Label, it will automatically execute all related queries
        to populate the object with all necessary information.

        Example usage:
            >>> annotation = Label.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = ...,
                    annotation_uri = "..."
                )

        :param config: the general DataModelConfig
        :param logger: logger object that can be used for logs
        :param request_handler: the request wrapper used for sparql requests
        :param uri: the uri which is used to find all relevant information
        :return: an instance of the Annotation Class
        """
        query = config.label.query.format(uri=uri, score_relation=config.label.score_relation,
                                          taxonomy_relation=config.label.taxonomy_relation)
        logger.debug(f"Label Query: ```{query}```")
        response = request_handler.post2json(query)

        return cls(config, logger=logger, uri=uri, taxonomy_node_uri=response[0]['taxonomy_node_uri'],
                   score=response[0]['score'])

    @property
    @wrap(entering, exiting)
    def subquery(self) -> str:
        """
        Property (getter only) to retrieve the subquery for the Label object.
        The sub queries are generally used for creation of insert statements, it checks for the user/model annotation status
        and calls the specific submodules in order to create the complete label statement.

        Example usage:
            >>> article = Label(...)
            >>> labels = article.subquery

        :return: The formatted subquery as string
        """

        query = self.config.label.sub_query.format(
            uri=self._ensure_encapsulation(self.uri),
            taxonomy_node_uri=self._ensure_encapsulation(self.taxonomy_node_uri),
            score=self.score
        )

        self.logger.debug(f"Subquery for labels: {query}")

        return query
