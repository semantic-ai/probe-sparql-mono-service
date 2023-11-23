from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger

from ..utils import wrap, entering, exiting
from ..enums import GraphType
from ..errors import CustomValueError

from .base import Base
from .article import Article
from .annotation import Annotation
from .taxonomy import Taxonomy
from .user import User
from .model import Model
from .label import Label

from datetime import datetime


class Decision(Base):
    """
    This class is used to parse Decisions (and all the submodules) from the structured input that is received from sparql,
    but also create insert statements for a custom defined instances of this object.

    Once the sparql input is parsed in the Decision Object, this class offers extra functionality.

    Typical usage example:
        >>> decision = Decision(
            config=DataModelConfig(),
            logger=logging.logger.
            uri="https://data_souce/content/some_article_id/",
            annotations=[Annotations(...), ...],
            articles=[Article(...), ...],
            ...
        )
    """

    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            uri: str,
            annotations: list[Annotation] | Annotation | None = None,
            articles: list[Article] | Article | None = None,
            uuid: str = None,
            description: str = None,
            short_title: str = None,
            motivation: str = None,
            publication_date: str = None,
            language: str = None,
            points: str = None
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )
        self._uri = None
        self._annotations = []
        self._articles = []
        self._uuid = None
        self._description = None
        self._short_title = None
        self._motivation = None
        self._publication_date = None
        self._language = None
        self._points = None

        self.uri = uri
        self.annotations = annotations
        self.uuid = uuid
        self.description = description
        self.motivation = motivation
        self.publication_date = publication_date
        self.short_title = short_title
        self.language = language
        self.points = points
        self.articles = articles

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            decision_uri: str,
            request_handler: RequestHandler,
            annotation_uri: str = None
    ) -> Decision:
        """
        Class method that creates a decision object from sparql.
        When provided with an uri from a decision, it will automatically execute all related (sub)queries to populate
        the Decision object.

        Example usage:
            >>> annotation = Decision.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = ...,
                    annotation_uri = "..."
                )

        :param annotation_uri: the uri to pull all information for form sparql
        :param config: the general config of the project
        :param logger: object that can be used to generate logs
        :param decision_uri: the string value of the uri that will be used to extract all decision information from
        :param request_handler: the request wrapper for sparql interactions
        :return: an instance of the Decision class
        """

        # Load annotations
        annotations = []

        # when a specified annotation uri is provided only pull that annotation for further processing
        if annotation_uri is not None:
            annotations.append(
                Annotation.from_sparql(
                    config=config,
                    logger=logger,
                    request_handler=request_handler,
                    annotation_uri=annotation_uri
                )
            )
        else:
            full_decision_query = config.decision.create_decision_from_uri.format(
                decision_uri=decision_uri
            )

            logger.debug(f"Decision query: {full_decision_query}")

            full_response = request_handler.post2json(full_decision_query)
            for anno in full_response:
                scores = anno.get("scores", "").split("|")
                taxonomy_node_uris = anno.get("taxonomy_node_uris", "").split("|")
                label_uris = anno.get("label_uris", "")

                labels = [
                    Label(config=config, logger=logger, uri=label_uris, taxonomy_node_uri=uri, score=score)
                    for uri, score in zip(taxonomy_node_uris, scores)
                ]

                model = None
                if model_uri := anno.get("model_uri", None):
                    model = Model(
                        config=config,
                        logger=logger,
                        name=anno.get("model_name", None),
                        mlflow_reference=anno.get("mlflow_link", None),
                        date=int(anno.get("create_data", None)),
                        category=anno.get("category", None),
                        registered_model=anno.get("mlflow_model", None),
                        uri=model_uri
                    )

                user = None
                if user_uri := anno.get("user_uri", None):
                    user = User(
                        config=config,
                        logger=logger,
                        uri=user_uri
                    )

                taxonomy = Taxonomy(
                    config=config,
                    logger=logger,
                    uri=anno.get("taxonomy_uri"),
                )

                annotations.append(
                    Annotation(
                        config=config,
                        logger=logger,
                        taxonomy=taxonomy,
                        user=user,
                        model=model,
                        labels=labels,
                        date=datetime.fromtimestamp(int(anno.get("date"))),
                        uri="",
                    )
                )

        decision_query = config.decision.query_all_content.format(decision_uri=decision_uri)
        logger.debug(f"Query for decision content: {decision_query}")

        if decision_response := request_handler.post2json(decision_query):
            decision_response = decision_response[0]
        else:
            return None

        article_uris = decision_response.get("artikels", "").split("|")
        articles_numbers = decision_response.get("numbers", "").split("|")
        article_values = decision_response.get("waardes", "").split("|")

        articles = [
            Article(
                config=config,
                logger=logger,
                uri=uri,
                number=number,
                content=content
            )

            for uri, number, content in zip(article_uris, articles_numbers, article_values)
            if (len(number) > 2) or (len(content) > 2)
        ]

        return cls(
            config=config,
            logger=logger,
            uri=decision_uri,
            annotations=annotations,
            articles=articles,
            uuid=decision_response.get("uuid", None),
            description=decision_response.get("description", None),
            short_title=decision_response.get("short_title", None),
            motivation=decision_response.get("motivation", None),
            publication_date=decision_response.get("publication_date", None),
            language=decision_response.get("language", None), points=decision_response.get("points", None)
        )

    @property
    @wrap(entering, exiting)
    def annotations(self) -> list[Annotation] | None:
        """
        This property is used to set and retrieve annotation for the specific decision object.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.annotations = [Annotation(...), ...]
            >>> annotations = decision.annotations

        :return: current listed annotations or None if no available annotations
        """
        return self._annotations

    @annotations.setter
    @wrap(entering, exiting)
    def annotations(self, value: list[Annotation] | Annotation) -> None:

        self.logger.debug(f"kannotation value: {value}")
        if not value:
            self.logger.debug("No annotations provided")
            self._annotations = []
        elif isinstance(value, list) and isinstance(value[0], Annotation):
            self._annotations += value
        elif isinstance(value, Annotation):
            self._annotations.append(value)

        else:
            raise CustomValueError(
                property="Decision.annotations",
                expected_type=list[Annotation] | Annotation | None,
                received_type=type(value)
            )

    @property
    def uri(self) -> str:
        """
        This property is used for setting and retrieving the uri for the object.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.uri = "..."
            >>> uri = decision.uri

        :return: The string value for the uri
        """
        return self._uri

    @uri.setter
    def uri(self, value) -> None:
        if value is None:
            self._uri = None
        elif isinstance(value, str):
            self._uri = value
        else:
            try:
                self._uri = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.uri",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def articles(self) -> list[Article] | None:
        """
        This property is used for setting and retrieving the articles that are linked to the decision.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.articles = [Article(...), ...]
            >>> articles = decision.articles

        :return: A list of found Articles
        """
        return self._articles

    @articles.setter
    def articles(self, value: list[Article] | Article | None) -> None:
        if value is None or value == []:
            self._articles = []
        elif isinstance(value, list) and isinstance(value[0], Article):
            self._articles += value
        elif isinstance(value, Article):
            self._articles.append(value)
        else:
            error = CustomValueError(
                property="Decision.articles",
                expected_type=list[Article] | Article,
                received_type=type(value)
            )
            self.logger.critical(error.message)
            raise error

    @property
    def uuid(self) -> str:
        """
        This property is used for setting and getting the uuid value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.uuid = "..."
            >>> uuid = decision.uuid

        :return: The string uuid linked to the decision
        """
        return self._uuid

    @uuid.setter
    def uuid(self, value: str) -> None:
        if isinstance(value, str):
            self._uuid = value
        elif value is None:
            self._uuid = None
        else:
            try:
                self._uuid = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.uuid",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def description(self) -> str:
        """
        This property is used for setting and getting the description value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.description = "..."
            >>> description = decision.description

        :return: The string representation for description
        """
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        if isinstance(value, str):
            self._description = value
        elif value is None:
            self._description = None
        else:
            try:
                self._description = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.description",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def short_title(self) -> str:
        """
        This property is used for setting and getting the short_title value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.short_title = "..."
            >>> short_title = decision.short_title

        :return: The string representation for short_title
        """
        return self._short_title

    @short_title.setter
    def short_title(self, value: str) -> None:
        if isinstance(value, str):
            self._short_title = value
        elif value is None:
            self._short_title = None
        else:
            try:
                self._short_title = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.short_title",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def motivation(self) -> str:
        """
        This property is used for setting and getting the motivation value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.motivation = "..."
            >>> motivation = decision.motivation

        :return: The string representation for motivation
        """
        return self._motivation

    @motivation.setter
    def motivation(self, value: str) -> None:
        if isinstance(value, str):
            self._motivation = value
        elif value is None:
            self._motivation = None
        else:
            try:
                self._motivation = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.motivation",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def publication_date(self) -> str:
        """
        This property is used for setting and getting the publication_date value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.publication_date = "..."
            >>> motivation = decision.publication_date

        :return: The string representation for publication_date
        """
        return self._publication_date

    @publication_date.setter
    def publication_date(self, value: str) -> None:
        if isinstance(value, str):
            self._publication_date = value
        elif value is None:
            self._publication_date = None
        else:
            try:
                self._publication_date = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.publication_date",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def language(self) -> str:
        """
        This property is used for setting and getting the language value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.language = "..."
            >>> motivation = decision.language

        :return: The string representation for language
        """
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        if isinstance(value, str):
            self._language = value
        elif value is None:
            self._language = None
        else:
            try:
                self._language = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.language",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def points(self) -> str:
        """
        This property is used for setting and getting the points value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> decision = Decision(...)
            >>> decision.points = "..."
            >>> points = decision.points

        :return: The string representation for language
        """
        return self._points

    @points.setter
    def points(self, value: str) -> None:
        if isinstance(value, str):
            self._points = value
        elif value is None:
            self._points = None
        else:
            try:
                self._points = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Decision.points",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    @wrap(entering, exiting)
    def insert_query(self) -> str:
        """
        This property is used for getting the insert_query value.

        Example usage:
            >>> decision = Decision(...)
            >>> points = decision.insert_query

        :return: The string representation for the insert_query
        """
        graph_type = None
        self.logger.debug(f"Annotations {self.annotations}")
        # using first annotation by default? this should only be called when there is an annotation added. ie. only one
        annotation = [] if len(self.annotations) == 0 else self.annotations[0]

        if annotation.model is not None:
            graph_type = GraphType.MODEL_ANNOTATION
        if annotation.user is not None:
            graph_type = GraphType.USER_ANNOTATION

        query: str = self.config.decision.insert_query.format(
            graph_uri=GraphType.match(
                config=self.config,
                value=graph_type
            ),
            uri=self.uri,
            annotation_uri=annotation.uri,
            annotation_subquery=annotation.subquery
        )
        self.logger.debug(f"Decision insert query: {query}")

        return query

    @property
    @wrap(entering, exiting)
    def last_human_annotation(self) -> Annotation | None:
        """
        This property is used for getting the latest human annotation from all current linked annotations.

        Example usage:
            >>> decision = Decision(...)
            >>> lha = decision.last_human_annotation

        :return: The last human annotation
        """

        if len(self.annotations) == 0:
            return None

        self.logger.debug(f"Annotations: {self.annotations}")
        user_annotations = [anno for anno in self.annotations if anno.user is not None]
        date_sorted_annotations = sorted(user_annotations, key=lambda anno: int(anno.date))

        return date_sorted_annotations[0] if len(date_sorted_annotations) != 0 else None

    @property
    @wrap(entering, exiting)
    def article_list(self) -> list[str] | None:
        """
        This property is used for getting the formatted linked articles.

        Example usage:
            >>> decision = Decision(...)
            >>> formatted_articles = decision.article_list

        :return: A list of string representations for the linked articles
        """
        if not self.articles:
            return None

        return [a.formatted_article for a in self.articles]

    @property
    def train_record(self) -> dict[str, str | list[str]]:
        """
        This property is used for getting the formatted training records.
        This can be seen as a dictionary with all relevant values that can be used for training for the specific
         Descision object.

        Example usage:
            >>> decision = Decision(...)
            >>> train_records = decision.train_record

        :return: A list of string representations for the linked articles
        """

        label_uris = None
        if annotation := self.last_human_annotation:
            label_uris = annotation.label_uris
            self.logger.debug(f"label uris: {label_uris}")

        return dict(
            uri=self.uri,
            uuid=self.uuid,
            description=self.description,
            articles=self.article_list,
            short_title=self.short_title,
            language=self.language,
            labels=label_uris
        )
