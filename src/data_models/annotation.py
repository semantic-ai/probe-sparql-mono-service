from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from logging import Logger
    from ..sparql import RequestHandler

from ..utils import wrap, entering, exiting
from ..errors import CustomValueError

from .base import Base
from .label import Label
from .model import Model
from .taxonomy import Taxonomy
from .user import User

from datetime import datetime


class Annotation(Base):
    """
    This class is used for parsing annotations from the structured input that is received from the sparql
    into a python object with extended functionality.

    Using the config it is possible to extend this classes functionality for loading and
    saving custom configs. Given there is some resemblance with the initial sparql schema

    An annotation can come from two seperate sources, these are specified as specific properties.
    The two possible annotation types are:
        1. User annotation (annotation made by a user)
            >>> user_annotation = Annotation(
                    date=datetime.now(),
                    config=DataModelConfig(),
                    logger=logging.logger,
                    taxonomy=Taxonomy(...),
                    user=User(...),
                    labels=[Labels(...), ...],
                )
        2. Model annotation (annotation made by a model)
            >>> model_annotation = Annotation(
                    date=datetime.now(),
                    config=DataModelConfig(),
                    logger=logging.logger,
                    taxonomy=Taxonomy(...),
                    model=Model(...),
                    labels=[Labels(...), ...],
                )

        For more specific usage, check functions below.
    """

    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            taxonomy: Taxonomy,
            date: datetime | int,
            user: User = None,
            labels: list[Label] = None,
            model: Model = None,
            uri: str = None
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )

        # internal values
        self._labels = None
        self._taxonomy = None
        self._model = None
        self._date = None
        self._user = None

        self.uri = uri or self.generate_uri(self.config.annotation.uri_base)

        self.labels = labels
        self.taxonomy = taxonomy
        self.model = model
        self.date = date
        self.user = user

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            annotation_uri: str
    ) -> Annotation:
        """
        Class method for class initialization from sparql.
        When provided with an uri from an annotations, it will automatically execute all related queries
        to populate the object with all necessary information.

        Example usage:
            >>> annotation = Annotation.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = ...,
                    annotation_uri = "..."
                )

        :param config: the general DataModelConfig
        :param logger: logger object that can be used for logs
        :param request_handler: the request wrapper used for sparql requests
        :param annotation_uri: the uri which is used to find all relevant information
        :return: an instance of the Annotation Class
        """

        # Get annotation information
        query_get_annotation_info = config.annotation.query_annotation_info.format(
            annotation_uri=annotation_uri
        )

        query_all_labels = config.annotation.query_linked_labels.format(annotation_uri=annotation_uri)

        logger.debug(f"All label Query: ```{query_all_labels}```")
        logger.debug(f"One annotation Query: ```{query_get_annotation_info}```")

        all_response = request_handler.post2json(query_get_annotation_info)
        label_response = request_handler.post2json(query_all_labels)

        logger.debug(f"All label response: {all_response}")
        logger.debug(f"Label response: {label_response}")

        date = all_response[0].get("date", None)
        model_uri = all_response[0].get("model_uri", None)
        taxonomy_uri = all_response[0].get("taxonomy_uri", None)
        user_uri = all_response[0].get("user_uri", None)

        taxonomy = Taxonomy(
            config=config,
            logger=logger,
            uri=taxonomy_uri)

        # the following code is commented out, since we are currently not implementing
        # user logic for assigning labels

        user = User.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            uri=user_uri
        ) if user_uri else None

        # user = None

        model = Model.from_sparql(
            config=config,
            logger=logger,
            request_handler=request_handler,
            uri=model_uri
        ) if model_uri else None

        labels = []
        for label in label_response:
            labels.append(
                Label.from_sparql(
                    config=config,
                    logger=logger,
                    request_handler=request_handler,
                    uri=label["label_uri"]
                )
            )

        return cls(
            config=config,
            logger=logger,
            taxonomy=taxonomy,
            labels=labels,
            user=user,
            model=model,
            date=date
        )

    @property
    @wrap(entering, exiting)
    def subquery(self) -> str:
        """
        Property (getter only) to retrieve the subquery for the annotation object.
        The sub queries are generally used for creation of insert statements, it checks for the user/model annotation status
        and calls the specific submodules in order to create the complete annotation statement.

        Example usage:
            >>> article = Annotation(...)
            >>> labels = article.subquery

        :return: The formatted subquery as string
        """

        formated_labels = "\n".join([l.subquery for l in self.labels])

        model_reference = None
        user_reference = None
        if self.model:
            self.logger.debug("Model is registered")

            model_reference = f"""\
            {self.uri}    {self.config.annotation.probe_model_relation}     {self._ensure_encapsulation(self.model.uri)} .
             
            {self.model.subquery if self.model.register else ""}
            """

        if self.user:
            self.logger.debug("User is registered")
            user_reference = f"""\
            {self.uri} {self.config.annotation.user_relation} {self._ensure_encapsulation(self.user.uri)} .
            """

        subquery = self.config.annotation.sub_query.format(
            uri=self._ensure_encapsulation(self.uri),
            taxonomy=self._ensure_encapsulation(self.taxonomy.uri),
            timestamp=self.date,
            label_uris=", ".join([l.uri for l in self.labels]),
            # not to be confused with taxonomy label uris (unique reference)
            user_reference=user_reference or "",
            model_reference=model_reference or "",
            labels=formated_labels
        )

        self.logger.debug(f"Subquery for annotation: {subquery}")
        return subquery

    @property
    @wrap(entering, exiting)
    def label_uris(self) -> list[str]:
        """
        This property is used to retrieve the list of taxonomy uris for all the linked labels.

        :return: The linked label uris as a list of strings
        """
        return [label.taxonomy_node_uri for label in self.labels]

    @property
    def labels(self) -> list[Label | ...]:
        """
        This property returns a the linked label object(s).
        The property is extended with extra logic to force/check if the input is of type list[Label] or Label

        Example usage:
            >>> article = Annotation(...)
            >>> article.labels = [Label(...), ...]
            >>> labels = article.labels

        :return: All the linked labels as List[Label] or an empty list if there are no labels linked
        """
        return self._labels

    @labels.setter
    def labels(self, value: list[Label] | None) -> None:
        if value is None:
            self._labels = []
        elif isinstance(value, list):
            self._labels = value
        else:
            raise CustomValueError(
                property="Annotation.labels",
                expected_type=list[Label] | None,
                received_type=type(value)
            )

    @property
    def taxonomy(self):
        """
        The property is used for setting and getting the taxonomy value.
        The setter contains extra functionality to cast it to the specific required type.

        Example usage:
            >>> article = Annotation(...)
            >>> article.date = Taxonomy(...)
            >>> taxonomy = article.taxonomy

        :return: The specific linked taxonomy
        """
        return self._taxonomy

    @taxonomy.setter
    def taxonomy(self, value: Taxonomy) -> None:
        if isinstance(value, Taxonomy):
            self._taxonomy = value
        else:
            raise CustomValueError(
                property="Annotation.taxonomy",
                expected_type=Taxonomy,
                received_type=type(value)
            )

    @property
    def model(self) -> Model | None:
        """
        This property is used for setting and getting the model value.
        The setter contains extra functionality to cast it to the specific required type.

        Example usage:
            >>> article = Annotation(...)
            >>> article.model = Model(...)
            >>> model = article.model

        :return: The provided model or None if no model is available
        """
        return self._model

    @model.setter
    def model(self, value: Model | None) -> None:
        if isinstance(value, Model) or (value is None):
            self._model = value
        else:
            raise CustomValueError(
                property="Annotation.model",
                expected_type=Model | None,
                received_type=type(value)
            )

    @property
    def date(self) -> int:
        """
        This property is used for setting and getting the date value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> article = Annotation(...)
            >>> article.date = datetime.now()
            >>> date = article.date

        :return: The integer epoch time value for the provided timestamp
        """
        return self._date

    @date.setter
    def date(self, value: datetime | int | float) -> None:
        try:
            if isinstance(value, datetime):
                self.logger.debug(f"Converting datetime to integer timestamp (ts: {value})")
                self._date = int(value.timestamp())
                return

            elif isinstance(value, float):
                self.logger.debug(f"Converting float (presumably timestamp) to int timestamp")
                self._date = int(value)
                return

            elif isinstance(value, int):
                self.logger.debug(f"Asigning value")
                self._date = value
                return

            elif value is None:
                self.logger.debug(f"Date is none")
                self._date = value
                return

            elif isinstance(value, str):
                self._date = int(float(value))
                return
        except:
            pass

        error = CustomValueError(
            property="Annotation.date",
            expected_type=datetime | int | float,
            received_type=type(value)
        )

        self.logger.critical(error.message)
        raise error


    @property
    def user(self) -> User:
        """
        This property is used for setting and getting the user value.
        The setter contains extra functionality to cast it to the specific required type.

        Example usage:
            >>> article = Annotation(...)
            >>> article.user = "..."
            >>> user = article.user

        :return: The provided user or None if no model is available
        """
        return self._user

    @user.setter
    def user(self, value) -> None:
        if isinstance(value, User) or (value is None):
            self._user = value
        else:
            raise CustomValueError(
                property="Annotation.user",
                expected_type=User | None,
                received_type=type(value)
            )
