from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger

from ..utils import wrap, entering, exiting
from ..errors import CustomValueError
from ..enums import GraphType, EndpointType

from .base import Base

from logging import Logger
from datetime import datetime


class Model(Base):
    """
    This class is used for parsing Model isntances from the structured input that is received from the sparql
    into a python object with extended functionality.

    Using the config it is possible to extend this classes functionality for loading and
    saving custom configs. Given there is some resemblance with the initial sparql schema

    Typical usage example:
        >>> model = Model(
            config=DataModelConfig(),
            logger=logging.logger.
            mlflow_reference="...",
            category="...",
            register=True
        )
    """

    def __init__(
            self, config: DataModelConfig,
            logger: Logger,
            name: str = None,
            mlflow_reference: str = None,
            date: int = datetime.now(),
            category: str = None,
            registered_model: str = None,
            uri: str = None,
            register: bool = False
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )

        self._name = None
        self._category = None
        self._date = None
        self._mlflow_reference = None
        self._registered_model = None
        self._register = None
        self._uri = None

        # uri
        self.uri = uri or self.generate_uri(self.config.model.uri_base)

        # relations
        self.name = name
        self.category = category
        self.mlflow_reference = mlflow_reference
        self.registered_model = registered_model
        self.date = date
        self.register = register

    @property
    def name(self) -> str:
        """
        This property is used for setting and getting the name value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> model = Model(...)
            >>> model.name = "..."
            >>> name = model.name

        :return: The string uri as string
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value, str):
            self._name = value
        elif value is None:
            self._name = None
        else:
            try:
                self._name = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Model.name",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def category(self) -> str:
        """
        This property is used for setting and getting the category value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> model = Model(...)
            >>> model.category = "..."
            >>> name = model.category

        :return: The string category as string
        """
        return self._category

    @category.setter
    def category(self, value: str) -> None:
        if isinstance(value, str):
            self._category = value
        elif value is None:
            self._category = None
        else:
            try:
                self._category = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Model.category",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def mlflow_reference(self) -> str:
        """
        This property is used for setting and getting the mlflow_reference value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> model = Model(...)
            >>> model.mlflow_reference = "..."
            >>> mlflow_reference = model.mlflow_reference

        :return: The string mlflow_reference as string
        """
        return self._mlflow_reference

    @mlflow_reference.setter
    def mlflow_reference(self, value: str) -> None:
        if isinstance(value, str):
            self._mlflow_reference = value
        elif value is None:
            self._mlflow_reference = None
        else:
            try:
                self._mlflow_reference = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Model.mlflow_reference",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def registered_model(self) -> str:
        """
        This property is used for setting and getting the registered_model value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> model = Model(...)
            >>> model.registered_model = "..."
            >>> registered_model = model.registered_model

        :return: The string registered_model as string
        """
        return self._registered_model

    @registered_model.setter
    def registered_model(self, value: str) -> None:
        if isinstance(value, str):
            self._registered_model = value
        elif value is None:
            self._registered_model = None
        else:
            try:
                self._registered_model = str(value)
            except Exception as ex:
                raise CustomValueError(
                    property="Model.registered_model",
                    expected_type=str,
                    received_type=type(value)
                )

    @property
    def register(self) -> bool:
        """
        This property is used for setting and getting the register value.
        The setter contains extra functionality to cast it to the specifically required type

         Example usage:
            >>> model = Model(...)
            >>> model.register = True
            >>> register = model.register

        :return: The bool value for register
        """
        return self._register

    @register.setter
    def register(self, value: bool) -> None:
        if isinstance(value, bool):
            self._register = value
        else:
            raise ValueError

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            uri: str
    ) -> Model:
        """
        This function is the classmethod that creates an instance of the model class from a given model uri.

        :param config: the generatl config used in the project
        :param logger: the object that can be used for logging
        :param request_handler: the request wrapper for sparql
        :param uri: the model uri used to poppulate the model object
        :return:
        """

        query = config.model.query.format(uri=uri)
        logger.debug(f"Model Query: ```{query}```")
        model_response = request_handler.post2json(query)

        return cls(
            config=config,
            logger=logger,
            uri=uri, name=model_response[0]['model_name'],
            mlflow_reference=model_response[0]['mlflow_link'],
            date=model_response[0]['create_data'],
            category=model_response[0]['category'],
            registered_model=model_response[0]['mlflow_model']
        )

    @property
    @wrap(entering, exiting)
    def subquery(self) -> str:
        """
        Property (getter only) to retrieve the subquery for the Model object.
        The sub queries are generally used for creation of insert statements.
        It will automaticly execute the calls for the submodules in order to create the complete annotation statement.

        Example usage:
            >>> model = Model(...)
            >>> sub_query = model.subquery

        :return: The formatted subquery as string
        """

        query = self.config.model.sub_query.format(
            uri=self.uri,
            date=self.date,
            name=self.name or "",
            category=self.category or "",
            mlflow_reference=self.mlflow_reference or "",
            registered_model=self.registered_model or ""
        )

        self.logger.debug(f"Subquery for model: {query}")

        return query

    @property
    def date(self) -> int:
        """
        This property is used for setting and getting the date value.
        The setter contains extra functionality to cast it to the specifically required type.

        Example usage:
            >>> model = Model(...)
            >>> model.date = datetime.now()
            >>> date = model.date

        :return: The integer epoch time value for the provided timestamp
        """
        return self._date

    @date.setter
    def date(self, value) -> None:

        if isinstance(value, datetime):
            self.logger.debug(f"Converting datetime to integer timestamp (ts: {value})")
            self._date = int(value.timestamp())
        elif isinstance(value, float):
            self.logger.debug(f"Converting float (presumably timestamp) to int timestamp")
            self._date = int(value)
        elif isinstance(value, int):
            self.logger.debug(f"Asigning value")
            self._date = value
        else:
            self.logger.critical(f"Unsuported timestamp caught ({type(value)})")
            raise Exception(f"{type(value)} is not supported")

    def write_to_sparql(self, request_handler: RequestHandler):
        subquery = self.config.model.sub_query.format(
            uri=self._ensure_encapsulation(self.uri),
            date=self.date,
            name=self.name,
            category=self.category,
            mlflow_reference=self.mlflow_reference,
            registered_model=self.registered_model
        )

        graph_uri = GraphType.match(
            config=self.config,
            value=GraphType.MODEL_INFORMATION
        )

        query = f"""\
        PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        
        INSERT DATA {{
            GRAPH {graph_uri} {{
                {subquery}
            }}
        }}
        """
        request_handler.post2json(
            query=query,
            endpoint=EndpointType.DECISION
        )


