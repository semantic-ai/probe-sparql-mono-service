from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger


from ..utils import wrap, entering, exiting, deprecated
from .base import Base



class User(Base):
    """
    This class is used for parsing the linked user(s) for a given decision.

    Currently, this class is not linked to the data infrastructure (annonimety issue)

    Typical usage example:
        >>> taxonomy = User(
                config=Config(),
                logger=logging.logger,
                username="...",
                email="...@...",
                uri="..."
            )
    """
    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            username: str = None,
            email: str = None,
            uri: str = None
    ) -> None:
        super().__init__(
            config=config,
            logger=logger
        )

        # uri
        self.uri = uri or self.generate_uri(self.config.user.uri_base)
        self.username = username
        self.email = email

    @classmethod
    # @deprecated
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            uri: str
    ) -> User:
        """
        Class method for class initialization from sparql uri

        ===
        This function is currently not used and fully implemented
        ===

        :param config: the general config used in the project
        :param logger: object used for logging
        :param request_handler: the request wrapper for sparql
        :param uri: the user uri to use
        :return: instance of a user
        """
        return cls(config, logger=logger, uri=uri)

    @wrap(entering, exiting)
    def __repr__(self):
        return f"<User: {self.uri, self.username}>"
