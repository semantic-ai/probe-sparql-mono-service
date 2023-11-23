from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger

from ..enums import EndpointType
from ..utils import wrap, entering, exiting

from .base import Base
from .taxonomy import Taxonomy


from tqdm import tqdm


class TaxonomyTypes(Base):
    """
    This class is used for providing access to all different taxonomy nodes under the taxonomy masternode, this masternode
    is a default value that can be overwritten with environment variables. (see config)

    Using a config enables you to have a custom SPARQL retrieval approach, you can adapt the relations and base query.

    Typical usage example (different from all other sparql models -> this one only makes sense to init from sparql):
        >>> taxonomies = TaxonomyTypes.from_sparql(
                config=DataModelConfig(),
                logger=logging.logger,
                request_handler= RequestHandler(...),
                endpoint = ...
            )
    """
    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            taxonomies: list[Taxonomy]
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )
        self.base_uri = config.taxonomy.master_node_uri
        self.taxonomies = taxonomies

    @wrap(entering, exiting)
    def get(self, taxonomy_uri: str) -> Taxonomy | None:
        """
        This function checks the list of existing taxonomies and returns the matching taxonomy

        Example usage:
            >>> taxonomies = TaxonomyTypes(...)
            >>> taxoxnomy = taxonomies.get(taxonomy_uri="...")


        :param taxonomy_uri: taxonomy_uri to check for
        :return: taxonomy object when it exists
        """
        for taxonomy in self.taxonomies:
            if taxonomy.uri == taxonomy_uri:
                return taxonomy
        else:
            return None

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            endpoint: EndpointType
    ) -> TaxonomyTypes:
        """
        Class method for class initialization from sparql.
        This function loads all taxonomies linked to the parent taxonomy node

        Example usage:
            >>> taxonomy = TaxonomyTypes.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = ...,
                    endpoint = EndpointType.TAXONOMY
                )
        :param config: the general config used in the project
        :param logger: object used for logging
        :param request_handler: the request wrapper for sparql
        :param endpoint: endpoint enum to use for requests
        :return: an isntance of the taxonomytype object
        """

        taxonomies_query = config.taxonomy.query_master_nodes.format(
            master_node_uri=config.taxonomy.master_node_uri,
            master_child_relation=config.taxonomy.master_child_relation
        )

        logger.debug(f"taxonomy query: {taxonomies_query}")
        taxonomies_response = request_handler.post2json(taxonomies_query, endpoint)

        taxonomies = []
        for taxonomy in tqdm(taxonomies_response, desc="generating taxonomies"):
            taxonomies.append(Taxonomy.from_sparql(
                config=config,
                logger=logger,
                taxonomy_item_uri=taxonomy.get("uri"),
                request_handler=request_handler,
                endpoint=EndpointType.TAXONOMY)
            )

        return cls(
            config=config,
            logger=logger,
            taxonomies=taxonomies
        )
