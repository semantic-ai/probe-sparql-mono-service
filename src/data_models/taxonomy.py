from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DataModelConfig
    from ..sparql import RequestHandler

    from logging import Logger
    from typing import Any

from ..enums import EndpointType, TaxonomyFindTypes
from ..utils import wrap, entering, exiting

from .base import Base
from functools import cache
import os, json


class Taxonomy(Base):
    """
    This class is used for parsing the taxonomy from the structured input that is received from the sparql.
    The whole idea is that the taxonomy can be of various depth, which makes searching and other extra functionality a
    recursive problem.

    Using a config enables you to have a custom SPARQL retrieval approach, you can adapt the relations and base query.

    Typical usage example (different from all other sparql models -> this one only makes sense to init from sparql):
        >>> taxonomy = Taxonomy.from_sparql(
                config=DataModelConfig(),
                logger=logging.logger,
                request_handler= RequestHandler(...),
                endpoint = ...
                taxonomy_item_uri = "..."
            )
    """

    def __init__(
            self,
            config: DataModelConfig,
            logger: Logger,
            uri: str,
            label: str = None,
            children: list[Taxonomy] = None,
            level: int = 0
    ) -> None:

        super().__init__(
            config=config,
            logger=logger
        )
        self._children = None
        self._label = None
        self._uri = None

        self.uri = uri
        self.label = label
        self.level = level
        self.children = children

        self._uri2label = None
        self._label2uri = None

    def todict(
            self,
            with_children: bool = False,
            max_depth: int = 10,
            **kwargs
    ) -> dict[str, str | list[...]]:
        """
        This function parses the current taxonomy tree to a dictionary

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> full_taxonomy_dictionary = taxonomy.todict(with_children=True)

        :param max_depth: maximum depth to retrieve child nodes from
        :param with_children: flag that allows you to go for full depth
        :return: dictionary for the given object
        """

        curr_depth = kwargs.get("curr_depth", 1)

        _dict = dict(
            uri=self.uri,
            label=self.label,
            level=self.level,
            children=[],
        )

        if curr_depth == max_depth:
            return _dict

        if with_children:
            _dict.update(
                dict(
                    children=[
                        c.todict(
                            with_children=with_children,
                            max_depth=max_depth,
                            curr_depth=curr_depth + 1
                        ) for c in self.children
                    ]
                )
            )
        return _dict

    def get_level_specific_labels(self, level: int):
        """
        This function provides fucntionality to retrieve ONLY a specific level of the taxonomy.

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> level_2_labels = taxonomy.get_level_specific_labels(level=2)

        :param level: The depth you want to retreive the labels from
        :return: List of found labels
        """

        labels = []
        if self.level == level:
            labels.append(self.label)

        for child in self.children:
            if child.level <= level:
                labels += child.get_level_specific_labels(level=level)

        return labels

    @wrap(entering, exiting)
    def get_labels(
            self,
            max_depth: int = 10,
            include_tree_indication: bool = False,
            **kwargs
    ) -> list[str | dict[str, str]]:
        """
        This function generates a flat list of labels for all the nodes in the taxonomy tree.

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> labels_up_to_2 = taxonomy.get_labels(max_depth=2)

        :param include_tree_indication: when to include the level it is based upon
        :param max_depth: The maximum depth level to extract from the label tree
        :return: a flat list of labels
        """
        labels = []

        curr_depth = kwargs.get("curr_depth", 0)

        # id prefixing for formatted output
        default_id_value = ""
        id_prefix = kwargs.get("id_prefix", default_id_value)
        id_suffix = kwargs.get("id_suffix", default_id_value)

        ids = [id_prefix, id_suffix]
        new_prefix = ".".join([str(i) for i in ids if i != default_id_value])

        self.logger.debug(f"Level: {self.level}, current_depth: {curr_depth}, max_depth: {max_depth}")

        if self.label:
            if not include_tree_indication:
                label = self.label

            else:
                label = (new_prefix, self.label)

            labels.append(label)

        if (curr_depth > self.level) or (curr_depth == max_depth):
            return labels

        new_depth = curr_depth + 1
        for i, child in enumerate(self.children, start=1):
            labels += child.get_labels(
                include_tree_indication=include_tree_indication,
                max_depth=max_depth,
                curr_depth=new_depth,
                id_prefix=new_prefix,
                id_suffix=i,
            )

        return labels

    def create_blank_config(self):
        """
        This function generates a blank config for a given taxonomy, this config could be used for creating the correct
        model inference graph when working with the config based multi layer predictions


        :return: json object containing the blank configuration
        """

        child_response = []
        for child in self.children:
            if child.children:
                child_response.append(child.create_blank_config())
        return dict(
            model_id="",
            flavour="",
            stage="",
            threshold=0.5,
            uri=self.uri,
            sub_nodes=child_response
        )





    def get_labels_for_node(
            self,
            search_term: str,
            search_kind: TaxonomyFindTypes
    ):
        """
        This function provides the child labels for the given input search term
        It calls the find function to retrieve the relevant information from the taxonomy tree.

        :param search_term:
        :param search_kind:
        :return:
        """

        result = self.find(
            search_term=search_term,
            search_kind=search_kind,
            with_children=True
        )

        print(result)

        taxonomy_node = None
        prev_max = 0
        for k, v in result.items():
            print(k, v)
            k = int(k)
            if max(k, prev_max) == k:
                prev_max = k
                taxonomy_node = v

        self.logger.info(f"Taxonomy node: {taxonomy_node}")
        return [c.get("label")for c in taxonomy_node.get("children", [])]

    @wrap(entering, exiting)
    def _remap_tree(
            self,
            entire_tree: dict[str, list[dict[str, str]]],
            subselector: str,
            curr_depth: int = 1
    ) -> list[Taxonomy]:
        """
        An internal function that allows us to recursively create the taxonomy tree, from the flat query response
        we got from the sparql.

        Working this way is highly optimized compared to recursively executing queries to build the taxonomy tree.

        Example usage:
            >>> / # internal method, no usage provided

        :param entire_tree: Full tree that is pulled from the sparql
        :param subselector: Key that sub selects the entire tree
        :param curr_depth: Index that identifies what depth we are currently at
        :return:
        """
        taxonomy_list = []
        new_depth = curr_depth + 1

        selected_sub_tree = entire_tree.get(subselector, [])
        for item in selected_sub_tree:
            label = item.get("name")
            uri = item.get("id")
            children = self._remap_tree(
                entire_tree=entire_tree,
                subselector=uri,
                curr_depth=new_depth
            )

            taxonomy_list.append(
                Taxonomy(
                    config=self.config,
                    logger=self.logger,
                    uri=uri,
                    label=label,
                    children=children,
                    level=curr_depth
                )
            )
        return taxonomy_list

    @wrap(entering, exiting)
    @cache
    def find(
            self,
            search_term: str,
            search_kind: TaxonomyFindTypes = TaxonomyFindTypes.URI,
            max_depth: int = 10,
            **kwargs
    ) -> dict[int, dict[str, str | int]]:
        """
        This function allows users to find the exact location of an item on the taxonomy tree.
        The provided search term will be used in combination with search_kind (LABEL or URI) in order to find the location
        in the tree. Once the location is found, it will respond with a dictionary that contains all parent nodes.
        These parent nodes are returned in a structured manner dict[<integer for level>: <taxonomy represented as dict>]]

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> taxonomy.find(
                search_term = "Bestuur",
                search_kind = TaxonomyFindTypes.LABEL,
                max_depth = 2
                )
            >>> # if the label is not before reaching the max depth, it will not be found

        :param search_kind: enum of what values to search on
        :param search_term: uri of the taxonomy that has to be found
        :param max_depth: maximum depth for the tree search
        :param kwargs: extra variables
        :return: a dictionary containing each level up to the found taxonomy
        """

        response = dict()
        _with_children = kwargs.get("with_children", False)
        search_term = search_term.lower()

        # check for
        curr_depth = kwargs.get("curr_depth", 1)

        self.logger.debug(f"Current depth = {curr_depth}, max depth = {max_depth}")
        if curr_depth == max_depth:
            return response

        for child in self.children:
            self.logger.debug(f"Child: {child.uri}")

            match search_kind:

                case TaxonomyFindTypes.URI:
                    if search_term == child.uri.lower():
                        response[child.level] = child.todict(with_children=_with_children)
                        break

                case TaxonomyFindTypes.LABEL:
                    if search_term == child.label.lower():
                        response[child.level] = child.todict(with_children=_with_children)
                        break

                # check for taxonomy in child nodes &
            if child_response := child.find(
                    search_term=search_term,
                    search_kind=search_kind,
                    max_depth=max_depth,
                    curr_depth=curr_depth + 1,
                    with_children=_with_children
            ):
                response = {
                    **child_response,
                    child.level: child.todict(with_children=_with_children)
                }
                break  # assume that there is only one node that can contain the searched value?

        return response

    @property
    @wrap(entering, exiting)
    def uri2label(self) -> dict[str, str]:
        """
        Property (getter only) to retrieve the uri2label for the Taxonomy object.

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> uri2label = taxonomy.uri2label

        :return: The uri2label dictionary
        """
        if not self._uri2label:
            self._uri2label = dict()

            if self.label is not None:
                self._uri2label[self.uri] = self.label

            for child in self.children:
                self._uri2label.update(child.uri2label)

        return self._uri2label

    @property
    @wrap(entering, exiting)
    def label2uri(self) -> dict[str, str]:
        """
        Property (getter only) to retrieve the label2uri for the Taxonomy object.

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> uri2label = taxonomy.label2uri

        :return: The label2uri dictionary
        """
        return {v: k for k, v in self.uri2label.items()}

    @classmethod
    @wrap(entering, exiting)
    def from_sparql(
            cls,
            config: DataModelConfig,
            logger: Logger,
            request_handler: RequestHandler,
            endpoint: EndpointType,
            taxonomy_item_uri: str
    ) -> Taxonomy:
        """
        Class method for class initialization from sparql.
        This function creates the taxonomy tree from sparql, the taxonomy tree is created from taxonomy objects that are
        nested in the children property.

        Example usage:
            >>> taxonomy = Taxonomy.from_sparql(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    request_handler = ...,
                    endpoint = EndpointType.TAXONOMY,
                    taxonomy_item_uri = "..."
                )

        :param config: the general config used in the project
        :param logger: the object used for logging
        :param request_handler: the request wrapper for sparql
        :param endpoint: the endpoint enum for endpoint reference
        :param taxonomy_item_uri: the taxonomy uri
        :return: an instance of the Taxonomy object
        """

        query_all_labels = config.taxonomy.query_all_children.format(
            taxonomy_schema=taxonomy_item_uri
        )

        logger.debug(f"All labels query: {query_all_labels}")
        labels = request_handler.post2json(query_all_labels, endpoint=endpoint)

        tree_dict = {}
        for label in labels:
            label_id = label.get("concept")
            label_name = label.get("label")
            label_parent_id = label.get("broaderConcept", "")
            if label_parent_id == "":
                label_parent_id = "top_level"

            tree_dict[label_parent_id] = tree_dict.get(label_parent_id, []) + [{"id": label_id, "name": label_name}]

        # Check if top level domain is one node, remove and make toplevel second layer
        if len(tree_dict["top_level"]) == 1:
            label = tree_dict["top_level"][0]["id"]
            tree_dict["top_level"] = tree_dict[label]
            del tree_dict[label]

        _class = cls(
            config,
            logger=logger,
            uri=taxonomy_item_uri
        )

        _class.children = _class._remap_tree(
            entire_tree=tree_dict,
            subselector="top_level"
        )

        return _class

    @classmethod
    def from_dict(
            cls,
            config: DataModelConfig,
            logger: Logger,
            dictionary: dict[Any]
    ) -> Taxonomy:
        """
        Class method for class initialization from a dictionary.

        Example usage:
            >>> annotation = Taxonomy.from_dict(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    dictionary = {...: ...}
                )

        :param dictionary: dictionary containing the parsed taxonomy
        :param config: the general DataModelConfig
        :param logger: logger object that can be used for logs
        :return: an instance of the Annotation Class
        """

        uri = dictionary.get("uri")
        label = dictionary.get("label")
        level = dictionary.get("level")

        if children := dictionary.get("children"):
            children = [
                Taxonomy.from_dict(
                    config=config,
                    logger=logger,
                    dictionary=c
                ) for c in children
            ]

        return cls(
            config=config,
            logger=logger,
            uri=uri,
            label=label,
            level=level,
            children=children
        )

    @classmethod
    def from_checkpoint(
            cls,
            config: DataModelConfig,
            logger: Logger,
            checkpoint_folder: str
    ) -> Taxonomy:
        """
        Class method for class initialization from a created checkpoint.

        Example usage:
            >>> annotation = Taxonomy.from_checkpoint(
                    config = DataModelConfig(),
                    logger = logging.logger,
                    checkpoint_dir = "..."
                )

        :param checkpoint_folder: string indication of where the checkpoint is located
        :param config: the general DataModelConfig
        :param logger: logger object that can be used for logs
        :return: an instance of the Annotation Class
        """

        checkpoint_taxonomy_path = os.path.join(checkpoint_folder, "taxonomy.json")

        with open(checkpoint_taxonomy_path, "r") as f:
            taxonomy_dictionary: dict = json.load(f)

        return cls.from_dict(
            config=config,
            logger=logger,
            dictionary=taxonomy_dictionary
        )

    @property
    def children(self):
        """
        Property that provides a getter and setter for the taxonomy children nodes.

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> taxonomy_children = taxonomy.children
            >>> taxonomy.children = [Taxonomy(...), ...]

        :return: The children that are linked to the taxonomy
        """
        return self._children

    @children.setter
    def children(self, value):
        self._children = value or []

    @property
    def label(self) -> str:
        """
        Property that provides acces to the label for a given taxonomy

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> taxonomy_label = taxonomy.label
            >>> taxonomy.label = "..."

        :return: The label for the taxonomy item
        """
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def uri(self) -> str:
        """
        Property that provides acces to the uri for a given taxonomy

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> taxonomy_uri = taxonomy.uri
            >>> taxonomy.uri = "..."

        :return: The uri for the taxonomy item
        """
        return self._uri

    @uri.setter
    def uri(self, value: str) -> None:
        self._uri = value

    @property
    def all_linked_labels(self) -> list[str]:
        """
        Property that provides getter access to all_linked_labels for the taxonomy isntance

        Example usage:
            >>> taxonomy = Taxonomy(...)
            >>> taxonomy_label = taxonomy.all_linked_labels

        :return: List of all labels linked to taxonomy
        """
        linked_labels = [self.label]

        for child in self.children:
            linked_labels += child.all_linked_labels

        return linked_labels
