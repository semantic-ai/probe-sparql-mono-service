from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..models import get_model, Model
    from ..sparql import RequestHandler
    from ..dataset import DatasetBuilder
    from logging import Logger

from ..data_models import TaxonomyTypes, Decision, Label, Annotation, Taxonomy, Model as data_model
from ..models import get_model
from ..dataset import create_dataset
from ..enums import EndpointType, TaxonomyFindTypes

from datetime import datetime
from uuid import uuid4
from tqdm import tqdm
from collections import Counter
import copy


class InferenceModel:
    """
    This class wraps all different types of models into a inference focused framework.
    The wrapper contains extra relevant information extracted from the provided config.
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model: Model,
            model_config: dict
    ) -> None:
        self._label = None
        self._uri = None
        self._model_config = None

        self.config = config
        self.logger = logger
        self.model = model
        self.model_config = model_config

    def predict(self, text: str, **kwargs) -> dict[str, float]:
        self.logger.debug(f"Model prediction type: {type(self.model)}")
        return self.model.classify(text=text, multi_label=True, **kwargs)

    @classmethod
    def from_custom_config(
            cls,
            config: Config,
            logger: Logger,
            model_config: dict,
            taxonomy: Taxonomy
    ) -> InferenceModel:
        _class = cls(
            config=config,
            logger=logger,
            model=get_model(
                config=config,
                logger=logger,
                taxonomy=taxonomy,
                model_id=model_config.get("model_id"),
                specific_model_type=model_config.get("flavour"),
                model_stage=model_config.get("stage")
            ),
            model_config=model_config
        )

        uri_value = model_config.get("uri").split("__")[-1]
        label_in_tree = taxonomy.find(
            search_term=uri_value,
            search_kind=TaxonomyFindTypes.URI,
        )

        label = None
        for k, v in label_in_tree.items():
            if v.get("uri") == uri_value:
                label = v.get("label")

        # when label is not found raise
        logger.debug(f"label: {label}")

        _class.label = "main" if "parent_node" in model_config.get("model_id", "") else label
        _class.uri = uri_value

        if "parent_node" in model_config.get("model_id", ""):
            _class.label = "main"
            _class.labels = [c.label for c in taxonomy.children]
        else:
            _class.label = label

            child_labels = taxonomy.get_labels_for_node(
                search_term=uri_value,
                search_kind=TaxonomyFindTypes.URI
            )
            _class.labels = child_labels

        return _class

    @property
    def labels(self) -> list[str]:
        """
        Property containing the labels that are explicitly extracted form the provided config and the parent node, these
         are helpfull for downstream model tasks/ passthrough logic for deeper levels
        :return: Current labels linked to taxonomy node
        """
        return self._labels

    @labels.setter
    def labels(self, value) -> None:
        self._labels = value

    @property
    def model_config(self) -> dict[str, str | float | dict]:
        """
        Property that keeps the model config (could be used to add extra validation to model_config file)
        :return: Model config
        """
        return self._model_config

    @model_config.setter
    def model_config(self, value: dict) -> None:
        self._model_config = value

    @property
    def label(self) -> str:
        """
        Property that enables the getting and setting of the label for the taxonomy node
        :return: current taxonomy label
        """
        return self._label

    @label.setter
    def label(self, value) -> None:
        self._label = value.lower()

    @property
    def uri(self) -> str:
        """
        Property that enables the getting and setting of the uri for the taxonomy node
        :return: current taxonomy uri
        """
        return self._uri

    @uri.setter
    def uri(self, value) -> None:
        self._uri = value


class InferenceModelTree:
    """
    This class implements the config based prediction tree.

    The config provides all information that is used for predicting the provided text, more information on the prediction process
    can be find at the predict function.
    The config should look like this (up to level two Gent Words config):
        >>>      {
        >>>          "model_id": "mlflow:/bert__gent_words__parent_node",
        >>>          "flavour": "huggingface_model",
        >>>          "stage": "Production",
        >>>          "threshold": 0.5,
        >>>          "uri": "http://stad.gent/id/concepts/gent_words",
        >>>          "sub_nodes": [
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__over_gent_en_het_stadsbestuur",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/22",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__cultuur,_sport_en_vrije_tijd",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/23",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__mobiliteit_en_openbare_werken",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/24",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__groen_en_milieu",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/25",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__onderwijs_en_kinderopvang",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/26",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__samenleven,_welzijn_en_gezondheid",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/28",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        >>>              "flavour": "embedding_regular",
        >>>              "stage": "Production",
        >>>              "threshold": 0.41,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/3",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__werken_en_ondernemen",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/30",
        >>>              "sub_nodes": []
        >>>            },
        >>>            {
        >>>              "model_id": "mlflow:/bert__gent_words__wonen_en_(ver)bouwen",
        >>>              "flavour": "huggingface_model",
        >>>              "stage": "Production",
        >>>              "threshold": 0.5,
        >>>              "uri": "https://stad.gent/id/concepts/gent_words/31",
        >>>              "sub_nodes": []
        >>>            }
        >>>          ]
        >>>        }

    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            model_configuration: dict,
            models: dict[str, InferenceModel],
            taxonomy: Taxonomy
    ) -> None:
        self.config = config
        self.logger = logger
        self.request_handler = request_handler
        self.model_config = model_configuration
        self.models = models
        self.taxonomy = taxonomy

    def predict(self, text: str) -> dict[str, float]:
        """
        This function handles the prediction process for the inference class.
        It processes the provided text into the designated classes, this is done recursively over the predefined config.

        This block functions as follows:
            1. Predict for the base level
            2. If a label exceeds the threshold, execute sub predictions until there is nothing exceeding the threshold.
            3. Return ALL found labels that are above the threshold

        :param text: The input text that should be used for the predictions
        :return: All labels that can be linked to the text, using the config pre-defined models
        """
        labels = {}
        # start with prediction of parent model
        prediction = self.models["main"].predict(text=text)

        # easy solution for recursive predictions
        def sub_predict(pred: dict, m_conf: dict):
            """
            This sub function handles the recursion based prediciton

            :param pred: the prediction that has been made
            :param m_conf: the model config that is used for the models providing the predictions
            :return:
            """
            self.logger.debug(f"Prediction: {pred}")
            for k, v in pred.items():
                k = k.lower()

                threshold = m_conf.get("threshold", 0.5)
                thresh_diff = abs(0.5 - threshold)

                corrected_v = v - thresh_diff if threshold > 0.5 else v + thresh_diff
                if not v >= threshold:
                    self.logger.debug(f"Current label {k} did not pass the threshold {threshold}")
                    continue

                labels[k] = corrected_v

                if model := self.models.get(k, None):
                    self.logger.debug(f"current label: {k} not in {self.models.keys()}")
                    new_pred = model.predict(text=text, labels=model.labels)
                    sub_predict(
                        pred=new_pred,
                        m_conf=self.models[k].model_config
                    )

        sub_predict(
            pred=prediction,
            m_conf=self.models["main"].model_config
        )

        return labels

    @classmethod
    def from_model_config(
            cls,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            model_configuration: dict
    ) -> InferenceModelTree:
        """
        Class method that initializes the inference model tree class.

        This class is responsible for the tree based taxonomy predictions, the models that are used for these predictions are specified int he model config.
        :param config: The global project config
        :param logger: The global project logger
        :param request_handler:  The provided request handler for executing sparql requests
        :param model_configuration: the configuration file that contains all relevant information for building the taxonomy tree
        :return: An instance of the InferenceModelTree class
        """

        taxonomies = TaxonomyTypes.from_sparql(
            config=config.data_models,
            logger=logger,
            request_handler=request_handler,
            endpoint=EndpointType.TAXONOMY
        )

        # selection of specific taxonomy
        for taxonomy in taxonomies.taxonomies:
            if taxonomy.uri == model_configuration.get("uri"):
                break

        models = dict()

        def load_models(model_config: dict) -> None:
            """
            This function that loads the models out of the provided config recursively since there is no predefined depth.

            :param model_config: The config containing all the configuration
            :return: Nothing
            """
            model = InferenceModel.from_custom_config(
                config=config,
                logger=logger,
                model_config=model_config,
                taxonomy=taxonomy
            )

            models[model.label] = model

            for sub_config in model_config.get("sub_nodes", []):
                load_models(sub_config)

        load_models(model_config=model_configuration)

        return cls(
            config=config,
            logger=logger,
            model_configuration=model_configuration,
            request_handler=request_handler,
            models=models,
            taxonomy=taxonomy
        )


class Inference:
    """
    Class that provides basic inference functionality

    It acts as the glue between the dataset, the config based inference tree and the
    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            dataset_builder: DatasetBuilder,
            inference_model_tree: InferenceModelTree,
            dataset_type: str,
            model_reference: Model
    ):
        self.config = config
        self.config.run.dataset.type = dataset_type

        self.logger = logger
        self.request_handler = request_handler
        self.dataset_builder = dataset_builder
        self.inference_model = inference_model_tree
        self.model_reference = model_reference

        self.label2id = {k.lower(): v for k, v in self.dataset_builder.taxonomy.label2uri.items()}

    def _single_prediction(self, text: str) -> dict[str, float]:
        return self.inference_model.predict(text=text)

    def _prediction_to_labels(self, predictions: dict[str, float]) -> list[Label]:
        """
        This function remaps the string labels to the data_models label object with all correct
        references

        :param predictions: The predictions that are retrieved from the model
        :return:
        """
        new_labels: list[Label] = list()
        already_added_node: set[str] = set()  # using set for optimized 'in' statement

        for label, score in predictions.items():
            label_uri = self.label2id.get(label)
            find_out = self.dataset_builder.taxonomy.find(
                search_term=label,
                search_kind=TaxonomyFindTypes.LABEL
            )

            self.logger.info(f"Found uri for label {label} -> {label_uri}")
            self.logger.info(f"Find output: {find_out}")

            if label_uri is None:
                self.logger.warning(f"Label skipped ({label}) because uri not found")
                continue

            if label_uri in already_added_node:
                self.logger.info(f"Label already added {label_uri}")
                continue

            already_added_node.update([v.get("uri") for _, v in find_out.items()])

            new_labels.append(
                Label(
                    config=self.config.data_models,
                    logger=self.logger,
                    score=score,
                    taxonomy_node_uri=label_uri
                )
            )

        def filter_label_list(labels: list[Label], depth=1):

            label_uris: dict[int, str] = dict()  # index <-> uri value mapping
            label_level: dict[int, int] = dict()  # index <-> level mapping
            idx_to_remove: list[int] = list()  # mem for idx to remove before stepping in
            new_depth = depth + 1
            # create dictionary with index, value for further processing
            for i, label in enumerate(labels):
                found_label = self.dataset_builder.taxonomy.find(
                    search_term=label.taxonomy_node_uri.replace('<', '').replace('>', ''),
                    search_kind=TaxonomyFindTypes.URI
                )
                label_uris[i] = found_label.get(depth, {}).get("uri", None)
                label_level[i] = len(found_label.items())  # length indicates to find label!

            # create counter instance of label objects
            counter = Counter(list(label_uris.values()))

            self.logger.info(f"Counter: {counter}")

            # direct return statement i.e. all processing is done?
            if counter[None] == len(label_uris.keys()):
                self.logger.info(f"Returning at {depth}")
                return labels

            # check what the count is and if the count is greater than 1, remove items that are of depth length (
            # those are duplicate)
            for k, v in counter.items():
                if v == 1:
                    # if the length is 1, this means it's a unique node, and we can keep it!
                    continue

                # get index of all the items with 'v' as value
                ids = [idx for idx, v in label_uris.items() if v == k]

                # if label_level idx is the same as depth and there are more than 1 labels, remove the duplicate
                # parent node
                idx_to_remove += [idx for idx in ids if label_level[idx] == depth]

            _new_labels = [label for i, label in enumerate(labels) if i not in idx_to_remove]

            self.logger.info(f"Prev label len : {len(labels)} new label len: {len(new_labels)}")

            return filter_label_list(labels=_new_labels, depth=new_depth)

        return filter_label_list(new_labels)

    def _single_query_generation(self, uri: str, labels: list[Label]):
        """
        This function generates the insert query in for the provided labels, and document.

        :return:
        """
        decision_obj = Decision(
            config=self.config.data_models,
            logger=self.logger,
            uri=uri,
            annotations=Annotation(
                config=self.config.data_models,
                logger=self.logger,
                date=datetime.now(),
                taxonomy=Taxonomy(
                    config=self.config.data_models,
                    logger=self.logger,
                    uri=self.dataset_builder.taxonomy.uri
                ),
                labels=labels,
                model=self.model_reference
            )
        )
        insert_query = decision_obj.insert_query

        return insert_query

    def _single_insert(self, query: str):

        self.logger.debug("Execute query: ")
        self.logger.info(query)
        self.request_handler.post2json(
            query=query,
            endpoint=EndpointType.DECISION
        )

    def execute(self):

        dataset = create_dataset(
            config=self.config,
            logger=self.logger,
            dataset=self.dataset_builder.train_dataset,
            taxonomy=self.dataset_builder.taxonomy
        )

        for record in tqdm(dataset, desc="processing docs"):
            text = record.get("text")
            decision_uri = record.get("decision_uri")
            print(f"URI: {decision_uri}")

            self.logger.info(f"Current text: {text}")
            predictions = self._single_prediction(text=text)
            labels = self._prediction_to_labels(predictions=predictions)
            query = self._single_query_generation(
                uri=decision_uri,
                labels=labels
            )
            self._single_insert(query)
            self.logger.info(f"Succesfully inserted predictions for: {decision_uri}")
