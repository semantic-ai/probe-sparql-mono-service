from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.models import get_model, Model
    from src.sparql import RequestHandler
    from src.dataset import DatasetBuilder
    from logging import Logger

from src.data_models import TaxonomyTypes, Decision, Label, Annotation, Taxonomy, Model as data_model
from src.models import get_model
from src.dataset import create_dataset
from src.enums import EndpointType, TaxonomyFindTypes

from datetime import datetime
from uuid import uuid4
from tqdm import tqdm
import copy


class InferenceModel:
    """
    This class wraps some basic functionality with the model initializing



    """

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model: Model,
            model_config: dict
    ):
        self._label = None
        self._uri = None
        self._model_config = None

        self.config = config
        self.logger = logger
        self.model = model
        self.model_config = model_config

    def predict(self, text: str, **kwargs):
        self.logger.debug(f"Model prediction type: {type(self.model)}")
        return self.model.classify(text=text, multi_label=True, **kwargs)

    @classmethod
    def from_custom_config(
            cls,
            config: Config,
            logger: Logger,
            model_config: dict,
            taxonomy: Taxonomy
    ):
        _class = cls(
            config=config,
            logger=logger,
            model=get_model(
                config=config,
                logger=logger,
                taxonomy=taxonomy,
                model_id=model_config.get("model_id"),
                specific_model_type=model_config.get("flavour"),
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
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def model_config(self):
        return self._model_config

    @model_config.setter
    def model_config(self, value: dict):
        self._model_config = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value.lower()

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value):
        self._uri = value


class InferenceModelTree:

    def __init__(
            self,
            config: Config,
            logger: Logger,
            request_handler: RequestHandler,
            model_configuration: dict,
            models: dict[str, InferenceModel],
            taxonomy: Taxonomy
    ):
        self.config = config
        self.logger = logger
        self.request_handler = request_handler
        self.model_config = model_configuration
        self.models = models
        self.taxonomy = taxonomy

    def predict(self, text: str):
        labels = {}
        # start with prediction of parent model
        prediction = self.models["main"].predict(text=text)

        # easy solution for recursive predictions
        def sub_predict(pred: dict, m_conf: dict):
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
    ):

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

        def load_models(model_config: dict):
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
            dataset_type: str
    ):
        self.config = config
        self.config.run.dataset.type = dataset_type

        self.logger = logger
        self.request_handler = request_handler
        self.dataset_builder = dataset_builder
        self.inference_model = inference_model_tree

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
        new_labels = []

        for label, score in predictions.items():
            label_uri = self.label2id.get(label)
            find_out = self.dataset_builder.taxonomy.find(
                search_term=label,
                search_kind=TaxonomyFindTypes.LABEL
            )

            self.logger.debug(f"Found uri for label {label} -> {label_uri}")
            self.logger.debug(f"Find output: {find_out}")

            if label_uri is None:
                self.logger.warning(f"Label skipped ({label}) because uri not found")

            new_labels.append(
                Label(
                    config=self.config.data_models,
                    logger=self.logger,
                    score=score,
                    taxonomy_node_uri=label_uri
                )
            )

        return new_labels

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
                model=data_model(
                    config=self.config.data_models,
                    logger=self.logger,
                    uri=self.config.data_models.model.uri_base + "custom_configuration/" + str(uuid4().hex),
                    category="custom compound model",
                    registered_model=f"Config: self.inference_model.model_config",
                    register=False

                )
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
