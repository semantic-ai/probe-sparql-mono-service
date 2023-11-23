import unittest

from ..data_models import Annotation, Taxonomy, Label, Model, User


class AnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        from ..sparql import RequestHandler
        from ..config import Config
        from ..utils import LoggingBase
        from datetime import datetime

        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        self.user_annotation = Annotation(
            date=1,
            config=self.config.data_models,
            logger=self.logger,
            taxonomy=Taxonomy(
                config=self.config.data_models,
                logger=self.logger,
                uri="<http://stad.gent/id/concepts/gent_words>"),
                labels=[
                    Label(
                        config=self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/3>"
                    ),
                    Label(
                        config=self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/148>")
                ],
            user=User(
                config=self.config.data_models,
                logger=self.logger,
                username="ruben",
                email="ruben@ml2grow.cloud"
            )
        )

        self.model_annotation = Annotation(
            date=1,
            config=self.config.data_models,
            logger=self.logger,
            taxonomy=Taxonomy(
                self.config.data_models,
                logger=self.logger,
                uri="<http://stad.gent/id/concepts/gent_words>"),
                labels=[
                    Label(
                        self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/3>",
                        score=0.71
                    ),
                    Label(
                        self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/148>",
                        score=0.55
                    )
                ],
            model=Model(
                self.config.data_models,
                logger=self.logger,
                name="test_model_name",
                mlflow_reference="http://mlflow.probe.ml2grow.cloud/models/test_model_name",
                category="zero-shot-model",
                registered_model="test_model_name"
            )
        )

        self.date_test_annotation = self.model_annotation
        self.date_test_annotation.date = datetime.now()

    def test_annotation_subquery(self):
        generated_subquery = self.user_annotation.subquery
        generated_subquery = " ".join(
            [q.lstrip().rstrip() for q in generated_subquery.split("\n") if not q.lstrip().rstrip() == ""])

        expected_subquery = "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:withTaxonomy " \
                            "<http://stad.gent/id/concepts/gent_words> . " \
                            "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:creationDate 1 ." \
                            " <https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:hasLabel <https://lblod.data.gift/concepts/ml2grow/label/test>, " \
                            "<https://lblod.data.gift/concepts/ml2grow/label/test> . " \
                            "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:withUser " \
                            "<https://classifications.ghent.com/ml2grow/user/test> . " \
                            "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                            "<https://stad.gent/id/concepts/gent_words/3> . " \
                            "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 1.0. " \
                            "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                            "<https://stad.gent/id/concepts/gent_words/148> . " \
                            "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 1.0."

        self.assertEqual(generated_subquery, expected_subquery)

    def test_annotation_label_uris(self):
        generated_label_uris = self.user_annotation.label_uris
        expected_label_uris = ['<https://stad.gent/id/concepts/gent_words/3>',
                               '<https://stad.gent/id/concepts/gent_words/148>']

        self.assertEqual(generated_label_uris, expected_label_uris)

    def test_annotation_label_type(self):
        self.assertTrue(isinstance(self.user_annotation.labels, list))

    def test_annotation_model(self):
        self.assertTrue(self.user_annotation.model is None, "Model is not None when not provided")
        self.assertTrue(isinstance(self.model_annotation.model, Model), "Model is not of type Model")

    def test_annotation_user(self):
        self.assertTrue(isinstance(self.user_annotation.user, User), "User is not of type User when filled in")
        self.assertTrue(self.user_annotation.model is None, "User is not None when not provided")

    def test_annotation_date(self):
        self.assertTrue(isinstance(self.user_annotation.date, int),
                        f"The date must be an integer value {self.user_annotation.date}")
        self.assertTrue(isinstance(self.date_test_annotation.date, int), "Datetime must be an integer value")

    def test_annotation_labels(self):
        self.assertTrue(isinstance(self.user_annotation.labels, list), "Labels must be of type list")
        self.assertTrue(isinstance(self.user_annotation.labels[0], Label), "Element of labels must be of type Label")

    def test_annotation_uri(self):
        self.assertEqual(self.user_annotation.uri, "<https://lblod.data.gift/concepts/ml2grow/annotations/test>",
                         "Generated uri does not match expected uri")

    def test_from_sparql(self):
        # todo: fix bug here and test
        annotation_obj = Annotation.from_sparql(
            config=self.config.data_models,
            logger=self.logger,
            request_handler=self.request_handler,
            annotation_uri="https://lblod.data.gift/concepts/ml2grow/annotations/4e7b7164-a5eb-4f19-8291-76da05a6e69b	"
        )

        self.assertEqual(annotation_obj.uri, "<https://lblod.data.gift/concepts/ml2grow/annotations/test>")
