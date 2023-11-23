import unittest

from ..data_models import Decision


class DecisionTests(unittest.TestCase):

    def setUp(self):
        from ..data_models import Annotation, Taxonomy, Label, User, Model, Article
        from ..sparql import RequestHandler
        from ..config import Config
        from ..utils import LoggingBase

        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        # Existing decision to verify loading
        self.user_decision = Decision(
            config=self.config.data_models,
            logger=self.logger,
            uri=r"https://data.gent.be/id/besluiten/69.6969.6969.6969",
            annotations=Annotation(
                date=1,
                config=self.config.data_models,
                logger=self.logger,
                taxonomy=Taxonomy(
                    config=self.config.data_models,
                    logger=self.logger,
                    uri="<http://stad.gent/id/concepts/gent_words>"
                ),
                labels=[
                    Label(
                        config=self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/3>"
                    ),
                    Label(
                        config=self.config.data_models,
                        logger=self.logger,
                        taxonomy_node_uri="<https://stad.gent/id/concepts/gent_words/148>"
                    )
                ],
                user=User(
                    config=self.config.data_models,
                    logger=self.logger,
                    username="ruben",
                    email="ruben@ml2grow.cloud")
            ),
            articles=[
                Article(
                    config=self.config.data_models,
                    logger=self.logger,
                    uri="",
                    number=1,
                    content="Eerste article"
                )
            ]
        )

        self.model_decision = Decision(
            config=self.config.data_models,
            logger=self.logger,
            annotations=Annotation(
                date=1,
                config=self.config.data_models,
                logger=self.logger,
                taxonomy=Taxonomy(
                    self.config.data_models,
                    logger=self.logger,
                    uri="<http://stad.gent/id/concepts/gent_words>"
                ),
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
                ),
            ),
            uri=r"https://data.gent.be/id/besluiten/69.6969.6969.6969"
        )

    def test_from_sparql_not_exists(self):
        decision = Decision.from_sparql(
            config=self.config.data_models,
            logger=self.logger,
            request_handler=self.request_handler,
            decision_uri="random_uri"
        )

        self.assertEqual(decision, None)

    def test_from_sparql_exists(self):
        self.decision = Decision.from_sparql(
            config=self.config.data_models,
            logger=self.logger,
            request_handler=self.request_handler,
            decision_uri="https://data.gent.be/id/besluiten/22.1123.3459.5994"
        )

        self.assertTrue(isinstance(self.decision, Decision))

    def test_decision_has_annotations(self):
        self.assertTrue(self.user_decision.annotations is not [])
        self.assertTrue(self.model_decision.annotations is not [])

    def test_insert_query_model(self):
        correct_query: str = "PREFIX ext: <http://mu.semte.ch/vocabularies/ext/> INSERT DATA { GRAPH " \
                             "<http://mu.semte.ch/application/probe/model-annotations> { " \
                             "<https://data.gent.be/id/besluiten/69.6969.6969.6969> ext:hasAnnotation " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:withTaxonomy " \
                             "<http://stad.gent/id/concepts/gent_words> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:creationDate 1 . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:hasLabel " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test>, " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test>    ext:withModel     " \
                             "<https://lblod.data.gift/concepts/ml2grow/model/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                             "<https://stad.gent/id/concepts/gent_words/3> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 0.71. " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                             "<https://stad.gent/id/concepts/gent_words/148> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 0.55. } }"

        query = self.model_decision.insert_query
        generated_query = (" ".join([q.lstrip().rstrip() for q in query.split("\n") if not q.lstrip().rstrip() == ""]))
        self.assertEqual(correct_query, generated_query)

    def test_insert_query_user(self):
        correct_query: str = "PREFIX ext: <http://mu.semte.ch/vocabularies/ext/> INSERT DATA { GRAPH " \
                             "<http://mu.semte.ch/application/probe/user-annotations> { " \
                             "<https://data.gent.be/id/besluiten/69.6969.6969.6969> ext:hasAnnotation " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:withTaxonomy " \
                             "<http://stad.gent/id/concepts/gent_words> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:creationDate 1 . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:hasLabel " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test>, " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/annotations/test> ext:withUser " \
                             "<https://classifications.ghent.com/ml2grow/user/test> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                             "<https://stad.gent/id/concepts/gent_words/3> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 1.0. " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:isTaxonomy " \
                             "<https://stad.gent/id/concepts/gent_words/148> . " \
                             "<https://lblod.data.gift/concepts/ml2grow/label/test> ext:hasScore 1.0. } }"

        query = self.user_decision.insert_query
        generated_query = (" ".join([q.lstrip().rstrip() for q in query.split("\n") if not q.lstrip().rstrip() == ""]))
        self.assertEqual(correct_query, generated_query)

    def test_last_human_annotation(self):
        from ..data_models import Annotation
        # TODO: add user annotations to test that aswel
        self.assertTrue(self.user_decision.last_human_annotation is not None, "User annotations are provided")
        self.assertTrue(self.model_decision.last_human_annotation is None, "Model has no user annotations")

    def test_article_list_empty(self):
        self.assertTrue(self.model_decision.article_list is None)

    def test_article_list_filled(self):
        self.assertEqual(self.user_decision.article_list, ["1: Eerste article"])

    def test_train_record_no_decision_info(self):
        correct_response = {
            'uri': 'https://data.gent.be/id/besluiten/69.6969.6969.6969',
            'uuid': None,
            'description': None,
            'articles': ['1: Eerste article'],
            'short_title': None,
            'language': None,
            'labels': [
                '<https://stad.gent/id/concepts/gent_words/3>',
                '<https://stad.gent/id/concepts/gent_words/148>'
            ]
        }

        self.assertEqual(self.user_decision.train_record, correct_response)
