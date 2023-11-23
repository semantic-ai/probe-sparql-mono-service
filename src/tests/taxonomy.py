import unittest

from ..config import Config
from ..data_models import Taxonomy
from ..sparql import RequestHandler
from ..utils import LoggingBase


class TaxonomyTests(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.logger = LoggingBase(self.config.logging).logger
        self.request_handler = RequestHandler(config=self.config, logger=self.logger)

        # Existing decision to verify loading
        self.child_3 = self.user = Taxonomy(
            config=self.config.data_models,
            logger=self.logger,
            uri="ghent_words/3",
            label="label3",
            level=3
        )

        self.child_2 = self.user = Taxonomy(
            config=self.config.data_models,
            logger=self.logger,
            uri="ghent_words/2",
            label="label2",
            level=2,
            children=[self.child_3]
        )

        self.child_1 = Taxonomy(
            config=self.config.data_models,
            logger=self.logger,
            uri="ghent_words/1",
            label="label1",
            level=1,
            children=[self.child_2]
        )

        self.taxonomy = Taxonomy(
            config=self.config.data_models,
            logger=self.logger,
            uri="ghent_words_master_node",
            children=[self.child_1]
        )

        self.double_taxonomy = Taxonomy(
            config=self.config.data_models,
            logger=self.logger,
            uri="ghent_words_master_node",
            children=[self.child_1, self.child_1]
        )

    def test_taxonomy_uri(self):
        self.assertEqual(self.taxonomy.uri, "ghent_words_master_node")

    def test_taxonomy_label(self):
        self.assertEqual(self.taxonomy.label, None, "When label is not provided, none is expected for label default "
                                                    "value")
        self.assertEqual(self.taxonomy.children[0].label, "label1", "When label is provided, string return is expected")

    def test_taxonomy_level(self):
        self.assertEqual(self.taxonomy.level, 0, "When level is not provided, none is expected for level default "
                                                 "value")
        self.assertEqual(self.taxonomy.children[0].level, 1, "When level is provided, string return is expected")

    def test_taxonomy_children(self):
        self.assertTrue(isinstance(self.taxonomy.children, list))
        self.assertTrue(isinstance(self.taxonomy.children[0], Taxonomy))

    def test_taxonomy_todict_no_children(self):
        correct_dict_value = {
            'label': None,
            'level': 0,
            'uri': 'ghent_words_master_node',
            'children': []
        }
        self.assertEqual(correct_dict_value, self.taxonomy.todict())

    def test_taxonomy_todict_with_children(self):
        correct_dict_value = {
            'label': None,
            'level': 0,
            'uri': 'ghent_words_master_node',
            'children': [
                {
                    'label': 'label1',
                    'level': 1,
                    'uri': 'ghent_words/1',
                    'children': [
                        {
                            'label': 'label2',
                            'level': 2,
                            'uri': 'ghent_words/2',
                            'children': [
                                {
                                    'children': [],
                                    'label': 'label3',
                                    'level': 3,
                                    'uri': 'ghent_words/3'
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        self.assertEqual(self.taxonomy.todict(with_children=True), correct_dict_value)

    def test_taxonomy_todict_with_children_and_max_depth(self):
        # TODO: implement this
        correct_dict_value = {
            'children': [{'children': [], 'label': 'label1', 'level': 1, 'uri': 'ghent_words/1'}],
            'label': None,
            'level': 0,
            'uri': 'ghent_words_master_node'
        }

        self.assertEqual(correct_dict_value, self.taxonomy.todict(with_children=True, max_depth=2))

    def test_taxonomy_get_labels(self):
        correct_labels = ['label1', 'label2', 'label3']

        self.assertEqual(self.taxonomy.get_labels(), correct_labels)

    def test_taxonomy_get_labels_with_max_depth(self):
        correct_labels = ['label1', 'label2']

        self.assertEqual(self.taxonomy.get_labels(max_depth=2), correct_labels)

    def test_taxonomy_get_labels_with_max_depth_and_level(self):
        correct_labels = [('1', 'label1'), ('1.1', 'label2'), ('2', 'label1'), ('2.1', 'label2')]

        self.assertEqual(correct_labels, self.double_taxonomy.get_labels(max_depth=2, include_tree_indication=True))

    def test_taxonomy_remap_tree(self):
        pass

    def test_taxonomy_find(self):
        correct_result = {
            1: {
                'label': 'label1',
                'level': 1,
                'uri': 'ghent_words/1',
                'children': []
            },
            2: {
                'label': 'label2',
                'level': 2,
                'uri': 'ghent_words/2',
                'children': []
            },
            3: {
                'label': 'label3',
                'level': 3,
                'uri': 'ghent_words/3',
                'children': []
            }
        }

        self.assertEqual(self.taxonomy.find("ghent_words/3"), correct_result)

    def test_taxonomy_find_with_max_depth(self):
        correct_result = {
            1: {
                'label': 'label1',
                'level': 1,
                'uri': 'ghent_words/1',
                'children': []
            },
            2: {
                'label': 'label2',
                'level': 2,
                'uri': 'ghent_words/2',
                'children': []
            }
        }

        # TODO: fix this! -> for some reason it does not take the max depth param

        self.assertEqual(correct_result, self.taxonomy.find("ghent_words/2", max_depth=3))

    def test_taxonomy_find_unreachable_with_max_depth(self):
        correct_result = {}

        # TODO: fix this! -> for some reason it does not take the max depth param

        self.assertEqual(correct_result, self.taxonomy.find("ghent_words/3", max_depth=3))

    def test_taxonomy_find_not_exist(self):
        correct_result = {}

        # TODO: fix this! -> for some reason it does not take the max depth param

        self.assertEqual(self.taxonomy.find("ghent_words/20"), correct_result)

    def test_taxonomy_uri2label(self):
        correct_response = {
            'ghent_words/1': 'label1',
            'ghent_words/2': 'label2',
            'ghent_words/3': 'label3'
        }

        self.assertEqual(self.taxonomy.uri2label, correct_response)

    def test_taxonomy_label2uri(self):
        correct_response = {
            'label1': 'ghent_words/1',
            'label2': 'ghent_words/2',
            'label3': 'ghent_words/3'
        }

        self.assertEqual(self.taxonomy.label2uri, correct_response)

    def test_taxonomy_from_dict(self):
        example_dict = {
            'label': None,
            'level': 0,
            'uri': 'ghent_words_master_node',
            'children': [
                {
                    'label': 'label1',
                    'level': 1,
                    'uri': 'ghent_words/1',
                    'children': [
                        {
                            'label': 'label2',
                            'level': 2,
                            'uri': 'ghent_words/2',
                            'children': [
                                {
                                    'children': [],
                                    'label': 'label3',
                                    'level': 3,
                                    'uri': 'ghent_words/3'
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        taxo = Taxonomy.from_dict(
            config=self.config.data_models,
            logger=self.logger,
            dictionary=example_dict
        )
        self.assertEqual(taxo.todict(with_children=True), example_dict)

    def test_taxonomy_level_specific_labels(self):

        self.assertEqual(['label1'], self.taxonomy.get_level_specific_labels(level=1))
        self.assertEqual(['label2'], self.taxonomy.get_level_specific_labels(level=2))