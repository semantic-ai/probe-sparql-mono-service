# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os

sys.path.insert(0, os.path.abspath('../..'))

extensions = ['python']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sparql-mono'
copyright = '2023, Ruben Delaet'
author = 'Ruben Delaet'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.githubpages',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
]

autodoc_default_options = {
    'ignore-module-all': True,
    'autoclass_content': 'both',
    "private-members": True

}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

master_doc = "index"

nitpicky = True
nitpick_ignore = [
    # Ignore "parent class reference not found" errors for subclasses of ``object``
    ("py:class", "object"),
    ("py:class", "enum.Enum"),
    ("py:class", "bytes"),
    ("py:class", "bytearray"),
    # Suppress warnings for missing references in type annotations
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "matplotlib.figure.Figure"),
    ("py:class", "plotly.graph_objects.Figure"),
    ("py:class", "pathlib.Path"),
    ("py:class", "Path"),
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "ConfigDict"),
    ("py:class", "FieldInfo"),
    ("py:class", "DotenvType"),
    ("py:class", "Config"),
    ("py:class", "numpy.array"),
    ("py:class", "Logger"),
    ("py:class", "abc.ABC"),
    ("py:class", "unittest.case.TestCase"),
    ("py:class", "torch.Tensor"),
    ("py:class", "AutoTokenizer"),
    ("py:class", "device"),
    ("py:class", "torch.utils.data.dataset.Dataset"),
    ("py:class", "transformers.trainer.Trainer"),
    ("py:class", "torch.nn.modules.module.Module")


]
