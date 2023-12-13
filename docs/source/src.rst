Codebase overview
=================

Modules
*******

.. toctree::
   :maxdepth: 1

   src.benchmark
   src.config
   src.data_models
   src.dataset
   src.enums
   src.errors
   src.helpers
   src.models
   src.sparql
   src.tests
   src.training
   src.utils

Entrypoints
***********

In this section you can find all entrypoints scripts with their technical documentation.
These entrypoints are directly called from the airflow DAGs

Generate dataset statistics
---------------------------

.. automodule:: src.dataset_statistics
   :members:
   :undoc-members:
   :show-inheritance:

Export dataset
---------------

.. automodule:: src.dataset_export
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarking
-------------

.. automodule:: src.benchmarking
   :members:
   :undoc-members:
   :show-inheritance:


Train supervised model with specific taxonomy node
--------------------------------------------------

.. automodule:: src.train_supervised_specific
   :members:
   :undoc-members:
   :show-inheritance:


Create blank configs
---------------------

.. automodule:: src.create_blank_config
   :members:
   :undoc-members:
   :show-inheritance:


Train supervised model for entire taxonomy tree
-----------------------------------------------

.. automodule:: src.train_supervised_tree
   :members:
   :undoc-members:
   :show-inheritance:


Execute inference based on a provided model tree configuration
--------------------------------------------------------------

.. automodule:: src.topic_modeling
   :members:
   :undoc-members:
   :show-inheritance:
