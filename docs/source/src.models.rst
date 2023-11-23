Models
============

TOC
-----------

.. toctree::
   :maxdepth: 2

   src.models.classifier
   src.models.embedding
   src.models.hybrid
   src.models.zeroshot

Model overview
--------------

Supervised Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Supervised classification is the most common type of classification task. In supervised classification, the model is trained on a labeled dataset, where each data point is associated with a known class label. The model learns to map the input data points to their corresponding class labels.

Zero-Shot Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Zero-shot classification is a more challenging task, as the model is not given any labeled training data for the new classes it needs to classify. Instead, the model is trained on a large dataset of text or images, and it learns to represent these data points in a way that captures their semantic meaning. The model then uses this semantic understanding to classify new data points into unseen classes.

Embedding Similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~
Embedding similarity is a technique for measuring the similarity between two data points. In the context of classification, embedding similarity can be used to classify new data points by comparing them to known examples of each class. The data point is assigned to the class whose known example it is most similar to.


.. csv-table:: Comparison of the Three Approaches
   :file: _source/model_overview.csv
   :header-rows: 1

Base class
----------

.. automodule:: src.models.base
   :members:
   :undoc-members:
   :show-inheritance:

Other
---------------

.. automodule:: src.models
   :members:
   :undoc-members:
   :show-inheritance:
