.. sparql-mono documentation master file, created by
   sphinx-quickstart on Mon Nov 13 08:35:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################################
Welcome to sparql-mono's documentation!
#######################################

*************
Introduction
*************
In the era of Big Data, extracting meaningful insights from vast repositories of information is crucial for informed decision-making.
This project addresses the challenge of analyzing and utilizing annotation data stored in a SPARQL database.
It encompasses a comprehensive framework that integrates with SPARQL, extracts statistical information from the annotations,
performs benchmarking for various machine learning approaches, trains supervised models based on a taxonomy tree,
and executes inference using predefined configurations.

Key features:
#############
* **SPARQL Integration**: Seamlessly connects with SPARQL endpoints to retrieve and manipulate annotation data.

* **Annotation Statistics Extraction**: Provides detailed statistical insights into the annotation data, such as frequency distributions, co-occurrence patterns, and semantic relationships.

* **Benchmarking Suite**: Evaluates the performance of supervised, embedding, and zero-shot approaches for annotation tasks, enabling informed model selection.

* **Taxonomy-Driven Model Training**: Trains supervised models guided by a taxonomy tree, leveraging the hierarchical structure to capture fine-grained semantic relationships.

* **Configurable Inference**: Executes inference based on predefined configurations, allowing for the integration of various machine learning models and unsupervised approaches.

* **Model Prediction Population**: Populates new model predictions directly into the SPARQL database, enriching the annotation knowledge base.

To enable all this, we used:
############################
* **Airflow**, an opensource workflow management platform for data engineering pipelines. A more in depth review of how we utilize this can be found here
* **MlFlow**, another opensource platform for managing the end-to-end machine learning lifecycle. A more in depth review of how we utilize this can be found here
* **Annotation** tool, our own opensource UI created with react Admin. More information can be found under the UI section of the docs here

***************
Infrastructure
***************

Add infra schema and introduction here

Airflow
#######
Apache Airflow is an open-source workflow management platform designed to automate and manage workflows as directed acyclic graphs (DAGs). It is a popular tool for building and managing data pipelines, ETL/ELT processes, and other batch-oriented workflows.

General information about airflow
#################################

Key Features of Airflow
*********************

* **DAG-based workflow definition**: Airflow workflows are defined using DAGs, which provide a clear and concise way to represent the tasks in a workflow and their dependencies.

* **Scheduling and execution**: Airflow can schedule tasks to run at specific times or intervals, and it can also handle dependencies between tasks, ensuring that tasks are executed in the correct order.

* **Monitoring and alerts**: Airflow provides a web interface for monitoring the status of DAGs and tasks, and it can also send alerts when tasks fail or run unexpectedly.

* **Scalability**: Airflow is designed to be scalable, and it can be deployed on a single server or on a distributed cluster.

Benefits of Using Airflow
*********************

* **Increased productivity**: Airflow can automate many of the manual tasks involved in managing workflows, which can free up developers to focus on more creative work.

* **Improved reliability**: Airflow can help to ensure that workflows are executed reliably and consistently, which can reduce the risk of errors and data loss.

* **Reduced costs**: Airflow can help to reduce the cost of managing workflows by automating tasks and improving efficiency.

Use Cases for Airflow
*********************
Airflow is a versatile tool that can be used for a wide variety of workflows. Some common use cases include:

* **ETL/ELT data pipelines**: Airflow can be used to automate the process of extracting, transforming, and loading data from one system to another.

* **Batch processing**: Airflow can be used to automate batch processing tasks, such as data analysis, report generation, and file processing.

* **Machine learning pipelines**: Airflow can be used to automate the training and deployment of machine learning models.

* **Business process automation**: Airflow can be used to automate business processes, such as order processing and customer onboarding.

Entrypoints and DAGs in Airflow
*******************************
In Apache Airflow, entrypoints and DAGs (Directed Acyclic Graphs) are two closely related concepts that play a crucial role in managing and executing workflows.

Entrypoints
===========
A Python entrypoint is a mechanism for defining a function or other callable object that can be invoked from the command line when a package is installed.
This allows developers to create scripts that can be easily run by other users without requiring them to have any knowledge of the internal workings of the package.

Typically, an airflow component calls an entrypoint (or script) in order to execute the functionality that is provided via the recipe (DAG)


DAGs
====
A DAGs represent the workflows themselves, it defines a certain 'recipe'/workflow that should be executed when the dag is started.
They are defined using Python code and specify the tasks to be executed, their dependencies, and the schedule for running them.
Airflow parses DAG files and creates a DAG object for each DAG definition.

Relationship between Entrypoints and DAGs
=========================================
In essence, in this project entrypoints are a near one-on-one mapping for the provided functionality.
They serve as the gateway between the UI representing the DAGS and the backend that contains all processing logic in order to execute the specified task succesfully


Getting Started with Airflow
*********************
To get started with Airflow, you can follow the official installation guide: https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html

Additional Resources
*********************
Airflow Documentation: https://airflow.apache.org/docs/

Airflow Github: https://github.com/apache/airflow

Airflow Community Forum: https://forum.astronomer.io/

MLFLOW: An open-source platform for managing the machine learning lifecycle
###########################################################################

In the rapidly evolving field of machine learning, managing the entire machine learning lifecycle, from experimentation to deployment, is a complex and challenging task. This is where MLflow, an open-source platform, comes into play. MLflow streamlines the entire machine learning lifecycle by providing a comprehensive set of tools for tracking experiments, packaging models, deploying them, and managing them throughout their lifecycle.

Why Use MLflow?
***************

MLflow offers a compelling set of reasons to adopt it as your go-to machine learning platform:

* **Centralized Experiment Tracking**: MLflow keeps track of all your machine learning experiments, including parameters, metrics, and results, providing a centralized repository for experiment data.

* **Model Packaging and Versioning**: MLflow provides a standardized format for packaging machine learning models, ensuring reproducibility and simplifying model sharing.

* **Model Deployment and Serving**: MLflow streamlines model deployment to production environments, enabling you to serve your models to applications and systems.

* **Model Lifecycle Management**: MLflow manages the entire machine learning lifecycle, from experimentation to deployment, ensuring consistent and reliable model management practices.

Key Advantages of MLflow
*************************

* **Reproducibility**: MLflow's experiment tracking and model packaging capabilities ensure that experiments and models can be reproduced with ease.

* **Collaboration**: MLflow facilitates collaboration among data scientists by providing a shared platform for experiment tracking, model sharing, and model deployment.

* **Governance**: MLflow enables governance by providing a centralized repository for experiment data and models, promoting transparency and accountability.

* **Portability**: MLflow's standardized model format ensures that models can be seamlessly deployed across different platforms and environments.

* **Scalability**: MLflow's modular architecture and support for distributed environments make it scalable for large-scale machine learning projects.

In conclusion, MLflow stands as a powerful tool for managing the machine learning lifecycle, offering reproducibility, collaboration, governance, portability, and scalability. By adopting MLflow, data scientists and organizations can streamline their machine learning workflows, enhance collaboration, and accelerate the deployment of effective machine learning solutions.

Additional Resources
*********************
Mlflow Documentation: https://mlflow.org/docs/latest/index.html

Mlflow Github: https://github.com/mlflow/mlflow

Mlflow forum: https://github.com/mlflow/mlflow/issues


BeAT (Annotation UI)
####################


*************
Current setup
*************

In this section, you will find a more in depth overview on how each entrypoint is integrated with each-other.
These integrations will be visualized using screenshots, code-snippits and general information on what they can/should be used for.

<Add picture of airflow interface>

Benchmarking
#############
The benchmarking DAGs are like the name suggest used to benchmark certain types of models.
During development, we defined three model types that are interesting for this classification use-case:

#. **Supervised classification** is the most common type of classification task. In supervised classification, the model is trained on a labeled dataset, where each data point is associated with a known class label. The model learns to map the input data points to their corresponding class labels.

#. **Zero-shot classification** is a more challenging task, as the model is not given any labeled training data for the new classes it needs to classify. Instead, the model is trained on a large dataset of text or images, and it learns to represent these data points in a way that captures their semantic meaning. The model then uses this semantic understanding to classify new data points into unseen classes.

#. **Embedding similarity** is a technique for measuring the similarity between two data points. In the context of classification, embedding similarity can be used to classify new data points by comparing them to known examples of each class. The data point is assigned to the class whose known example it is most similar to.


.. csv-table:: Comparison of classification approaches
   :file: _source/model_overview.csv
   :header-rows: 1

Benchmarking Embeddings
*********************

Input
==================

Output
==================

Benchmark zeroshot
*********************

Input
==================

Output
==================

Helper
#############

Blank config
*********************

Input
==================

Output
==================

Dataset
#############

Dataset export
*********************

Input
==================

Output
==================

Dataset statistics
*********************

Input
==================

Output
==================

Inference
#############

Config based
*********************

Input
==================

Output
==================

Training
#############

Tree based
*********************

Input
==================

Output
==================

Node Based
*********************

Input
==================

Output
==================



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   src


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
